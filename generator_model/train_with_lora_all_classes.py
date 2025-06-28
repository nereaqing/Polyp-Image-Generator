import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import mlflow
import argparse
from colorama import Fore, Style
import matplotlib.pyplot as plt
import math
import shutil

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler,
    StableDiffusionPipeline
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig

from generator_model.config_diffusion import TrainingConfig
from PolypDataset import PolypDataset
from PolypDiffusionDataset import PolypDiffusionDataset

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def save_lora_weights(unet, save_path):
    os.makedirs(save_path, exist_ok=True)
    lora_params = {k: v.cpu() 
                   for k, v in unet.state_dict().items() 
                   if "lora_" in k}
    torch.save(lora_params, os.path.join(save_path, "lora_weights.pth"))

def load_lora_weights(device, unet, path):
    weights = torch.load(os.path.join(path, "lora_weights.pth"), map_location=device)
    unet.load_state_dict(weights, strict=False)

def log_sample_images(dir_path, cls, num_samples=5):
    files = [f for f in os.listdir(dir_path) if f.endswith(".png")]
    samples = random.sample(files, min(num_samples, len(files)))
    
    for file in samples:
        mlflow.log_artifact(os.path.join(dir_path, file), artifact_path=f"samples/{cls}")


def evaluate(config, pipeline, cls, prompt, num_images):
    out_dir = os.path.join(config.output_dir, "samples", cls)
    os.makedirs(out_dir, exist_ok=True)

    total = 0
    batch_id = 0
    while total < num_images:
        bs = min(config.eval_batch_size, num_images - total)
        images = pipeline(prompt * bs, 
                          height=config.image_size, 
                          width=config.image_size,
                          num_inference_steps=25, 
                          guidance_scale=7.5,
                          generator=torch.Generator(device='cpu').manual_seed(config.seed + batch_id)).images
        
        for i, img in enumerate(images):
            img.save(os.path.join(out_dir, f"{total + i + 1}.png"))
        
        total += len(images)
        batch_id += 1

    log_sample_images(out_dir, cls, num_samples=10)


def counts_per_class():
    df = pd.read_csv("../data/m_train2/m_train/train.csv")
    return df['cls'].value_counts().to_dict()


def get_num_images_to_generate(distribution, ad_minimum=1000, one_vs_rest=False):
    real_counts = counts_per_class()
    ad_target = max(real_counts['AD'], ad_minimum)
    total = int(ad_target / distribution[0])

    if one_vs_rest:
        rest_count = real_counts['HP'] + real_counts['ASS']
        rest_target = int(total * distribution[1])
        synthetic = {'AD': max(0, ad_target - real_counts['AD']),
                     'REST': max(0, rest_target - rest_count)}
    else:
        synthetic = {
            'AD': max(0, ad_target - real_counts['AD']),
            'HP': max(0, int(total * distribution[1]) - real_counts['HP']),
            'ASS': max(0, int(total * distribution[2]) - real_counts['ASS'])
        }

    print(Fore.CYAN + "Images to generate:", synthetic)
    return synthetic

def plot_loss(losses, output_dir, cls):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='blue')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plot_path = f"{output_dir}/loss_history_{cls}.png"
    plt.savefig(plot_path)
    
    return plot_path



def train_loop(device, config, unet, vae, text_encoder, tokenizer, noise_scheduler,
               optimizer, dataloader, lr_scheduler, prompt, cls, imgs_to_generate,
               train_text_encoder=False, latent_to_text_proj=None):

    unet.train()
    if train_text_encoder:
        text_encoder.train()

    loss_hist = []
    accumulation_steps = config.accumulation_steps

    for epoch in range(config.num_epochs):
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            pixel_values = batch[0].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            text_input = tokenizer(
                prompt * latents.shape[0],
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            ).to(device)

            if train_text_encoder:
                encoder_hidden_states = text_encoder(text_input.input_ids)[0]
            else:
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(text_input.input_ids)[0]

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred, noise) / accumulation_steps

            if latent_to_text_proj:
                text_emb_pooled = encoder_hidden_states.mean(dim=1)
                latent_pooled = latents.mean(dim=[2, 3])
                latent_projected = latent_to_text_proj(latent_pooled)
                loss_text = 1.0 - F.cosine_similarity(text_emb_pooled, latent_projected, dim=-1).mean()
                loss += (config.weight_img * loss + config.weight_text * loss_text) / accumulation_steps

            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                grad_params = list(unet.parameters())
                if train_text_encoder:
                    grad_params += list(text_encoder.parameters())
                if latent_to_text_proj:
                    grad_params += list(latent_to_text_proj.parameters())

                torch.nn.utils.clip_grad_norm_(grad_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            total_loss += loss.item() * accumulation_steps  # undo scaling for logging

        avg_loss = total_loss / len(dataloader)
        loss_hist.append(avg_loss)
        print(Fore.YELLOW + f"[{cls}] Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}")

        if epoch == config.num_epochs - 1:
            save_path = os.path.join(config.output_dir, f"lora_{cls}")
            save_lora_weights(unet, save_path)

            pipe = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=noise_scheduler,
                safety_checker=None,  # explained below
                feature_extractor=None
            ).to(device)

            load_lora_weights(device, pipe.unet, save_path)
            evaluate(config, pipe, cls, prompt, imgs_to_generate)

            pipe.unet.cpu()
            pipe.vae.cpu()
            pipe.text_encoder.cpu()
            if pipe.safety_checker is not None:
                pipe.safety_checker.cpu()
            pipe.save_pretrained(os.path.join(config.output_dir, f"model_{cls}"))

            mlflow.log_artifact(os.path.join(config.output_dir, f"model_{cls}"), f"models/{cls}")
            mlflow.log_artifact(save_path, f"models/lora_{cls}")
            
            print(Fore.GREEN + "Models sucessfully logged onto MLFlow" + Style.RESET_ALL)
            
            shutil.rmtree(save_path) # remove lora weights
            shutil.rmtree(os.path.join(config.output_dir, f"model_{cls}")) # remove diffusion model pipeline
            print(Fore.RED + "Models removed from local computer" + Style.RESET_ALL)

    plot_path = plot_loss(loss_hist, config.output_dir, cls)
    mlflow.log_artifact(plot_path)



def main():
    print("Starting")
    parser = argparse.ArgumentParser()
    parser.add_argument('--one_vs_rest', action='store_true', help='If provided, it will be performed AD vs REST')
    parser.add_argument('--unconditional', action='store_true')
    parser.add_argument('--class_condition', action='store_true')
    parser.add_argument('--train_text_encoder', action='store_true')
    parser.add_argument('--dreambooth', action='store_true')
    parser.add_argument('--add_visual_influence', action='store_true')
    parser.add_argument('--unfreeze_layers', action='store_true')
    parser.add_argument('--generate_subsamples', action='store_true')
    args = parser.parse_args()
    
    config = TrainingConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    print(Fore.GREEN + os.path.basename(config.output_dir) + Style.RESET_ALL)
    mlflow.set_experiment(config.experiment_name)
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    if args.one_vs_rest:
        classes = ['AD', 'REST']
        class_map = {
            'AD': ['AD'],
            'REST': ['HP', 'ASS']
        }
        acronyms_to_words = {
            'AD': 'adenomatous',
            'REST': 'hyperplastic and sessile serrated'
        }
        if args.dreambooth:
            words_to_special_tokens = {
                'AD': 'sks',
                'REST': 'zbt'
            }
    
    else:
        classes = ['AD', 'HP', 'ASS']
        class_map = {
            'AD': ['AD'], 
            'HP': ['HP'],
            'ASS': ['ASS']
        }
        
        acronyms_to_words = {
            'AD': 'adenomatous',
            'HP': 'hyperplastic',
            'ASS': 'sessile serrated'
        }
        if args.dreambooth:
            words_to_special_tokens = {
                'AD': 'sks',
                'HP': 'zbt',
                'ASS serrated': 'mjt'
            }

    with mlflow.start_run(run_name=os.path.basename(config.output_dir)):
        if args.generate_subsamples:
            num_imgs_to_generate = {
                'AD': 5,
                'HP': 5,
                'ASS': 5
            }
        
        else:
            if args.one_vs_rest:
                mlflow.log_param("technique", "AD vs REST")
                percentage_distribution = (0.6, 0.4)
                num_imgs_to_generate = get_num_images_to_generate(distribution=percentage_distribution, 
                                                                ad_minimum=1000, 
                                                                one_vs_rest=args.one_vs_rest)
            else:
                percentage_distribution = (0.4, 0.3, 0.3)
                num_imgs_to_generate = get_num_images_to_generate(distribution=percentage_distribution, 
                                                                ad_minimum=1000, 
                                                                one_vs_rest=args.one_vs_rest)

         
            mlflow.log_param("percentage_image_distribution", percentage_distribution)
        mlflow.log_param("images_to_generate_per_class", num_imgs_to_generate)
    
        for cls in classes:
            keep_classes = class_map[cls]
            dataset = PolypDiffusionDataset(image_dirs=["../data/m_train2/m_train/images", "../data/m_valid/m_valid/images"],
                                    csv_files=["../data/m_train2/m_train/train.csv", "../data/m_valid/m_valid/valid.csv"],
                                    transformations=True,
                                    keep_one_class=keep_classes)
            dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
 
            vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
            tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder").to(device)
            unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(device)
            noise_scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_rank,
                target_modules=config.modules_lora,
                lora_dropout=config.lora_dropout,
                init_lora_weights="gaussian"
            )
            
            
            if args.unfreeze_layers:
                for name, param in unet.named_parameters():
                    if any(x in name for x in ["to_q", "to_k", "to_v", "to_out.0"]):
                        param.requires_grad = True

            unet.add_adapter(lora_config)
            trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in unet.parameters())
            print(Fore.CYAN + f"Trainable params of unet: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")

            
            num_train_timesteps = len(dataloader) * config.num_epochs
            lr_warmup_steps = int(0.03 * num_train_timesteps)  # ~3% of training steps
            config.num_train_timesteps = num_train_timesteps
            config.lr_warmup_steps = lr_warmup_steps
            print(f"Number of timesteps: {config.num_train_timesteps}\nWarmup steps: {config.lr_warmup_steps}")

            
            params = {
                    "transformations": dataset.transformations_list,
                    "criterion": "MSELoss",
                    "optimizer": "AdamW",
                    "batch_size": config.train_batch_size,
                    "learning_rate": config.learning_rate,
                    "num_epohcs": config.num_epochs,
                    "image_size": config.image_size,
                    f"train_timesteps_{cls}": config.num_train_timesteps,
                    f"lr_warmup_steps_{cls}": config.lr_warmup_steps,
                    "noise_scheduler": "UniPCMultistepScheduler",
                    "lora_rank": config.lora_rank,
                    "lora_alpha": config.lora_rank,
                    "target_models_lora": config.modules_lora
                }
            
            
            if args.add_visual_influence:
                params["weight_image"] = config.weight_img
                params["weight_text"] = config.weight_text
            
            if args.dreambooth:
                params["weight_token_class"] = config.weight_token_class
                params["weight_token_polyp"] = config.weight_token_polyp
            
            mlflow.log_params(params)
            
            if args.dreambooth:
                special_token = words_to_special_tokens[cls]
                tokenizer.add_tokens([special_token])
                text_encoder.resize_token_embeddings(len(tokenizer))
                
                
                with torch.no_grad():
                    # Get token IDs
                    special_token_id = tokenizer.convert_tokens_to_ids(special_token)
                    polyp_token_id = tokenizer.convert_tokens_to_ids("polyp")
                    polyp_embedding = text_encoder.get_input_embeddings().weight[polyp_token_id]

                    if args.class_condition:
                        token_ids = tokenizer.convert_to_ids(cls)
                    else:
                        # Get class phrase (e.g., "sessile serrated")
                        class_phrase = acronyms_to_words[cls]  # e.g., "sessile serrated"
                        tokens = tokenizer.tokenize(class_phrase)
                        print("Tokens:", tokens)
                        token_ids = tokenizer.convert_tokens_to_ids(tokens)

                        embeddings = text_encoder.get_input_embeddings().weight
                        class_embeddings = embeddings[token_ids]  # shape: (n_tokens, embedding_dim)

                        class_avg = class_embeddings.mean(dim=0)
                        final_embedding = config.weight_token_class * class_avg + config.weight_token_polyp * polyp_embedding
                        text_encoder.get_input_embeddings().weight[special_token_id] = final_embedding


                original_embeds = text_encoder.get_input_embeddings().weight

                @torch.no_grad()
                def freeze_all_but_szk(grad):
                    mask = torch.zeros_like(grad)
                    mask[special_token_id] = 1.0
                    return grad * mask

                original_embeds.register_hook(freeze_all_but_szk)


            if args.train_text_encoder:
                text_lora_config = LoraConfig(
                    r=config.lora_rank,
                    lora_alpha=config.lora_rank,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
                )
                text_encoder.add_adapter(text_lora_config)
                trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in unet.parameters())
                print(f"Trainable params of text encoder: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")

            
                optimizer = torch.optim.AdamW(
                    filter(
                        lambda p: p.requires_grad, 
                        list(unet.parameters()) + list(text_encoder.parameters())
                    ), 
                    lr=config.learning_rate
                )
                if args.add_visual_influence:
                    proj_latent_to_text = torch.nn.Linear(4, 768).to(device)
                    optimizer = torch.optim.AdamW(
                        list(unet.parameters()) + list(text_encoder.parameters()) + list(proj_latent_to_text.parameters()),
                        lr=config.learning_rate
                    )

            else:
                optimizer = torch.optim.AdamW(
                    filter(
                        lambda p: p.requires_grad, 
                        list(unet.parameters())
                    ), 
                    lr=config.learning_rate
                )

            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.lr_warmup_steps,
                num_training_steps=config.num_train_timesteps
            )

            if args.unconditional:
                prompt = [""]
                print(f"Prompt: empty")
            else:
                if args.class_condition:
                    if args.dreambooth:
                        prompt = [f"{words_to_special_tokens[cls]} {cls}"]
                    else:
                        prompt = [f"{cls}"]
                else:
                    if args.dreambooth:
                        prompt = [f"a high-resolution endoscopic photo of {words_to_special_tokens[cls]} {acronyms_to_words[cls]} polyp"]
                        # prompt = [f"an endoscopic image showing a {words_to_special_tokens[cls]} {acronyms_to_words[cls]} polyp inside the colon"]
                        # prompt = [f"a realistic high-resolution medical endoscopy image of {words_to_special_tokens[cls]} {acronyms_to_words[cls]} polyp"]
                    else:
                        prompt = [f"a high-resolution endoscopic photo of {acronyms_to_words[cls]} polyp"]
                        # prompt = [f"an endoscopic image showing a {acronyms_to_words[cls]} polyp inside the colon"]
                        # prompt = [f"a realistic high-resolution medical endoscopy image of {acronyms_to_words[cls]} polyp"]
                print(f"Prompt: {prompt}")

            mlflow.log_param(f"prompt_{cls}", prompt[0])
            print(Fore.YELLOW + "\nStarting training...")
            train_loop(device, config, unet, vae, text_encoder, tokenizer, noise_scheduler, optimizer, dataloader, lr_scheduler, prompt, cls, num_imgs_to_generate[cls], train_text_encoder=args.train_text_encoder)
            print(Fore.GREEN + f"Training for class {cls} finished successfully\n" + Style.RESET_ALL)



if __name__ == "__main__":
    main()

