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
from pathlib import Path

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler,
    StableDiffusionPipeline
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig

from config import TrainingConfig
from PolypDataset import PolypDataset
from PolypDiffusionDataset import PolypDiffusionDataset

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("diffusion_pretrained")

def save_lora_weights(unet, save_path):
    os.makedirs(save_path, exist_ok=True)
    lora_params = {k: v.cpu() 
                   for k, v in unet.state_dict().items() 
                   if "lora_" in k}
    torch.save(lora_params, os.path.join(save_path, "lora_weights.pth"))

def load_lora_weights(device, unet, path):
    weights = torch.load(os.path.join(path, "lora_weights.pth"), map_location=device)
    unet.load_state_dict(weights, strict=False)
    
def load_pipeline(path_model, device):
    pipe = StableDiffusionPipeline.from_pretrained(
        path_model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False)
    pipe.to(device)
    
    return pipe
    

def log_sample_images(dir_path, cls, num_samples=5):
    files = [f for f in os.listdir(dir_path) if f.endswith(".png")]
    samples = random.sample(files, min(num_samples, len(files)))
    
    for file in samples:
        mlflow.log_artifact(os.path.join(dir_path, file), artifact_path=f"samples/{cls}")


def evaluate(config, pipeline, cls, prompt, num_images, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pipeline.unet.eval()
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    if pipeline.safety_checker is not None:
        pipeline.safety_checker.eval()



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
    
        print(Fore.CYAN + f"Generated {total}/{num_images} images" +  Style.RESET_ALL)

    log_sample_images(out_dir, cls, num_samples=10)


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



def train_loop(device, config, unet, vae, text_encoder, tokenizer, noise_scheduler, optimizer, dataloader, lr_scheduler, prompt, cls, imgs_to_generate, train_text_encoder=False, latent_to_text_proj=None):
    unet.train()
    if train_text_encoder:
        text_encoder.train()
    
    loss_hist = []
    
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            pixel_values = batch[0].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            text_input = tokenizer(
                prompt * latents.shape[0], 
                padding="max_length", 
                max_length=77, 
                return_tensors="pt"
                )
            if train_text_encoder:
                encoder_hidden_states = text_encoder(text_input.input_ids.to(device))[0]
            else:
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(text_input.input_ids.to(device))[0]

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred, noise)

            if latent_to_text_proj:
                text_emb_pooled = encoder_hidden_states.mean(dim=1)
                latent_pooled = latents.mean(dim=[2, 3])
                latent_projected = latent_to_text_proj(latent_pooled)
                loss_text = 1.0 - F.cosine_similarity(text_emb_pooled, latent_projected, dim=-1).mean()
                loss += (config.weight_img * loss + config.weight_text * loss_text) / accumulation_steps


            loss.backward()
            
            grad_params = list(unet.parameters())
            if train_text_encoder:
                grad_params += list(text_encoder.parameters())
            if latent_to_text_proj:
                grad_params += list(latent_to_text_proj.parameters())

            torch.nn.utils.clip_grad_norm_(grad_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_hist.append(avg_loss)
        print(Fore.YELLOW + f"[{cls}] Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

        if epoch == config.num_epochs - 1:
            save_path = os.path.join(config.output_dir, f"lora_{cls}")
            save_lora_weights(unet, save_path)

            pipe = StableDiffusionPipeline(
                vae=vae, 
                text_encoder=text_encoder, 
                tokenizer=tokenizer,
                unet=unet, 
                scheduler=noise_scheduler,
                safety_checker=None, 
                feature_extractor=None
            ).to(device)

            pipe.unet.cpu()
            pipe.vae.cpu()
            pipe.text_encoder.cpu()
            if pipe.safety_checker is not None:
                pipe.safety_checker.cpu()
            pipe.save_pretrained(os.path.join(config.output_dir, f"model_{cls}"))

            mlflow.log_artifact(os.path.join(config.output_dir, f"model_{cls}"), f"models/{cls}")
            mlflow.log_artifact(save_path, f"models/lora_{cls}")
    
    plot_path = plot_loss(loss_hist, config.output_dir, cls)
    mlflow.log_artifact(plot_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--classes_to_train", nargs='+', type=str, required=True) # AD HP ASS
    parser.add_argument("--num_imgs_to_generate", nargs='+', type=int, required=True) # 465 619 628
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument('--unconditional', action='store_true')
    parser.add_argument('--class_condition', action='store_true')
    parser.add_argument('--train_text_encoder', action='store_true')
    parser.add_argument('--dreambooth', action='store_true')
    parser.add_argument('--add_visual_influence', action='store_true')
    parser.add_argument('--unfreeze_layers', action='store_true')
    args = parser.parse_args()
    
    config = TrainingConfig()
    if os.path.exists(args.folder):
        print(Fore.MAGENTA + os.path.basename(args.folder) + Style.RESET_ALL)
    else:
        print(Fore.RED + "No folder exists" + Style.RESET_ALL)
        import sys
        sys.exit()
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    class_map = {}
    acronyms_to_words = {
        "AD": ("adenomatous", "sks"),
        "HP": ("hyperplastic", "zbt"),
        "ASS": ("sessile serrated", "mjt"),
        "REST": ("hyperplastic and sessile serrated", "zbt")
    }
    num_imgs_to_generate = {}
    
    for class_to_train, n_imgs in zip(args.classes_to_train, args.num_imgs_to_generate):
        if class_to_train != 'REST':
            class_map[class_to_train] = list(class_to_train)
        else:
            class_map["REST"] =  ['HP', 'ASS']
        
        num_imgs_to_generate[class_to_train] = n_imgs
            

    with mlflow.start_run(run_id=args.run_id):
        for cls, n_imgs_to_generate in zip(args.classes_to_train, args.num_imgs_to_generate):
            
            # If a model already exists (lora_cls and model_cls folders)
            if f"lora_{cls}" in os.listdir(args.folder) and f"model_{cls}" in os.listdir(args.folder):
                print(Fore.MAGENTA + f"Model for {cls} class already trained" + Style.RESET_ALL)
                
                if args.unconditional:
                    prompt = [""]
                else:
                    # prompt = [f"a high-resolution endoscopic photo of {words_to_special_tokens[cls]} {acronyms_to_words[cls][0]} polyp"]
                    # prompt = [f"an endoscopic image showing a {words_to_special_tokens[cls]} {acronyms_to_words[cls][0]} polyp inside the colon"]
                    prompt = [f"a realistic high-resolution medical endoscopy image of {acronyms_to_words[cls][1]} {acronyms_to_words[cls][0]} polyp"]
                        
                
                # If samples have been generated
                if os.path.exists(os.path.join(args.folder, "samples", cls)):
                    samples_path = Path(os.path.join(args.folder, "samples", cls))
                    n_files = sum(1 for f in samples_path.iterdir() if f.is_file())
                    
                    # If are less images than the target images to generate and generate the rest
                    if n_files < n_imgs_to_generate:
                        target_images = n_imgs_to_generate - n_files
                        
                        pipe = load_pipeline(os.path.join(args.folder, f"model_{cls}"),
                                             device)
                        
                        load_lora_weights(device, pipe.unet, os.path.join(args.folder, f"lora_{cls}"))
                        evaluate(config, pipe, cls, prompt, target_images, 
                                 out_dir=os.path.join(args.folder, "samples", cls))
                        
                        print(Fore.GREEN + f"Generated {target_images} images for class {cls} successfully!\n" + Style.RESET_ALL)
                
                # If model has been trained, but no images have been generated
                else:
                    pipe = load_pipeline(os.path.join(args.folder, f"model_{cls}"),
                                             device)
                        
                    load_lora_weights(device, pipe.unet, os.path.join(args.folder, f"lora_{cls}"))
                    evaluate(config, pipe, cls, prompt, n_imgs_to_generate, 
                                out_dir=os.path.join(args.folder, "samples", cls))

                    print(Fore.GREEN + f"Generated {n_imgs_to_generate} images for class {cls} successfully!\n" + Style.RESET_ALL)
            
            # If a model has not been trained at all
            else:
                print(Fore.MAGENTA + f"Model for {cls} class not trained" + Style.RESET_ALL)
                dataset = PolypDiffusionDataset(image_dirs=["../data/m_train2/m_train/images", "../data/m_valid/m_valid/images"],
                                        csv_files=["../data/m_train2/m_train/train.csv", "../data/m_valid/m_valid/valid.csv"],
                                        transformations=True,
                                        keep_one_class=class_map[cls])
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
                    special_token = acronyms_to_words[cls][1] # get special token
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
                            class_phrase = acronyms_to_words[cls][0]  # e.g., "sessile serrated"
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
                            prompt = [f"{acronyms_to_words[cls][1]} {cls}"]
                            prompt = [f"{cls}"]
                    else:
                        if args.dreambooth:
                            prompt = [f"a high-resolution endoscopic photo of {words_to_special_tokens[cls]} {acronyms_to_words[cls][0]} polyp"]
                            # prompt = [f"an endoscopic image showing a {words_to_special_tokens[cls]} {acronyms_to_words[cls][0]} polyp inside the colon"]
                            # prompt = [f"a realistic high-resolution medical endoscopy image of {acronyms_to_words[cls][1]} {acronyms_to_words[cls][0]} polyp"]
                        else:
                            prompt = [f"a high-resolution endoscopic photo of {acronyms_to_words[cls][0]} polyp"]
                            # prompt = [f"an endoscopic image showing a {acronyms_to_words[cls][0]} polyp inside the colon"]
                            # prompt = [f"a realistic high-resolution medical endoscopy image of {acronyms_to_words[cls][0]} polyp"]
                    print(f"Prompt: {prompt}")

                mlflow.log_param(f"prompt_{cls}", prompt[0])
                print(Fore.YELLOW + "\nStarting training...")
                train_loop(device, config, unet, vae, text_encoder, tokenizer, noise_scheduler, optimizer, dataloader, lr_scheduler, prompt, cls, n_imgs_to_generate, train_text_encoder=args.train_text_encoder)
                print(Fore.GREEN + f"Training for class {cls} finished and images generated successfully\n" + Style.RESET_ALL)



if __name__ == "__main__":
    main()

