import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import mlflow
import argparse
from colorama import Fore

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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def save_lora_weights(unet, save_path):
    os.makedirs(save_path, exist_ok=True)
    lora_params = {k: v.cpu() 
                   for k, v in unet.state_dict().items() 
                   if "lora_" in k}
    torch.save(lora_params, os.path.join(save_path, "lora_weights.pth"))

def load_lora_weights(unet, path):
    weights = torch.load(os.path.join(path, "lora_weights.pth"), map_location=device)
    unet.load_state_dict(weights, strict=False)

def log_sample_images(dir_path, cls, epoch, num_samples=5):
    files = [f for f in os.listdir(dir_path) if f.endswith(".png")]
    samples = random.sample(files, min(num_samples, len(files)))
    
    for file in samples:
        mlflow.log_artifact(os.path.join(dir_path, file), artifact_path=f"samples/{cls}/{epoch:04d}")


def evaluate(config, epoch, pipeline, cls, prompt, num_images):
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

    log_sample_images(out_dir, cls, epoch, num_samples=10)


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


def train_loop(config, unet, vae, text_encoder, tokenizer, noise_scheduler, optimizer, dataloader, lr_scheduler, prompt, cls, imgs_to_generate):
    unet.train()
    
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
            with torch.no_grad():
                encoder_hidden_states = text_encoder(text_input.input_ids.to(device))[0]

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred, noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
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

            load_lora_weights(pipe.unet, save_path)
            evaluate(config, epoch, pipe, cls, prompt, imgs_to_generate)
            
            pipe.save_pretrained(os.path.join(config.output_dir, f"model_{cls}"))
            mlflow.log_artifact(os.path.join(config.output_dir, f"model_{cls}"), f"models/{cls}")


def main():
    print("Starting")
    parser = argparse.ArgumentParser()
    parser.add_argument('--one_vs_rest', action='store_true', help='If provided, it will be performed AD vs REST')
    args = parser.parse_args()
    
    config = TrainingConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    print(Fore.CYAN + os.path.basename(config.output_dir))
    
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

    with mlflow.start_run(run_name=os.path.basename(config.output_dir)):
        if args.one_vs_rest:
            mlflow.log_param("technique", "AD vs REST")
            percentage_distribution = (0.6, 0.4)
            num_imgs_to_generate = get_num_images_to_generate(distribution=percentage_distribution, 
                                                              ad_minimum=700, 
                                                              one_vs_rest=args.one_vs_rest)
        else:
            percentage_distribution = (0.4, 0.3, 0.3)
            num_imgs_to_generate = get_num_images_to_generate(distribution=percentage_distribution, 
                                                              ad_minimum=500, 
                                                              one_vs_rest=args.one_vs_rest)
            
        mlflow.log_param("images_to_generate_per_class", num_imgs_to_generate)
        mlflow.log_param("percentage_image_distribution", percentage_distribution)
    
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
                r=4,
                lora_alpha=4,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                init_lora_weights="gaussian"
            )
            unet.add_adapter(lora_config)
            trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in unet.parameters())
            print(f"Trainable params: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")

            params = {
                    "transformations": dataset.transformations_list,
                    "criterion": "MSELoss",
                    "optimizer": "AdamW",
                    "batch_size": config.train_batch_size,
                    "learning_rate": config.learning_rate,
                    "num_epohcs": config.num_epochs,
                    "image_size": config.image_size,
                    "train_timesteps": config.num_train_timesteps,
                    "noise_scheduler": "UniPCMultistepScheduler"
                }
            
            mlflow.log_params(params)

            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet.parameters()), lr=config.learning_rate)
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.lr_warmup_steps,
                num_training_steps=(len(dataloader) * config.num_epochs)
            )

            prompt = [f"a high-resolution endoscopic image of {acronyms_to_words[cls]} polyp"]
            mlflow.log_param(f"prompt", prompt[0])
            print(Fore.CYAN + prompt[0])
            print(Fore.YELLOW + "\nStarting training...")
            train_loop(config, unet, vae, text_encoder, tokenizer, noise_scheduler, optimizer, dataloader, lr_scheduler, prompt, cls, num_imgs_to_generate[cls])
            print(Fore.GREEN + f"Training for class {cls} finished successfully\n")



if __name__ == "__main__":
    main()
