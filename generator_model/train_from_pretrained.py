from PIL import Image
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusers import (
    AutoencoderKL, UNet2DConditionModel,
    UniPCMultistepScheduler, StableDiffusionPipeline
)
from diffusers.optimization import get_cosine_schedule_with_warmup

from transformers import CLIPTextModel, CLIPTokenizer

from peft import LoraConfig, PeftModel

from config import TrainingConfig
from PolypDataset import PolypDataset

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("generator_model")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def save_lora_weights(unet, save_path):
    os.makedirs(save_path, exist_ok=True)
    state_dict = {
        k: v.cpu()
        for k, v in unet.state_dict().items()
        if "lora_" in k  # or whatever your LoRA param names start with
    }
    torch.save(state_dict, os.path.join(save_path, "lora_weights.pth"))
    print(f"LoRA weights saved to {save_path}/lora_weights.pth")
    
    
def load_lora_weights(unet, load_path):
    lora_weights = torch.load(os.path.join(load_path, "lora_weights.pth"), map_location=device)
    unet.load_state_dict(lora_weights, strict=False)
    print(f"LoRA weights loaded from {load_path}/lora_weights.pth")




def evaluate(config, epoch, pipeline, input_prompt):
    images = pipeline(
        prompt=input_prompt * config.eval_batch_size,
        height=config.image_size,
        width=config.image_size,
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=torch.Generator(device="cpu").manual_seed(config.seed)
    ).images

    epoch_dir = os.path.join(config.output_dir, "samples", f"{epoch:04d}")
    os.makedirs(epoch_dir, exist_ok=True)
    for idx, image in enumerate(images, 1):
        image.save(f"{epoch_dir}/{idx}.png")
    print(f"  Images saved at {epoch_dir}")



def train_loop(config, unet, vae, text_encoder, tokenizer, noise_scheduler, optimizer, dataloader, lr_scheduler, prompt):
    unet.train()
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            pixel_values = batch[0].to(device)  # images
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Text conditioning
            text_input = tokenizer(
                prompt * latents.shape[0],
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            with torch.no_grad():
                encoder_hidden_states = text_encoder(text_input.input_ids.to(device))[0]


            # Predict noise
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred, noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Avg Loss = {total_loss / len(dataloader):.4f}")

        if epoch == config.num_epochs - 1:
            lora_save_path = os.path.join(config.output_dir, "lora_weights")
            save_lora_weights(unet, lora_save_path)

            pipeline = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=noise_scheduler,
                safety_checker=None,
                feature_extractor=None
            ).to(device)
            
            load_lora_weights(pipeline.unet, lora_save_path)
            evaluate(config, epoch, pipeline, prompt)

            unet.save_pretrained(config.output_dir)
            print(f"  Model saved at {config.output_dir}")


def main():
    config = TrainingConfig()
    config.image_size = 256  # Reduce to 256x256 for memory efficiency

    dataset = PolypDataset(
        image_dir="../data/m_train2/m_train/images",
        csv_file="../data/m_train2/m_train/train.csv",
        transformations=True,
        img_size=config.image_size  # Your dataset class should support this
    )
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    # Load pretrained components
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



    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=config.learning_rate
    )

    # Optimizer and LR scheduler
    # optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(dataloader) * config.num_epochs),
    )
    
    input_prompt = ["a realistic photo of colon polyp"]
    print(input_prompt)
    print("Starting training...")
    train_loop(config, unet, vae, text_encoder, tokenizer, noise_scheduler, optimizer, dataloader, lr_scheduler, input_prompt)


if __name__ == "__main__":
    main()
