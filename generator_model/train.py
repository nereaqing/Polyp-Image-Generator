import numpy as np
import torch
from diffusers import StableDiffusionImageVariationPipeline, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from PolypDataset import PolypDataset

assert torch.cuda.is_available(), "GPU is not enabled"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load pretrained pipeline
pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers",
    torch_dtype=torch.float16
).to(device)


# pipe = StableDiffusionPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
# ).to("cuda")

unet = pipe.unet  # This will work


# Load UNet model to fine-tune
unet = pipe.unet
unet = unet.to(torch.float32)

# Freeze all parts except the UNet
pipe.vae.requires_grad_(False)
pipe.image_encoder.requires_grad_(False)

print("Processing dataset...")



dataset = PolypDataset(image_dir="./data/m_train2/m_train/images",
                            csv_file="./data/m_train2/m_train/train.csv",
                            # mask_dir="./data/m_train2/m_train/masks",
                            transformations=True,
    )
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print("Dataset processed and dataloader created")

print("Setting hyperparemeters...")

# Optimizer
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-6)

print("Hyperparameters set\n Starting training...")

# Training loop
unet.train()
num_epochs = 10

torch.autograd.set_detect_anomaly(True)

for epoch in range(num_epochs):
    train_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        
        images = batch[0].to(device, dtype=torch.float16)
        labels = batch[1].to(device, dtype=torch.float16)

        # Encode image into latents using VAE
        print("  Encoding image into latents using VAE...")
        with torch.no_grad():
            # latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
            vae_out = pipe.vae.encode(images)

            # Print VAE latent distribution stats BEFORE sampling
            print("VAE latents mean:", vae_out.latent_dist.mean.mean().item())
            # Get standard deviation from variance
            # std = torch.sqrt(vae_out.latent_dist.variance + 1e-6)  # Add small epsilon for numerical stability
            # print("VAE latents std:", std.mean().item())

            latents = vae_out.latent_dist.sample() * 0.18215


        print("  Adding noise...")
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()

        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        print("  Getting embeddings...")
        with torch.no_grad():
            image_embeddings = pipe.image_encoder(images).image_embeds.unsqueeze(1)
        print("image_embeddings shape:", image_embeddings.shape)

        print("  Passing embeddings to unet diffusion model...")
        model_pred = unet(noisy_latents.float(), timesteps, encoder_hidden_states=image_embeddings.float()).sample
        
        # Add these BEFORE the loss
        print("  model_pred mean:", model_pred.mean().item(), "std:", model_pred.std().item())
        print("  noise mean:", noise.mean().item(), "std:", noise.std().item())

        if torch.isnan(model_pred).any():
            print("  NaNs in model_pred!")
        if torch.isnan(noise).any():
            print("  NaNs in noise!")

        loss = torch.nn.functional.mse_loss(model_pred, noise.detach().float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        optimizer.step()
        print(loss.item())
        train_loss += loss.item()
        
        
    print("latents", latents.mean().item(), latents.std().item())
    print("noisy_latents", noisy_latents.mean().item(), noisy_latents.std().item())

    avg_train_loss = train_loss / len(dataloader)
    print(f"Epoch {epoch+1} Loss: {avg_train_loss:.4f}")


# Save fine-tuned UNet
version = "5"
path_model = f"./models/generator_model/fine_tuned_polyp_generator{version}"
pipe.save_pretrained(path_model)

print(f"Training finished and model saved at {path_model}")