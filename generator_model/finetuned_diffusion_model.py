import os

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline, UNet2DConditionModel, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

from config import TrainingConfig
from PolypDataset import PolypDataset

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "generator_model"
mlflow.set_experiment(EXPERIMENT_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed),  # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    # Create a subfolder for the current epoch
    epoch_dir = os.path.join(config.output_dir, "samples", f"{epoch:04d}")
    os.makedirs(epoch_dir, exist_ok=True)

    # Save each image separately
    for idx, image in enumerate(images, 1):
        image.save(f"{epoch_dir}/{idx}.png")
    
    print(f"  Images saved at {epoch_dir}")




def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    model.to(device)

    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)

    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

        for _, batch in enumerate(train_dataloader):
            clean_images = batch[0].to(device) 
            noise = torch.randn(clean_images.shape, device=device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=device,
                dtype=torch.int64
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            global_step += 1

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)

        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            evaluate(config, epoch, pipeline)

        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            pipeline.save_pretrained(config.output_dir)
            print(f"  Model saved at {config.output_dir}")



def main():
    # Load and set config
    config = TrainingConfig()

    train_data = PolypDataset(image_dir="./data/m_train2/m_train/images",
                            csv_file="./data/m_train2/m_train/train.csv",
                            transformations=True)
    train_loader = DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True)


    # Fine-tune model
    model = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-2-1", subfolder="unet"
    )


    # Create scheduler
    noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1")


    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_loader) * config.num_epochs),
    )
    
    params = {
            "transformations": train_data.transformations_list,
            "criterion": "MSELoss",
            "optimizer": "AdamW",
            "batch_size": config.train_batch_size,
            "learning_rate": config.learning_rate,
            "num_epohcs": config.num_epochs,
            "image_size": config.image_size,
            "train_timesteps": config.num_train_timesteps
        }

    print("Starting training...")
    train_loop(config, model, noise_scheduler, optimizer, train_loader, lr_scheduler)
    print("Training finished successfully")
    
if __name__ == "__main__":
    main()