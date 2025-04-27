from PIL import Image
import os

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid

from config import TrainingConfig
from PolypDataset import PolypDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    model.to(device)

    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)

    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            clean_images = batch[0].to(device)  # assuming (image, label) tuple
            noise = torch.randn(clean_images.shape, device=device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=device,
                dtype=torch.int64
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            print('Predicting noise')
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            print(loss)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            global_step += 1

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{config.num_epochs} - Loss: {avg_loss:.2f}")

        pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)

        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            evaluate(config, epoch, pipeline)

        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            pipeline.save_pretrained(config.output_dir)



def main():
    # Load and set config
    config = TrainingConfig()

    train_data = PolypDataset(image_dir="./data/m_train2/m_train/images",
                            csv_file="./data/m_train2/m_train/train.csv",
                            transformations=True)
    train_loader = DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True)


    # Fine-tune model
    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    # sample_image, label = next(iter(train_loader))
    # print("Input shape:", sample_image.shape)

    # timesteps = torch.tensor([0])
    # print("Output shape:", model(sample_image, timestep=timesteps).sample.shape)


    # Create scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    # noise = torch.randn(sample_image.shape)
    # timesteps = torch.LongTensor([50])
    # noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

    # noise_pred = model(noisy_image, timesteps).sample
    # loss = F.mse_loss(noise_pred, noise)


    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_loader) * config.num_epochs),
    )

    print("Starting training...")
    train_loop(config, model, noise_scheduler, optimizer, train_loader, lr_scheduler)
    print("Training finished successfully")
    
if __name__ == "__main__":
    main()