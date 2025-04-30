import os
import random

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

from config import TrainingConfig
from generator_model.PolypDiffusionDataset import PolypDataset

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "diffusion_from_scratch"
mlflow.set_experiment(EXPERIMENT_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

def log_sample_images(samples_dir, cls, epoch, num_samples=5):
    image_files = [f for f in os.listdir(samples_dir) if f.endswith(".png")]
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    for img_file in selected_files:
        full_path = os.path.join(samples_dir, img_file)
        mlflow.log_artifact(full_path, artifact_path=f"samples/images_{cls}/{epoch:04d}") 


def evaluate(config, epoch, pipeline, cls, num_imgs_to_generate):
    # Create directory to save the generated images
    cls_dir = os.path.join(config.output_dir, "samples", f"{cls}", f"{epoch:04d}")
    os.makedirs(cls_dir, exist_ok=True)

    total_saved = 0
    batch_id = 0

    while total_saved < num_imgs_to_generate:
        # Images to generate in this batch
        current_batch_size = min(config.eval_batch_size, num_imgs_to_generate - total_saved)

        images = pipeline(
            batch_size=current_batch_size,
            generator=torch.Generator(device='cpu').manual_seed(config.seed + batch_id),
        ).images

        for idx, image in enumerate(images, 1):
            image.save(os.path.join(cls_dir, f"{total_saved + idx}.png"))

        total_saved += len(images)
        batch_id += 1
        
        print(f"   Saved {total_saved} images")

    print(f"  {num_imgs_to_generate} images saved at {cls_dir}")
    
    log_sample_images(cls_dir, cls, epoch, num_samples=10)



def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, cls, num_imgs_to_generate):
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

        # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
        if epoch in [49,99]:
            evaluate(config, epoch, pipeline, cls, num_imgs_to_generate)

        # if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
        if epoch in [49, 99]:
            path_model = os.path.join(config.output_dir, "models", f"model_{cls}")
            pipeline.save_pretrained(path_model)
            print(f"  Model saved at {path_model}")
            
            mlflow.log_artifact(path_model, f"diffusion_model/model_{cls}")
            
def get_num_images_to_generate(cls):
    if cls == 'AD':
        return 100
    elif cls == 'ASS' or cls == 'HP':
        return 300


def main():
    # Load and set config
    config = TrainingConfig()
    
    classes = ['AD', 'HP', 'ASS']
        
    with mlflow.start_run(run_name=os.path.basename(config.output_dir)):
    
        for cls in classes:
            num_imgs_to_generate = get_num_images_to_generate(cls)
            train_data = PolypDataset(image_dirs=["./data/m_train2/m_train/images", "./data/m_valid/m_valid/images"],
                                    csv_files=["./data/m_train2/m_train/train.csv", "./data/m_valid/m_valid/valid.csv"],
                                    transformations=True,
                                    keep_one_class=cls)
            
            train_loader = DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True)
            
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
        
            mlflow.log_params(params)


            # Fine-tune model
            model = UNet2DModel(
                sample_size=config.image_size, 
                in_channels=3, 
                out_channels=3, 
                layers_per_block=2, 
                block_out_channels=(128, 128, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D", 
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D", 
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )

            # Create scheduler
            noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)


            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.lr_warmup_steps,
                num_training_steps=(len(train_loader) * config.num_epochs),
            )
            

            print("Starting training...")
            train_loop(config, model, noise_scheduler, optimizer, train_loader, lr_scheduler, cls, num_imgs_to_generate)
            print(f"Training for class {cls} finished successfully\n")
    
if __name__ == "__main__":
    main()