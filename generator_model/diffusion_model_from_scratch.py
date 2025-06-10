import os
import random
import pandas as pd
import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

from config import TrainingConfig
from PolypDiffusionDataset import PolypDiffusionDataset
from PolypGeneratorModel import PolypGeneratorModel

import mlflow
print("connecting to mlflow")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "diffusion_from_scratch"
mlflow.set_experiment(EXPERIMENT_NAME)
print("connected")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

def log_sample_images(samples_dir, cls, epoch, num_samples=5):
    image_files = [f for f in os.listdir(samples_dir) if f.endswith(".png")]
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    for img_file in selected_files:
        full_path = os.path.join(samples_dir, img_file)
        mlflow.log_artifact(full_path, artifact_path=f"samples/images_{cls}/{epoch:04d}") 


def evaluate(config, epoch, pipeline, cls, imgs_to_generate):
    # Create directory to save the generated images
    cls_dir = os.path.join(config.output_dir, "samples", cls)
    os.makedirs(cls_dir, exist_ok=True)

    total_saved = 0
    batch_id = 0

    while total_saved < imgs_to_generate:
        # Images to generate in this batch
        current_batch_size = min(config.eval_batch_size, imgs_to_generate - total_saved)

        images = pipeline(
            batch_size=current_batch_size,
            generator=torch.Generator(device='cpu').manual_seed(config.seed + batch_id),
        ).images

        for idx, image in enumerate(images, 1):
            image.save(os.path.join(cls_dir, f"{total_saved + idx}.png"))

        total_saved += len(images)
        batch_id += 1
        
        print(f"   Saved {total_saved} images")

    print(f"  {imgs_to_generate} images saved at {cls_dir}")
    
    log_sample_images(cls_dir, cls, epoch, num_samples=10)



def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, cls, imgs_to_generate, text_embeddings=None):
    model.to(device)

    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)

    global_step = 0
    scaler = torch.amp.GradScaler(device)

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

            with torch.amp.autocast(device):
                if text_embeddings is not None:
                    text_embeddings = text_embeddings.repeat(bs, 1, 1) 
                    noise_pred = model(noisy_images, timesteps, encoder_hidden_states=text_embeddings, return_dict=False)[0]
                else:
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                
            scaler.scale(loss).backward()
            

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # optimizer.step()
            # lr_scheduler.step()
            # optimizer.zero_grad()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()

            total_loss += loss.item()
            global_step += 1

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)

        # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
        if epoch in [199]:
            evaluate(config, epoch, pipeline, cls, imgs_to_generate)

        # if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
        if epoch in [199]:
            path_model = os.path.join(config.output_dir, "models", f"model_{cls}")
            pipeline.save_pretrained(path_model)
            print(f"  Model saved at {path_model}")
            
            mlflow.log_artifact(path_model, f"diffusion_model/model_{cls}")
            
def counts_per_class():
    df = pd.read_csv("../data/m_train2/m_train/train.csv")
    df_proportion = pd.DataFrame(df['cls'].value_counts())
    df_proportion.columns = ['count']
    return df_proportion['count'].to_dict()
            
def get_num_images_to_generate(distribution, ad_minimum=1000, one_vs_rest=False):
    real_counts = counts_per_class()

    # Determine total number of samples required to satisfy AD minimum and distribution
    ad_target = max(real_counts['AD'], ad_minimum)
    total_target = int(ad_target / distribution[0])

    if one_vs_rest:
        rest_count = real_counts['HP'] + real_counts['ASS']
        rest_target = int(total_target * distribution[1])
        
        synthetic_needed = {
        'AD': max(0, ad_target - real_counts['AD']),
        'REST': max(0, rest_target - rest_count)
        }
    
    else:
        hp_target = int(total_target * distribution[1])
        ass_target = int(total_target * distribution[2])

        synthetic_needed = {
            'AD': max(0, ad_target - real_counts['AD']),
            'HP': max(0, hp_target - real_counts['HP']),
            'ASS': max(0, ass_target - real_counts['ASS']),
        }
    
    print(f"Images that will be generated: \n {synthetic_needed}")

    return synthetic_needed



def main():
    print("starting")
    parser = argparse.ArgumentParser()
    parser.add_argument('--one_vs_rest', action='store_true', help='If provided, it will be performed AD vs REST')
    parser.add_argument('--conditional_generation', action='store_true', help='If provided, text embeddings will be provided to the model to train')
    args = parser.parse_args()
    
    # Load and set config
    config = TrainingConfig()
    print(os.path.basename(config.output_dir))
    
    if args.one_vs_rest:
        classes = ['AD', 'REST']
        class_map = {
            'AD': ['AD'],
            'REST': ['HP', 'ASS']
        }
        if args.conditional_generation:
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
        
        if args.conditional_generation:
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
                                                              ad_minimum=1000, 
                                                              one_vs_rest=args.one_vs_rest)
        else:
            percentage_distribution = (0.4, 0.3, 0.3)
            num_imgs_to_generate = get_num_images_to_generate(distribution=percentage_distribution, 
                                                              ad_minimum=1000, 
                                                              one_vs_rest=args.one_vs_rest)
        mlflow.log_param("images_to_generate_per_class", num_imgs_to_generate)
        mlflow.log_param("percentage_image_distribution", percentage_distribution)
    
    
        for cls in classes:
            keep_classes = class_map[cls]
            train_data = PolypDiffusionDataset(image_dirs=["../data/m_train2/m_train/images", "../data/m_valid/m_valid/images"],
                                    csv_files=["../data/m_train2/m_train/train.csv", "../data/m_valid/m_valid/valid.csv"],
                                    transformations=True,
                                    keep_one_class=keep_classes)
            
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
            

            # Create model
            model = PolypGeneratorModel(args.conditional_generation).get_model()
            
            if args.conditional_generation:
                tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

                input_prompt = [f"a high-resolution endoscopic image of {acronyms_to_words[cls]} polyp"]
                tokens_prompt = tokenizer(input_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    prompt_embeddings = text_encoder(**tokens_prompt).last_hidden_state
                prompt_embeddings = prompt_embeddings.to(device)
                mlflow.log_param("input_prompt", "a high-resolution endoscopic image of x polyp")
                
                print(input_prompt)


            # Create scheduler
            noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)


            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.lr_warmup_steps,
                num_training_steps=(len(train_loader) * config.num_epochs),
            )
            

            print("Starting training...")
            train_loop(config, model, noise_scheduler, optimizer, train_loader, lr_scheduler, 
                       cls, num_imgs_to_generate[cls])
            print(f"Training for class {cls} finished successfully\n")
    
if __name__ == "__main__":
    main()