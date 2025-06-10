from dataclasses import dataclass
from datetime import datetime

@dataclass
class TrainingConfig:
    image_size = 224
    train_batch_size = 4
    eval_batch_size = 20
    num_epochs = 200
    num_train_timesteps = 2000
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    # save_image_epochs = 10
    # save_model_epochs = 10
    mixed_precision = "fp16"
    seed = 0
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"models/generator_model/diffusion_with_lora_{timestamp}" # model_name