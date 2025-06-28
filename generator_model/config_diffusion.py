from dataclasses import dataclass
from datetime import datetime

@dataclass
class TrainingConfig:
    image_size = 224
    train_batch_size = 8
    accumulation_steps = 1
    eval_batch_size = 20
    num_epochs = 200
    learning_rate = 1e-4
    # save_image_epochs = 10
    # save_model_epochs = 10  
    mixed_precision = "fp16"
    seed = 0
    lora_rank = 8
    device = "cuda:0"
    num_train_timesteps = 1 # default (computed dynamically)
    lr_warmup_steps = 1 # default
    lora_dropout = 0.3
    weight_img = 1.0
    weight_text = 0.1
    weight_token_class = 0.5
    weight_token_polyp = 0.5
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_dir = f"new_models/conditional_dreambooth_weighted_loss_one_vs_rest_{timestamp}" # model_name
    output_dir = f"new_models/conditional_one_vs_rest_20250620_115016"
    
    # experiment_name = "lora_with_attention"
    experiment_name = "baseline_with_lora"
    # experiment_name = "text_encoder_trained"
    
    modules_lora = ["to_q", "to_k", "to_v", "to_out.0"]
    # modules_lora = ["to_q", "to_k", "to_v", "to_out.0", "add_k_proj", "add_v_proj"]
    # modules_lora = ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"]
    # modules_lora=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2", "time_emb_proj"]
