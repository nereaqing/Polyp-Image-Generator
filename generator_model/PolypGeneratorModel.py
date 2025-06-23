from diffusers import UNet2DModel, UNet2DConditionModel
from config import TrainingConfig
from diffusers import (
    AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler,
    StableDiffusionPipeline
)
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig

config = TrainingConfig()

class PolypGeneratorModel():
    def __init__(self, device, pretrained, add_lora):
        self.pretrained = pretrained
        self.add_lora = add_lora
        
        if self.pretrained:
            self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
            self.tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder").to(device)
            self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(device)
            self.noise_scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
            
        else:
            self.unet = UNet2DModel(
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

    def get_model(self):
        return self.unet
    
    
    def add_lora_config(self, lora_config):
        self.unet.add_adapter(lora_config)
        trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.unet.parameters())
        print(f"Trainable params of unet: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")


    def unfreeze_layers(self, layers_to_unfreeze):
        for name, param in self.unet.named_parameters():
                if any(x in name for x in layers_to_unfreeze):
                    param.requires_grad = True