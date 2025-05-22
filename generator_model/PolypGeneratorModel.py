from diffusers import UNet2DModel, UNet2DConditionModel
from config import TrainingConfig

config = TrainingConfig()

class PolypGeneratorModel():
    def __init__(self, conditional_generation=False):
        self.conditional_generation = conditional_generation
        if self.conditional_generation:
            self.model = UNet2DConditionModel(
                    sample_size=config.image_size,
                    in_channels=3,
                    out_channels=3,
                    layers_per_block=2,
                    block_out_channels=(128, 128, 256, 256, 512, 512),
                    down_block_types=(
                        "CrossAttnDownBlock2D",
                        "DownBlock2D",
                        "DownBlock2D",
                        "DownBlock2D",
                        "CrossAttnDownBlock2D",
                        "DownBlock2D",
                    ),
                    up_block_types=(
                        "UpBlock2D",
                        "CrossAttnUpBlock2D", 
                        "UpBlock2D",
                        "UpBlock2D",
                        "UpBlock2D",
                        "UpBlock2D",
                    ),
                    cross_attention_dim=512,
                )
        
        else:
            self.model = UNet2DModel(
                        sample_size=config.image_size, 
                        in_channels=3, 
                        out_channels=3, 
                        layers_per_block=2, 
                        # block_out_channels=(128, 128, 256, 256, 512, 512),
                        block_out_channels = (128, 256, 384, 512, 512, 768),
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
        return self.model