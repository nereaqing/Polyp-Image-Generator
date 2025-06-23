import torch
import sys

# Load the LoRA state dict
# path = sys.argv[1]
path = "./models/generator_model/unconditional_with_lora_and_unfreezing_20250617_155343/lora_AD/lora_weights.pth"
state_dict = torch.load(path, map_location="cpu")

# Extract top-level module names like "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight"
target_modules = set()

for key in state_dict.keys():
    if "lora_A" in key or "lora_B" in key:
        parts = key.split(".")
        for i in range(len(parts)):
            if parts[i] in ["lora_A", "lora_B"]:
                # Module path before .lora_A / .lora_B
                target_modules.add(".".join(parts[:i]))
                break

# Show the deduplicated target modules
print("Recovered LoRA target modules:")
for module in sorted(target_modules):
    print(f"- {module}")
