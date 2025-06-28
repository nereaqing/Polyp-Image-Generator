from dataclasses import dataclass
from datetime import datetime

@dataclass
class ConfigClassification:
    image_size = 224
    batch_size = 16
    num_epochs = 100
    patience = 10
    learning_rate = 0.001
    weight_decay = 0.001
    hidden_features = 256
    dropout = 0.5
    
    weighted_sampling = True
    weighted_loss = False
    
    device = "cuda:0"