import torch.nn as nn
import torchvision.models as models       

class PolypClassificationModel(nn.Module):
    def __init__(self, num_classes, dropout, hidden_features):
        super(PolypClassificationModel, self).__init__()
        
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        in_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)