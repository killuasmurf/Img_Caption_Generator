import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size: int = 256, train_backbone: bool = False) -> None:
        """I use pretrained ResNet18 as a feature extractor then project features to decoder embedding size."""
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        for param in self.backbone.parameters():
            param.requires_grad = train_backbone

        self.projection = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """I return one projected feature vector per image."""
        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            features = self.backbone(images)
        features = features.view(features.size(0), -1)
        features = self.projection(features)
        features = self.bn(features)
        return features
