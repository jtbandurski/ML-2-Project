import torch
import torch.nn as nn
from utils import CONFIG
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
    efficientnet_v2_l,
    EfficientNet_V2_L_Weights,
)
from utils import CONFIG, DEVICE


class MobileNetV3(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV3, self).__init__()
        self.base_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.base_model.classifier[-1] = nn.Linear(
            self.base_model.classifier[3].in_features, num_classes
        )

    def forward(self, x):
        x = self.base_model(x)
        return x


class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        self.base_model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        self.base_model.classifier[-1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        return x
