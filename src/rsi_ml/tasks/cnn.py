"""CIFAR ResNet: BasicBlock topology with an explicit 3x3/stride-1 stem.

Reference topology: torchvision.models.resnet (BasicBlock). The CIFAR adaptation
has no initial max-pool. All weights are initialized locally, never downloaded.
"""

from __future__ import annotations

import torch
from torch import nn

from rsi_ml.schemas import Recipe


class ResidualBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.shortcut = (
            nn.Identity()
            if incoming == outgoing and stride == 1
            else nn.Sequential(
                nn.Conv2d(incoming, outgoing, 1, stride, bias=False), nn.BatchNorm2d(outgoing)
            )
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(y)) + self.shortcut(x))


def head(input_dim: int, classes: int, recipe: Recipe) -> nn.Sequential:
    layers = []
    act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[recipe.activation]
    for width in recipe.hidden_sizes:
        layers.extend([nn.Linear(input_dim, width), act(), nn.Dropout(recipe.dropout)])
        input_dim = width
    layers.extend([nn.Dropout(recipe.dropout), nn.Linear(input_dim, classes)])
    return nn.Sequential(*layers)


class CifarResNet(nn.Module):
    def __init__(self, classes: int, recipe: Recipe):
        super().__init__()
        if recipe.cnn_channels is None or recipe.cnn_blocks is None:
            raise ValueError("CNN stage channels and block counts must be explicit.")
        width = recipe.cnn_channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU()
        )
        stages = []
        for stage, (outgoing, count) in enumerate(zip(recipe.cnn_channels, recipe.cnn_blocks)):
            for block in range(count):
                stride = 2 if stage > 0 and block == 0 else 1
                stages.append(ResidualBlock(width, outgoing, stride))
                width = outgoing
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = head(width, classes, recipe)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.stages(self.stem(x))).flatten(1))


def transform(images: torch.Tensor, recipe: Recipe, dataset, *, training: bool):
    """Images remain uint8 in storage; augment per example, then normalize."""
    x = images.to(torch.float32) / 255.0
    if training and recipe.augmentation != "none":
        padded = torch.nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
        offsets = torch.randint(0, 9, (len(x), 2))
        x = torch.stack([padded[i, :, y : y + 32, z : z + 32] for i, (y, z) in enumerate(offsets)])
        flip = torch.rand(len(x)) < 0.5
        x[flip] = x[flip].flip(-1)
        if recipe.augmentation == "crop_flip_color":
            # This exact operation and range is part of the declared implementation.
            brightness = torch.rand(len(x), 1, 1, 1) * 0.4 + 0.8
            x = (x * brightness).clamp(0, 1)
    mean = torch.tensor(dataset.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(dataset.image_std).view(1, 3, 1, 1)
    return (x - mean) / std
