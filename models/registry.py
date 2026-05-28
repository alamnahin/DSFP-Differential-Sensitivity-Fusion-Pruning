"""
models/registry.py
Provides get_model() and load_model() for all supported architectures.

Supported archs : vgg16 | alexnet | resnet56 | resnet18
Supported datasets: cifar10 (10 classes) | cifar100 (100) | tiny-imagenet (200)

VGG-16 and AlexNet are adapted for 32×32 CIFAR inputs (smaller FC layers,
no maxpool on early layers).  ResNet-56 is the canonical CIFAR variant from
He et al. (2016).  ResNet-18 uses the standard torchvision implementation with
the first conv/stride changed for 64×64 Tiny-ImageNet inputs.
"""

from __future__ import annotations
import logging
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VGG-16 (CIFAR-adapted)
# ---------------------------------------------------------------------------

class VGG16CIFAR(nn.Module):
    """VGG-16 adapted for 32×32 CIFAR inputs.

    Changes vs. ImageNet VGG-16:
      • Removes the first MaxPool so spatial dims stay 32→16→8→4→2.
      • Replaces the 3-layer 4096-FC classifier with a compact head.
    """

    CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
           512, 512, 512, "M", 512, 512, 512, "M"]

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = self._make_layers()
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def _make_layers(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_channels = 3
        for v in self.CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),  # type: ignore[arg-type]
                    nn.BatchNorm2d(v),  # type: ignore[arg-type]
                    nn.ReLU(inplace=True),
                ]
                in_channels = v  # type: ignore[assignment]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# AlexNet (CIFAR-adapted)
# ---------------------------------------------------------------------------

class AlexNetCIFAR(nn.Module):
    """AlexNet adapted for 32×32 CIFAR inputs.

    Reduces kernel sizes and strides to fit small spatial dimensions.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),           # 32→16
            # Conv2
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),           # 16→8
            # Conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # Conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),           # 8→4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# ResNet-56 for CIFAR (He et al., 2016)
# ---------------------------------------------------------------------------

class _ResBlockCIFAR(nn.Module):
    """Basic residual block for CIFAR ResNets (no bottleneck)."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet56CIFAR(nn.Module):
    """ResNet-56 for CIFAR-10/100 (n=9, depth = 6n+2 = 56)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        n = 9
        self.conv1  = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.relu   = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, 16, n, stride=1)
        self.layer2 = self._make_layer(16, 32, n, stride=2)
        self.layer3 = self._make_layer(32, 64, n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(64, num_classes)
        self._init_weights()

    @staticmethod
    def _make_layer(in_planes: int, planes: int,
                    n_blocks: int, stride: int) -> nn.Sequential:
        layers = [_ResBlockCIFAR(in_planes, planes, stride)]
        for _ in range(n_blocks - 1):
            layers.append(_ResBlockCIFAR(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return self.fc(x.view(x.size(0), -1))

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# ResNet-18 for Tiny-ImageNet (64×64)
# ---------------------------------------------------------------------------

def _resnet18_tiny_imagenet(num_classes: int = 200) -> nn.Module:
    """Standard ResNet-18 with first conv/stride adapted for 64×64 inputs."""
    model = tv_models.resnet18(weights=None)
    # Adapt stem for 64×64 (remove aggressive downsampling)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                              padding=1, bias=False)
    model.maxpool = nn.Identity()   # type: ignore[assignment]
    model.fc      = nn.Linear(512, num_classes)
    return model


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, dict] = {
    "vgg16":    {"cifar10": VGG16CIFAR,   "cifar100": VGG16CIFAR},
    "alexnet":  {"cifar10": AlexNetCIFAR, "cifar100": AlexNetCIFAR},
    "resnet56": {"cifar10": ResNet56CIFAR, "cifar100": ResNet56CIFAR},
    "resnet18": {"tiny-imagenet": _resnet18_tiny_imagenet},
}


def get_model(arch: str, num_classes: int = 10,
              device: Optional[torch.device] = None) -> nn.Module:
    """Instantiate a model by architecture name."""
    arch = arch.lower()
    if arch not in _REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. "
                         f"Choose from: {list(_REGISTRY.keys())}")
    # pick any constructor from registry (all constructors accept num_classes)
    constructor = next(iter(_REGISTRY[arch].values()))
    model = constructor(num_classes=num_classes)
    if device is not None:
        model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Built {arch} | {num_classes} classes | {n_params:.3f}M params")
    return model


def load_model(arch: str, path: str, num_classes: int = 10,
               device: Optional[torch.device] = None) -> nn.Module:
    """Instantiate and load weights from a .pth checkpoint."""
    model = get_model(arch, num_classes=num_classes, device=device)
    state = torch.load(path, map_location=device or "cpu")
    # Handle both raw state_dict and wrapped checkpoints
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    logger.info(f"Loaded weights from {path}")
    return model
