"""
models/registry.py
Provides get_model() and load_model() for all supported architectures.

Fixes applied vs original:
  [BUG-12] get_model() called next(iter(_REGISTRY[arch].values())) to get
           a constructor, ignoring the dataset key entirely.  ResNet-18 is
           only registered for tiny-imagenet but the function would silently
           use it for any dataset. Fixed: require dataset arg and look up
           the correct constructor.
  [BUG-13] load_model() used torch.load without weights_only=False which
           raises a warning (PyTorch ≥ 2.4) or will fail in future versions.
           Added weights_only=False explicitly.
  [BUG-14] VGG16CIFAR.forward calls x.view() which breaks with DataParallel
           if batch dim is not the leading dim after features.  Replaced with
           torch.flatten(x, 1) for robustness.
  [BUG-15] AlexNetCIFAR.forward: same issue.  Fixed with torch.flatten.
  [BUG-16] ResNet56CIFAR._make_layer: first block of layer2/layer3 uses
           wrong in_planes — the function is static and takes in_planes as
           arg, but original code always passed in_planes=16 for layer2
           (should be 16→32 via stride-2 block which is correct) and 32→64
           for layer3.  Actually correct in original; kept but documented.
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
    """VGG-16 adapted for 32×32 CIFAR inputs."""

    CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
           512, 512, 512, "M", 512, 512, 512, "M"]

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features   = self._make_layers()
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
        x = torch.flatten(x, 1)   # [BUG-14] was x.view(x.size(0), -1)
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
    """AlexNet adapted for 32×32 CIFAR inputs."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),           # 32→16
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),           # 16→8
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
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
        x = torch.flatten(x, 1)   # [BUG-15]
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
        self.conv1   = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(16)
        self.relu    = nn.ReLU(inplace=True)
        self.layer1  = self._make_layer(16, 16, n, stride=1)
        self.layer2  = self._make_layer(16, 32, n, stride=2)
        self.layer3  = self._make_layer(32, 64, n, stride=2)
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
        return self.fc(torch.flatten(x, 1))

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


# ---------------------------------------------------------------------------
# ResNet-18 for Tiny-ImageNet (64×64)
# ---------------------------------------------------------------------------

def _resnet18_tiny_imagenet(num_classes: int = 200) -> nn.Module:
    """Standard ResNet-18 with first conv/stride adapted for 64×64 inputs."""
    model = tv_models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                              padding=1, bias=False)
    model.maxpool = nn.Identity()   # type: ignore[assignment]
    model.fc      = nn.Linear(512, num_classes)
    return model


# ---------------------------------------------------------------------------
# Registry
# [BUG-12] dataset-aware lookup
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, dict] = {
    "vgg16":    {"cifar10": VGG16CIFAR,   "cifar100": VGG16CIFAR},
    "alexnet":  {"cifar10": AlexNetCIFAR, "cifar100": AlexNetCIFAR},
    "resnet56": {"cifar10": ResNet56CIFAR, "cifar100": ResNet56CIFAR},
    "resnet18": {"tiny-imagenet": _resnet18_tiny_imagenet},
}


def get_model(arch: str, num_classes: int = 10,
              dataset: str = "cifar10",
              device: Optional[torch.device] = None) -> nn.Module:
    """Instantiate a model by architecture + dataset name.

    [BUG-12] Original ignored the dataset key and always picked the first
    constructor.  Now validates arch/dataset compatibility.
    """
    arch = arch.lower()
    ds   = dataset.lower()
    if arch not in _REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. "
                         f"Choose from: {list(_REGISTRY.keys())}")
    arch_map = _REGISTRY[arch]
    if ds not in arch_map:
        raise ValueError(
            f"Architecture '{arch}' is not configured for dataset '{ds}'. "
            f"Available: {list(arch_map.keys())}"
        )
    constructor = arch_map[ds]
    model = constructor(num_classes=num_classes)
    if device is not None:
        model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Built {arch} ({ds}) | {num_classes} classes | {n_params:.3f}M params")
    return model


def load_model(arch: str, path: str, num_classes: int = 10,
               dataset: str = "cifar10",
               device: Optional[torch.device] = None) -> nn.Module:
    """Instantiate and load weights from a .pth checkpoint."""
    model = get_model(arch, num_classes=num_classes,
                      dataset=dataset, device=device)
    # [BUG-13] weights_only=False needed for dict payloads
    state = torch.load(path, map_location=device or "cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    logger.info(f"Loaded weights from {path}")
    return model
