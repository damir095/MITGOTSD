import timm
import torch.nn as nn

from src.config import NUM_CLASSES


def build_model(freeze_backbone: bool = True) -> nn.Module:
    """EfficientNet-B0 pretrained on ImageNet, head replaced for NUM_CLASSES."""
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=NUM_CLASSES,
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    return model


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True
