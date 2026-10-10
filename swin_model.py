"""
swin_model.py – Swin-Tiny model builder for binary audio deepfake detection.

Uses timm's pretrained Swin-Tiny (ImageNet-1k weights) and replaces the
classification head with a 2-class linear layer.
"""

import timm
import torch
import torch.nn as nn

import config as cfg


class SwinBinaryClassifier(nn.Module):
    """
    Pretrained Swin-Tiny with a custom binary classification head.

    Input:  (B, 3, 224, 224) float32 tensor
    Output: (B, 2) logits
    """

    def __init__(self, model_name: str = cfg.SWIN_MODEL, num_classes: int = cfg.NUM_CLASSES,
                 pretrained: bool = True, drop_rate: float = 0.1):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,        # remove built-in head; we add our own
            drop_rate=drop_rate,
        )
        in_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(drop_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats  = self.backbone(x)    # (B, in_features)
        logits = self.head(feats)    # (B, num_classes)
        return logits


def build_model(pretrained: bool = True) -> SwinBinaryClassifier:
    model = SwinBinaryClassifier(pretrained=pretrained)
    return model
