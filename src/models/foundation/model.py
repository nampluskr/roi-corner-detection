# src/models/foundation/model.py: frozen DINOv2 backbone with a spatial-preserving strided-conv head

import torch
import torch.nn as nn
import timm

from src.models.base.base_model import BaseModel

DINOV2_WEIGHTS = {
    "vits14": "/mnt/d/backbones/dinov2_vits14_pretrain.pth",
    "vitb14": "/mnt/d/backbones/dinov2_vitb14_pretrain.pth",
    "vitl14": "/mnt/d/backbones/dinov2_vitl14_pretrain.pth",
    "vits14_reg4": "/mnt/d/backbones/dinov2_vits14_reg4_pretrain.pth",
    "vitb14_reg4": "/mnt/d/backbones/dinov2_vitb14_reg4_pretrain.pth",
    "vitl14_reg4": "/mnt/d/backbones/dinov2_vitl14_reg4_pretrain.pth",
}

DINOV2_TIMM_NAMES = {
    "vits14": "vit_small_patch14_dinov2",
    "vitb14": "vit_base_patch14_dinov2",
    "vitl14": "vit_large_patch14_dinov2",
    "vits14_reg4": "vit_small_patch14_reg4_dinov2",
    "vitb14_reg4": "vit_base_patch14_reg4_dinov2",
    "vitl14_reg4": "vit_large_patch14_reg4_dinov2",
}


class FoundationModel(BaseModel):
    """Frozen DINOv2 backbone feeding a spatial-preserving strided-conv head for corner regression."""

    def __init__(self, backbone="vits14", pretrained=True):
        super().__init__()
        if backbone not in DINOV2_TIMM_NAMES:
            raise ValueError("Unknown backbone: %s" % backbone)

        self.backbone = timm.create_model(DINOV2_TIMM_NAMES[backbone], pretrained=False,
                                          num_classes=0, img_size=518, dynamic_img_size=True)
        if pretrained:
            state_dict = torch.load(DINOV2_WEIGHTS[backbone], map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state_dict, strict=False)

        self.num_prefix_tokens = self.backbone.num_prefix_tokens
        embed_dim = self.backbone.embed_dim
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 8),
        )

    def forward(self, images):
        features = self.backbone.forward_features(images)
        patches = features[:, self.num_prefix_tokens:, :]
        n, num_patches, dim = patches.shape
        grid = int(round(num_patches ** 0.5))
        feature_map = patches.transpose(1, 2).reshape(n, dim, grid, grid)
        return self.head(feature_map)
