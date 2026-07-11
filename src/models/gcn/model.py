# src/models/gcn/model.py: CNN backbone with initial regression head and iterative GCN corner refinement

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.models.base.base_model import BaseModel

BACKBONE_WEIGHTS = {
    "resnet18": "/mnt/d/backbones/resnet18-f37072fd.pth",
    "resnet34": "/mnt/d/backbones/resnet34-b627a593.pth",
    "resnet50": "/mnt/d/backbones/resnet50-0676ba61.pth",
}

BACKBONE_BUILDERS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}

NUM_CORNERS = 4
GCN_HIDDEN = 256
NUM_ITER = 3
NUM_GCN_LAYERS = 2
OFFSET_RADIUS = 0.1


def build_normalized_adjacency():
    """Return the symmetrically normalized adjacency of the 4-cycle corner graph with self-loops."""
    adjacency = torch.tensor([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ])
    adjacency = adjacency + torch.eye(NUM_CORNERS)
    degree = adjacency.sum(dim=1)
    d_inv_sqrt = torch.diag(degree.pow(-0.5))
    return d_inv_sqrt @ adjacency @ d_inv_sqrt


class GCNModel(BaseModel):
    """CNN backbone with a global regression head and a weight-shared GCN that iteratively refines corners."""

    def __init__(self, backbone="resnet50", pretrained=True):
        super().__init__()
        if backbone not in BACKBONE_BUILDERS:
            raise ValueError("Unknown backbone: %s" % backbone)

        net = BACKBONE_BUILDERS[backbone](weights=None)
        if pretrained:
            state_dict = torch.load(BACKBONE_WEIGHTS[backbone], map_location="cpu", weights_only=True)
            net.load_state_dict(state_dict)

        in_channels = net.fc.in_features
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        self.init_head = nn.Linear(in_channels, NUM_CORNERS * 2)

        gcn_layers = []
        input_dim = in_channels + 2
        for _ in range(NUM_GCN_LAYERS):
            gcn_layers.append(nn.Linear(input_dim, GCN_HIDDEN))
            input_dim = GCN_HIDDEN
        self.gcn_layers = nn.ModuleList(gcn_layers)
        self.offset_head = nn.Linear(GCN_HIDDEN, 2)

        self.register_buffer("adjacency", build_normalized_adjacency())

    def sample_vertex_features(self, features, corners):
        grid = (corners * 2.0 - 1.0).unsqueeze(1)
        sampled = F.grid_sample(features, grid, mode="bilinear",
                                padding_mode="border", align_corners=True)
        sampled = sampled.squeeze(2).permute(0, 2, 1)
        return torch.cat([sampled, corners], dim=2)

    def refine(self, features, corners):
        vertex_features = self.sample_vertex_features(features, corners)
        hidden = vertex_features
        for layer in self.gcn_layers:
            hidden = F.relu(torch.matmul(self.adjacency, layer(hidden)))
        offset = OFFSET_RADIUS * torch.tanh(self.offset_head(hidden))
        return corners + offset

    def forward(self, images):
        features = self.backbone(images)
        pooled = features.mean(dim=(2, 3))
        corners = torch.sigmoid(self.init_head(pooled)).reshape(-1, NUM_CORNERS, 2)

        outputs = [corners]
        for _ in range(NUM_ITER):
            corners = self.refine(features, corners)
            outputs.append(corners)
        return torch.stack(outputs, dim=1)
