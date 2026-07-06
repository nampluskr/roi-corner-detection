# src/models/base/base_model.py: base class for method-specific corner detection models

import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for method-specific raw-output models."""

    def forward(self, images):
        raise NotImplementedError
