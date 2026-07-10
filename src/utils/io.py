# src/utils/io.py: save/load model checkpoints as .pth files

import os
import torch


def save_model(model, checkpoint):
    """Save a model's state_dict to a .pth file, creating the parent directory if needed."""
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    torch.save(model.state_dict(), checkpoint)


def load_model(model, checkpoint):
    """Load a model's state_dict from a .pth file in place."""
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
