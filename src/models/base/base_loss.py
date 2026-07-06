# src/models/base/base_loss.py: base class for method-specific training losses

class BaseLoss:
    """Base class computing a scalar training loss from raw model output and target."""

    def __call__(self, raw_output, target):
        raise NotImplementedError
