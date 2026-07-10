# src/losses/base_loss.py: base class for reusable training losses

class BaseLoss:
    """Base class computing a batch-mean loss tensor and accumulating its running mean."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, count):
        self.total += value * count
        self.count += count

    def compute(self):
        return self.total / self.count if self.count > 0 else 0.0

    def __call__(self, raw_output, target):
        loss = self.forward(raw_output, target)
        self.update(loss.item(), len(target))
        return loss

    def forward(self, raw_output, target):
        raise NotImplementedError
