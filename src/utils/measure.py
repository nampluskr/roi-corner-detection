# src/utils/measure.py: model parameter count, size, and inference latency measurement helpers

import time
import torch


def measure_parameters(model):
    """Return the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def measure_size_mb(model):
    """Return the model size in megabytes from its state_dict tensors (params and buffers)."""
    total_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    return total_bytes / (1024.0 * 1024.0)


def measure_latency(model, device, image_size=224, batch_size=1, warmup=5, iters=20):
    """Return mean per-batch inference latency in milliseconds on the given device."""
    model = model.to(device)
    model.eval()
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    is_cuda = torch.device(device).type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            model(images)
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            model(images)
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed / iters * 1000.0
