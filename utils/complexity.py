"""Params / FLOPs / FPS for the efficiency table of the paper."""
import time

import torch


@torch.no_grad()
def model_complexity(model, device, size=224, warmup=10, iters=50):
    model.eval()
    info = {'params_m': sum(p.numel() for p in model.parameters()) / 1e6, 'input_size': size}
    x = torch.randn(1, 3, size, size, device=device)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        fc = FlopCounterMode(display=False)
        with fc:
            model(x)
        flops = fc.get_total_flops()
        info.update(gflops=flops / 1e9, gmacs=flops / 2e9, flops_note='torch FlopCounterMode')
    except Exception as e:  # torch < 2.1 or unsupported op
        info.update(gflops=None, gmacs=None, flops_note=f'not available: {e}')
    if device == 'cuda':
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
        dt = time.time() - t0
        info['fps'] = iters / dt
        info['latency_ms'] = 1000 * dt / iters
        info['gpu'] = torch.cuda.get_device_name(0)
    return info
