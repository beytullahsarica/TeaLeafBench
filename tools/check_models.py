"""
Check every benchmark model before the runs: pretrained weights load (and are cached for the
offline training jobs), forward + backward work at 224 px, Grad-CAM finds a feature map, and the model's input
normalisation and size are printed.

  python tools/check_models.py                      # the 12 benchmark models
  python tools/check_models.py --models yolo11s_cls segformer_b2
"""
import argparse
import os
import sys
import time
import traceback

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import DEFAULT_BENCHMARK, data_config, display_name, family, get_model, head_param_names  # noqa: E402
from utils.gradcam import GradCAM  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=DEFAULT_BENCHMARK)
    p.add_argument('--num_classes', type=int, default=5)
    a = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    failed = []
    for m in a.models:
        t0 = time.time()
        try:
            model = get_model(m, num_classes=a.num_classes, pretrained=True, img_size=224, drop_path_rate=0.1).to(dev)
            n_par = sum(p.numel() for p in model.parameters()) / 1e6
            n_head = len(head_param_names(model))
            mean, std = data_config(model)
            x = torch.randn(4, 3, 224, 224, device=dev)
            model.train()
            out = model(x)
            assert out.shape == (4, a.num_classes), f'output shape {tuple(out.shape)}'
            out.sum().backward()
            no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
            if no_grad:                          # e.g. a mask token that is only used in self-supervised pre-training
                print(f"     {m}: {len(no_grad)} parameters get no gradient: {no_grad[:5]}", flush=True)
            cam = GradCAM(model)
            g, gpp, logits = cam(x)
            cam.close()
            assert g.shape == (4, 224, 224) and torch.isfinite(g).all(), f'Grad-CAM shape {tuple(g.shape)}'
            print(f"OK   {m:<16} {display_name(m):<18} {family(m):<12} {n_par:6.1f} M params | head params {n_head} | "
                  f"mean {tuple(round(v, 3) for v in mean)} std {tuple(round(v, 3) for v in std)} | "
                  f"Grad-CAM {cam.method} | {time.time() - t0:.0f}s", flush=True)
            del model, cam
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            failed.append(m)
            print(f"FAIL {m}", flush=True)
            traceback.print_exc()
    print(f"\n{len(a.models) - len(failed)}/{len(a.models)} models OK" + (f"; FAILED: {failed}" if failed else ''))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
