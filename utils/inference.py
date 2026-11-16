"""Prediction (optional test-time augmentation: flip or d4) and the full test evaluation of one run."""
import csv
import os

import numpy as np
import torch

from .metrics import classification_metrics

TTA_VIEWS = {
    'none': [lambda x: x],
    'flip': [lambda x: x, lambda x: x.flip(-1)],
    'd4': [lambda x: x, lambda x: x.flip(-1), lambda x: x.flip(-2), lambda x: x.flip(-1).flip(-2),
           lambda x: x.transpose(-1, -2), lambda x: x.transpose(-1, -2).flip(-1),
           lambda x: x.transpose(-1, -2).flip(-2), lambda x: x.transpose(-1, -2).flip(-1).flip(-2)],
}


@torch.no_grad()
def predict(model, images, tta='none', amp_dtype=None):
    """Mean softmax over the TTA views -> (B, C) float32."""
    out = 0
    views = TTA_VIEWS[tta]
    for v in views:
        if amp_dtype is not None:
            with torch.autocast('cuda', dtype=amp_dtype):
                logits = model(v(images))
        else:
            logits = model(v(images))
        out = out + torch.softmax(logits.float(), 1)
    return out / len(views)


@torch.no_grad()
def collect(model, loader, device, tta='none', limit=0, amp_dtype=None):
    model.eval()
    probs, labels = [], []
    for i, (x, y) in enumerate(loader):
        if limit and i >= limit:
            break
        probs.append(predict(model, x.to(device, non_blocking=True), tta, amp_dtype).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


class LeafInference:
    def __init__(self, model, device, classes, tta='none'):
        self.model, self.device, self.classes, self.tta = model.to(device), device, list(classes), tta

    def evaluate(self, loader, save_dir, suffix='', limit=0):
        probs, labels = collect(self.model, loader, self.device, self.tta, limit)
        m = classification_metrics(probs, labels, self.classes)
        self._per_image_csv(loader.dataset, probs, labels, os.path.join(save_dir, f'test_per_image{suffix}.csv'))
        np.savez_compressed(os.path.join(save_dir, f'test_probs{suffix}.npz'), probs=probs.astype(np.float32),
                            labels=labels, classes=np.array(self.classes))
        print(f"[TEST] acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} bal-acc={m['balanced_accuracy']:.4f} "
              f"kappa={m['kappa']:.4f} MCC={m['mcc']:.4f} (n={m['n']}, tta={self.tta})")
        return m, probs, labels

    def _per_image_csv(self, ds, probs, labels, path):
        """One row per test image; used for McNemar tests between models and for the error figures."""
        pred = probs.argmax(1)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['path', 'label', 'pred', 'correct', 'p_label', 'p_pred'])
            for i in range(len(labels)):
                w.writerow([ds.samples[i][0] if i < len(ds.samples) else i, self.classes[labels[i]],
                            self.classes[pred[i]], int(pred[i] == labels[i]),
                            f'{probs[i, labels[i]]:.5f}', f'{probs[i, pred[i]]:.5f}'])
