"""
Classification losses. Every criterion accepts hard labels (B,) or soft targets (B, C) (Mixup / CutMix).

  ce     : cross-entropy with label smoothing (default)
  focal  : focal loss (gamma), for strongly imbalanced datasets
  Both can be class-weighted (--class_weights: inverse-sqrt frequency of the training set).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    def __init__(self, kind='ce', label_smoothing=0.1, focal_gamma=2.0, class_weights=None):
        super().__init__()
        self.kind, self.ls, self.gamma = kind, label_smoothing, focal_gamma
        self.register_buffer('weight', None if class_weights is None else torch.as_tensor(class_weights).float())

    def forward(self, logits, target):
        logits = logits.float()
        n_cls = logits.size(1)
        if target.ndim == 1:
            target = F.one_hot(target, n_cls).float()
        if self.ls > 0:
            target = target * (1 - self.ls) + self.ls / n_cls
        logp = F.log_softmax(logits, 1)
        loss = -(target * logp)                                    # (B, C)
        if self.kind == 'focal':
            loss = loss * (1 - logp.exp()).pow(self.gamma)
        if self.weight is not None:
            loss = loss * self.weight
        return {'total': loss.sum(1).mean()}


def class_weights_from_targets(targets, n_cls):
    counts = torch.bincount(torch.as_tensor(targets), minlength=n_cls).float().clamp(min=1)
    w = counts.rsqrt()
    return (w / w.mean()).tolist()


__all__ = ['ClassificationLoss', 'class_weights_from_targets']
