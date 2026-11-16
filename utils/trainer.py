"""
Training loop:
  * AdamW, pretrained part at lr * enc_lr_mult, new classifier head at lr; no weight decay on norms/biases
  * per-iteration linear warm-up + cosine decay
  * AMP: bf16 on Ampere+ (no GradScaler), else fp16 + GradScaler
  * EMA of the weights; raw and EMA weights are both validated, the better one is kept
  * model selection on the validation set only (monitor, default macro-F1); early stopping on the same metric
  * optional Mixup / CutMix (timm)
  * resumable: last_checkpoint.pth is written atomically every epoch (time limits, interruptions)
"""
import contextlib
import math
import os
import time
from copy import deepcopy

import numpy as np
import torch

from .inference import collect
from .logger import TrainLogger
from .metrics import classification_metrics, scalar_metrics


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:                       # torch < 1.13
        return torch.load(path, map_location='cpu')


def build_optimizer(model, lr, enc_lr_mult, weight_decay, head_names):
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = (n not in head_names, p.ndim <= 1 or n.endswith('.bias'))
        groups.setdefault(key, []).append(p)
    param_groups = []
    for (enc, no_wd), params in sorted(groups.items()):
        param_groups.append({'params': params, 'lr': lr * (enc_lr_mult if enc else 1.0),
                             'weight_decay': 0.0 if no_wd else weight_decay})
        print(f"[OPT] {'backbone' if enc else 'head'}{' (no wd)' if no_wd else ''}: "
              f"{sum(p.numel() for p in params) / 1e6:.2f}M params, lr={param_groups[-1]['lr']:.1e}")
    return torch.optim.AdamW(param_groups)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best_score, self.counter = None, 0

    def __call__(self, score):
        if self.patience <= 0:
            return False
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score, self.counter = score, 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            print(f"\n[EARLY STOP] no val improvement for {self.patience} validations (best={self.best_score:.4f})")
            return True
        return False


class LeafTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, classes, head_names, device='cuda', save_dir='.',
                 experiment_name='exp', lr=1e-3, enc_lr_mult=0.1, weight_decay=0.05, epochs=100,
                 warmup_epochs=5, min_lr_ratio=0.01, use_ema=True, ema_decay=0.999, mixup_fn=None,
                 grad_accum_steps=1, early_stopping_patience=20, val_every=1, amp='auto', grad_clip=1.0,
                 monitor='macro_f1', limit_batches=0):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.train_loader, self.val_loader, self.classes = train_loader, val_loader, list(classes)
        self.device, self.save_dir, self.epochs, self.monitor = device, save_dir, epochs, monitor
        self.mixup_fn = mixup_fn
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.val_every, self.grad_clip, self.limit_batches = max(1, val_every), grad_clip, limit_batches
        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = build_optimizer(self.model, lr, enc_lr_mult, weight_decay, head_names)
        n_batches = len(train_loader) if not limit_batches else min(limit_batches, len(train_loader))
        self.n_batches = n_batches
        steps_per_epoch = math.ceil(n_batches / self.grad_accum_steps)
        total = max(1, epochs * steps_per_epoch)
        warmup = min(warmup_epochs * steps_per_epoch, total - 1)

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            prog = min(1.0, (step - warmup) / max(1, total - warmup))
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.use_amp = (device == 'cuda' and amp != 'none')
        if amp == 'auto':
            # bf16 only on GPUs with native support (compute capability >= 8; V100 would emulate it slowly)
            native_bf16 = self.use_amp and torch.cuda.get_device_capability()[0] >= 8
            self.amp_dtype = torch.bfloat16 if native_bf16 else torch.float16
        else:
            self.amp_dtype = torch.bfloat16 if amp == 'bf16' else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)
        print(f"[TRAIN] amp={self.use_amp} dtype={self.amp_dtype if self.use_amp else 'fp32'} "
              f"steps/epoch={steps_per_epoch} total_steps={total} warmup={warmup} monitor=val {monitor}")

        self.ema = ModelEMA(self.model, ema_decay) if use_ema else None
        self.logger = TrainLogger(save_dir, experiment_name, monitor=monitor)
        self.early_stopping = EarlyStopping(early_stopping_patience)
        self.start_epoch, self.global_step, self.finished = 0, 0, False

    # ------------------------------------------------------------------ helpers
    def _autocast(self):
        return torch.autocast('cuda', dtype=self.amp_dtype) if self.use_amp else contextlib.nullcontext()

    @staticmethod
    def _atomic_save(obj, path):
        tmp = path + '.tmp'
        torch.save(obj, tmp)
        os.replace(tmp, path)

    # ------------------------------------------------------------------ train / val
    def train_epoch(self, epoch):
        self.model.train()
        sums, n, correct, seen = {}, 0, 0, 0
        self.optimizer.zero_grad(set_to_none=True)
        for bi, (images, labels) in enumerate(self.train_loader):
            if bi >= self.n_batches:
                break
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            target = labels
            if self.mixup_fn is not None:
                images, target = self.mixup_fn(images, labels)
            with self._autocast():
                logits = self.model(images)
            losses = self.criterion(logits, target)
            loss = losses['total']
            if not torch.isfinite(loss):
                print(f"  [WARN] non-finite loss at batch {bi}, skipped")
                self.optimizer.zero_grad(set_to_none=True)
                continue
            self.scaler.scale(loss / self.grad_accum_steps).backward()
            if (bi + 1) % self.grad_accum_steps == 0 or (bi + 1) == self.n_batches:
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1
                if self.ema is not None:
                    self.ema.update(self.model)
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            if self.mixup_fn is None:
                correct += int((logits.argmax(1) == labels).sum())
                seen += labels.numel()
            n += 1
            if bi % 50 == 0:
                print(f"  batch {bi}/{self.n_batches} loss={float(loss):.4f}")
        out = {k: v / max(n, 1) for k, v in sums.items()}
        if seen:
            out['train_acc'] = correct / seen
        return out

    def validate(self, model):
        probs, labels = collect(model, self.val_loader, self.device, limit=self.limit_batches,
                                amp_dtype=self.amp_dtype if self.use_amp else None)
        if not np.isfinite(probs).all() and self.use_amp:
            # fp16 overflow (e.g. early EMA weights of MobileNetV4): repeat this validation in fp32
            print("  [VAL] non-finite outputs under AMP -> repeating validation in fp32")
            probs, labels = collect(model, self.val_loader, self.device, limit=self.limit_batches)
        if not np.isfinite(probs).all():
            # still non-finite: these weights get uniform probabilities, i.e. a bad score, and are not selected
            print("  [VAL] non-finite outputs in fp32 -> scored as uniform predictions")
            bad = ~np.isfinite(probs).all(axis=1)
            probs[bad] = 1.0 / probs.shape[1]
        return scalar_metrics(classification_metrics(probs, labels, self.classes))

    def train(self):
        if self.finished:
            print("[TRAIN] checkpoint says training already finished -> skipping to evaluation")
            return self.logger
        print(f"\n{'=' * 60}\nTraining epochs {self.start_epoch + 1}..{self.epochs} -> {self.save_dir}\n{'=' * 60}")
        epoch = self.start_epoch - 1          # defined even if no epoch is left to run
        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            lc = self.train_epoch(epoch)
            vm, use_ema = None, False
            if (epoch + 1) % self.val_every == 0 or epoch + 1 == self.epochs:
                vm = vm_raw = self.validate(self.model)
                vm_ema = None
                if self.ema is not None:
                    vm_ema = self.validate(self.ema.module)
                    print(f"  [VAL] raw {self.monitor}={vm[self.monitor]:.4f} | EMA {self.monitor}={vm_ema[self.monitor]:.4f}")
                    if vm_ema[self.monitor] >= vm[self.monitor]:
                        vm, use_ema = vm_ema, True
                vm = dict(vm, ema_selected=float(use_ema))
                # val_* = selected weights (model selection); raw_* / ema_* = each set of weights on its own, so that
                # the training curves follow one model (val_* alone would switch between raw and EMA weights)
                for name, m in (('raw', vm_raw), ('ema', vm_ema)):
                    if m is not None:
                        vm.update({f'{name}_{k}': m[k] for k in ('accuracy', 'macro_f1', 'balanced_accuracy', 'nll')})
            lr = max(g['lr'] for g in self.optimizer.param_groups)
            is_best = self.logger.log_epoch(epoch, lc, vm, lr, time.time() - t0)
            msg = f"  loss={lc.get('total', 0):.4f} lr={lr:.2e} {time.time() - t0:.0f}s"
            if vm is not None:
                msg += (f" | val acc={vm['accuracy']:.4f} macro-F1={vm['macro_f1']:.4f} "
                        f"bal-acc={vm['balanced_accuracy']:.4f} loss={vm['nll']:.4f}")
            print(msg)
            if is_best:
                self._save_best(epoch, vm, use_ema)
            stop = vm is not None and self.early_stopping(vm[self.monitor])
            self._save_last(epoch, finished=stop or epoch + 1 == self.epochs)
            if stop:
                break
        self.finished = True
        self._save_last(epoch, finished=True, slim=True)   # weights no longer needed: best_model.pth has them
        self.logger.save_summary()
        if self.logger.best_epoch >= 0:
            print(f"\nDone. Best val {self.monitor}={self.logger.best_metrics[self.monitor]:.4f} "
                  f"@ epoch {self.logger.best_epoch + 1}")
        return self.logger

    # ------------------------------------------------------------------ checkpoints
    def _save_best(self, epoch, vm, use_ema=False):
        src = self.ema.module if (use_ema and self.ema is not None) else self.model
        self._atomic_save({'epoch': epoch, 'model_state_dict': src.state_dict(), 'classes': self.classes,
                           'metrics': vm, 'is_ema': bool(use_ema)},
                          os.path.join(self.save_dir, 'best_model.pth'))
        print(f"  * new best val {self.monitor}={vm[self.monitor]:.4f} ({'EMA' if use_ema else 'raw'}) -> best_model.pth")

    def _save_last(self, epoch, finished=False, slim=False):
        """Resumable state after every epoch. slim=True (end of training): only the 'finished' marker and the
        history (a few KB instead of ~400 MB), enough for --auto_resume to skip straight to evaluation."""
        state = {'epoch': epoch, 'finished': finished, 'global_step': self.global_step,
                 'early_stopping': (self.early_stopping.best_score, self.early_stopping.counter),
                 'logger': self.logger.state_dict()}
        if slim:
            self._atomic_save(state, os.path.join(self.save_dir, 'last_checkpoint.pth'))
            return
        state.update({'model_state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
                      'scaler': self.scaler.state_dict()})
        if self.ema is not None:
            state['ema_state_dict'] = self.ema.module.state_dict()
            state['ema_updates'] = self.ema.updates
        self._atomic_save(state, os.path.join(self.save_dir, 'last_checkpoint.pth'))

    def resume(self, path):
        ck = load_checkpoint(path)
        if 'model_state_dict' in ck:              # absent in the slim checkpoint of a finished run
            self.model.load_state_dict(ck['model_state_dict'])
        if 'optimizer' in ck:
            self.optimizer.load_state_dict(ck['optimizer'])
        if 'scheduler' in ck:
            self.scheduler.load_state_dict(ck['scheduler'])
        if ck.get('scaler'):
            self.scaler.load_state_dict(ck['scaler'])
        if self.ema is not None and ck.get('ema_state_dict') is not None:
            self.ema.module.load_state_dict(ck['ema_state_dict'])
            self.ema.updates = ck.get('ema_updates', 0)
        if ck.get('early_stopping'):
            self.early_stopping.best_score, self.early_stopping.counter = ck['early_stopping']
        if ck.get('logger'):
            self.logger.load_state_dict(ck['logger'])
        self.global_step = ck.get('global_step', 0)
        self.start_epoch = ck.get('epoch', -1) + 1
        self.finished = bool(ck.get('finished', False))
        print(f"[RESUME] {path}: continuing at epoch {self.start_epoch + 1} (finished={self.finished})")
