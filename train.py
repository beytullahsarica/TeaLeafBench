"""
TeaLeafBench - training + test entry point (train -> best checkpoint on val -> test, one command per run).

  # one run as in the paper (50 epochs; the default is 100)
  python train.py --dataset TeaLeafBD --model convnext_tiny --seed 0 --epochs 50
  # any timm model by its full name
  python train.py --dataset TeaSickness --model timm:efficientnet_b4.ra2_in1k --img_size 320
  # stronger regularisation for a small dataset
  python train.py --dataset TeaSickness --model swin_t --aug randaug --mixup 0.2 --cutmix 1.0 --tag reg
  # re-evaluate a finished run with 8-view TTA (reads <dir>/config.json, no retraining)
  python train.py --eval_dir results/<experiment> --tta d4
  # cross-dataset test (shared class names only, e.g. healthy / brown_blight / gray_blight)
  python train.py --eval_dir results/<experiment> --test_dataset TeaSickness

Datasets (folders of datasets_prepared/, made by tools/prepare_data.py): TeaLeafBD, TeaSickness, TeaDisease

Experiment name: {dataset}_{model}_{img_size}_{tag}_s{seed}
"""
import argparse
import gc
import json
import os
import re
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
torch.multiprocessing.set_sharing_strategy('file_system')

from data import LeafDataset, build_datasets, load_split, worker_init_fn  # noqa: E402
from data.leaf_datasets import build_transforms, leaf_groups  # noqa: E402
from losses import ClassificationLoss, class_weights_from_targets  # noqa: E402
from models import data_config, get_model, head_param_names, normalize_name  # noqa: E402
from utils import (FigureGenerator, LeafInference, LeafTrainer, ResultsReporter, TrainLogger,  # noqa: E402
                   load_checkpoint, model_complexity)
from utils.gradcam import evaluate_cam, select_figure_images  # noqa: E402
from utils.inference import collect  # noqa: E402

# Datasets photographed on a uniform paper background: the Grad-CAM leaf-focus score is defined only for these.
PAPER_BACKGROUND = ('TeaLeafBD', 'TeaSickness')
# Arguments that may differ when re-evaluating a finished run with --eval_dir.
EVAL_KEYS = ('eval_only', 'eval_dir', 'eval_checkpoint', 'tta', 'num_workers', 'data_root', 'complexity_size',
             'test_dataset', 'limit_batches', 'no_extra_tta_eval', 'save_dir', 'no_gradcam', 'cam_per_class')


def default_data_root():
    # datasets_prepared/ = output of tools/prepare_data.py (cleaned, split); datasets/ = raw downloads
    return os.path.join(PROJECT_ROOT, 'datasets_prepared')


def get_args():
    p = argparse.ArgumentParser(description='TeaLeafBench: tea leaf disease classification')
    g = p.add_argument_group('data')
    g.add_argument('--dataset', required=False, help='folder name under --data_root')
    g.add_argument('--data_root', default=default_data_root(),
                   help='folder holding the prepared datasets (default: ./datasets_prepared)')
    g.add_argument('--split_file', default=None, help='default: splits/<dataset>.json if it exists')
    g.add_argument('--val_ratio', type=float, default=0.15, help='held-out fraction of train when no val split exists')
    g.add_argument('--split_seed', type=int, default=42)
    g.add_argument('--img_size', type=int, default=224)
    g.add_argument('--aug', default='standard', choices=['none', 'standard', 'randaug'])
    g.add_argument('--allow_duplicates', action='store_true', help='do not stop on train/test duplicate images')
    g.add_argument('--num_workers', type=int, default=8)

    g = p.add_argument_group('model')
    g.add_argument('--model', default='convnext_tiny', help='short name (models/registry.py) or timm:<name>')
    g.add_argument('--no_pretrained', action='store_true')
    g.add_argument('--drop_rate', type=float, default=0.0)
    g.add_argument('--drop_path', type=float, default=0.1)

    g = p.add_argument_group('loss')
    g.add_argument('--loss', default='ce', choices=['ce', 'focal'])
    g.add_argument('--label_smoothing', type=float, default=0.1)
    g.add_argument('--focal_gamma', type=float, default=2.0)
    g.add_argument('--class_weights', action='store_true', help='inverse-sqrt class frequency weights')

    g = p.add_argument_group('optimisation')
    g.add_argument('--epochs', type=int, default=100)
    g.add_argument('--batch_size', type=int, default=32)
    g.add_argument('--lr', type=float, default=1e-3, help='head LR; pretrained backbone uses lr * enc_lr_mult')
    g.add_argument('--enc_lr_mult', type=float, default=0.1)
    g.add_argument('--weight_decay', type=float, default=0.05)
    g.add_argument('--warmup_epochs', type=int, default=5)
    g.add_argument('--min_lr_ratio', type=float, default=0.01)
    g.add_argument('--no_ema', action='store_true')
    g.add_argument('--ema_decay', type=float, default=0.999)
    g.add_argument('--mixup', type=float, default=0.0, help='Mixup alpha (0 = off)')
    g.add_argument('--cutmix', type=float, default=0.0, help='CutMix alpha (0 = off)')
    g.add_argument('--grad_accum_steps', type=int, default=1)
    g.add_argument('--early_stopping', type=int, default=20, help='patience in validations (0 = off)')
    g.add_argument('--monitor', default='macro_f1', choices=['macro_f1', 'accuracy', 'balanced_accuracy', 'mcc'])
    g.add_argument('--val_every', type=int, default=1)
    g.add_argument('--amp', default='auto', choices=['auto', 'bf16', 'fp16', 'none'])
    g.add_argument('--seed', type=int, default=0)

    g = p.add_argument_group('evaluation')
    g.add_argument('--test_dataset', default=None, help='cross-dataset test on this dataset\'s test split')
    g.add_argument('--tta', default='none', choices=['none', 'flip', 'd4'])
    g.add_argument('--no_extra_tta_eval', action='store_true', help='skip the extra flip-TTA test after the plain one')
    g.add_argument('--complexity_size', type=int, default=0, help='input size for GFLOPs/FPS (0 = img_size)')
    g.add_argument('--no_gradcam', action='store_true', help='skip Grad-CAM figure + leaf-focus score')
    g.add_argument('--cam_per_class', type=int, default=2, help='Grad-CAM figure: test images per class (+6 errors)')

    g = p.add_argument_group('run control')
    g.add_argument('--save_dir', default=os.path.join(PROJECT_ROOT, 'results'))
    g.add_argument('--experiment', default=None, help='override the auto-generated experiment name')
    g.add_argument('--tag', default='base', help='name tag for variants (ablations, settings)')
    g.add_argument('--eval_only', action='store_true')
    g.add_argument('--eval_dir', default=None, help='finished experiment dir: re-evaluate using its config.json')
    g.add_argument('--eval_checkpoint', default=None)
    g.add_argument('--resume', default=None)
    g.add_argument('--auto_resume', action='store_true', help='resume from <exp>/last_checkpoint.pth if it exists')
    g.add_argument('--limit_batches', type=int, default=0, help='debug: cap train/val/test batches')
    a = p.parse_args()
    if not a.dataset and not a.eval_dir:
        p.error('--dataset is required')
    return a


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def short(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def experiment_name(a):
    if a.experiment:
        return a.experiment
    model = normalize_name(a.model)
    model = short(model.split(':', 1)[1].split('.')[0]) if model.startswith('timm:') else short(model)
    return f'{short(a.dataset)}_{model}_{a.img_size}_{short(a.tag) or "base"}_s{a.seed}'


# Same class under different names in different datasets (cross-dataset tests only): the tea mosquito bug of
# TeaDisease is Helopeltis theivora, the "helopeltis" class of TeaLeafBD.
CLASS_ALIASES = {'tea_mosquito_bug': 'helopeltis'}


class SharedClassOutputs(torch.nn.Module):
    """Cross-dataset test: the model's logits restricted to the classes the test dataset shares with it."""
    def __init__(self, model, keep):
        super().__init__()
        self.model = model
        self.register_buffer('keep', torch.tensor(keep, dtype=torch.long))

    def forward(self, x):
        return self.model(x)[:, self.keep]


def eval_suffix(a):
    s = f'_on-{short(a.test_dataset)}' if a.test_dataset and a.test_dataset != a.dataset else ''
    return s + (f'_tta-{a.tta}' if a.tta != 'none' else '')


def main():
    t_start, dt_start = time.time(), datetime.now()
    a = get_args()
    if a.eval_dir:
        with open(os.path.join(a.eval_dir, 'config.json')) as f:
            cfg = json.load(f)['args']
        for k, v in cfg.items():
            if k not in EVAL_KEYS:
                setattr(a, k, v)
        a.eval_only = True
    set_seed(a.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp = experiment_name(a)
    save_dir = a.eval_dir or os.path.join(a.save_dir, exp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"{'=' * 70}\nSTART TIME: {dt_start:%Y-%m-%d %H:%M:%S}\n{'=' * 70}")
    print(f"Experiment: {exp}\nDataset={a.dataset} Model={a.model} Device={device}")
    print(f"ImgSize={a.img_size} BS={a.batch_size} Epochs={a.epochs} LR={a.lr} (backbone x{a.enc_lr_mult}) "
          f"WD={a.weight_decay} Seed={a.seed}")
    print(f"Loss={a.loss} LS={a.label_smoothing} ClassWeights={a.class_weights} Aug={a.aug} Mixup={a.mixup} "
          f"CutMix={a.cutmix} EMA={not a.no_ema} AMP={a.amp} TTA={a.tta} EarlyStop={a.early_stopping}")
    print(f"Results -> {save_dir}\n{'=' * 70}")

    # The model is built first: its pretrained config gives the normalisation the data pipeline must use.
    _, classes, _, _ = load_split(a.dataset, a.data_root, a.split_file, a.val_ratio, a.split_seed)
    model = get_model(a.model, num_classes=len(classes), pretrained=not a.no_pretrained, img_size=a.img_size,
                      drop_rate=a.drop_rate, drop_path_rate=a.drop_path)
    mean, std = data_config(model)
    train_ds, val_ds, test_ds, info = build_datasets(
        a.dataset, a.data_root, a.img_size, mean, std, a.aug, a.split_file, a.val_ratio, a.split_seed,
        check_dups=not a.allow_duplicates and not a.eval_only)
    cross_keep = None
    if a.test_dataset and a.test_dataset != a.dataset:
        # cross-dataset test: only the classes both datasets have; the model's outputs are restricted to these
        # classes (logits of the other classes dropped), so metrics are over the shared classes only
        root2, classes2, splits2, _ = load_split(a.test_dataset, a.data_root)
        own = {CLASS_ALIASES.get(c, c): c for c in classes}            # canonical name -> this model's class name
        other = {CLASS_ALIASES.get(c, c) for c in classes2}
        shared = [c for c in classes if CLASS_ALIASES.get(c, c) in other]
        if len(shared) < 2:
            raise SystemExit(f"cross-dataset test needs >= 2 shared classes, {a.dataset} / {a.test_dataset}: {shared}")
        cross_keep = [classes.index(c) for c in shared]
        samples = [(p, own[CLASS_ALIASES.get(c, c)]) for p, c in splits2['test'] if CLASS_ALIASES.get(c, c) in own]
        test_ds = LeafDataset(root2, samples, shared, build_transforms(a.img_size, mean, std)[1])
        print(f"[DATA] cross-dataset test on {a.test_dataset}: shared classes {shared}, {len(test_ds)} images")
    print(f"[DATA] classes ({len(classes)}): {classes}")
    print(f"[DATA] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} root={info['root']} "
          f"(split: {info['source']})")
    if not a.eval_only:
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump({'args': vars(a), 'experiment': exp, 'data': info, 'timm_name': getattr(model, 'timm_name', None),
                       'mean': mean, 'std': std, 'started': f'{dt_start:%Y-%m-%d %H:%M:%S}'}, f, indent=2)

    nw = a.num_workers
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=nw,
                              pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn,
                              persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=a.batch_size * 2, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=a.batch_size * 2, shuffle=False, num_workers=nw, pin_memory=True)

    if not a.eval_only:
        weights = class_weights_from_targets(train_ds.targets, len(classes)) if a.class_weights else None
        criterion = ClassificationLoss(a.loss, a.label_smoothing, a.focal_gamma, weights)
        mixup_fn = None
        if a.mixup > 0 or a.cutmix > 0:
            from timm.data import Mixup
            if a.batch_size % 2:
                raise ValueError('--mixup / --cutmix need an even --batch_size')
            # label smoothing is applied by the criterion, not by Mixup
            mixup_fn = Mixup(mixup_alpha=a.mixup, cutmix_alpha=a.cutmix, label_smoothing=0.0,
                             num_classes=len(classes))
        trainer = LeafTrainer(
            model, train_loader, val_loader, criterion, classes, head_param_names(model), device=device,
            save_dir=save_dir, experiment_name=exp, lr=a.lr, enc_lr_mult=a.enc_lr_mult,
            weight_decay=a.weight_decay, epochs=a.epochs, warmup_epochs=a.warmup_epochs,
            min_lr_ratio=a.min_lr_ratio, use_ema=not a.no_ema, ema_decay=a.ema_decay, mixup_fn=mixup_fn,
            grad_accum_steps=a.grad_accum_steps, early_stopping_patience=a.early_stopping,
            val_every=a.val_every, amp=a.amp, monitor=a.monitor, limit_batches=a.limit_batches)
        last = os.path.join(save_dir, 'last_checkpoint.pth')
        if a.resume:
            trainer.resume(a.resume)
        elif a.auto_resume and os.path.exists(last):
            trainer.resume(last)
        logger = trainer.train()
        FigureGenerator(save_dir).plot_training_curves(logger)
        del trainer
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    else:
        logger = TrainLogger.load(save_dir, exp, a.monitor)

    # ---------------------------------------------------------------- test
    print(f"\n{'=' * 60}\nTEST EVALUATION\n{'=' * 60}")
    ckpt_path = a.eval_checkpoint or os.path.join(save_dir, 'best_model.pth')
    ckpt = load_checkpoint(ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[EVAL] {ckpt_path} (epoch {ckpt.get('epoch', -1) + 1}, ema={ckpt.get('is_ema', False)})")
    if cross_keep is not None:
        # also the unrestricted outputs (all classes of the training dataset): a prediction of a class the test
        # dataset does not have then counts as an error (labels are indices into the model's own classes)
        full_probs, full_labels = collect(model.to(device), test_loader, device, limit=a.limit_batches)
        np.savez_compressed(os.path.join(save_dir, f'test_probs{eval_suffix(a)}_allclasses.npz'),
                            probs=full_probs.astype(np.float32), labels=np.array(cross_keep)[full_labels],
                            classes=np.array(classes))
        model = SharedClassOutputs(model, cross_keep)
        classes = [classes[i] for i in cross_keep]
        a.no_gradcam = True          # Grad-CAM needs the original model; cross-dataset tests report scores only
    fig = FigureGenerator(save_dir)
    meta = {'best_epoch': ckpt.get('epoch', -1) + 1, 'experiment': exp, 'dataset': a.dataset, 'model': a.model,
            'img_size': a.img_size, 'tag': a.tag, 'seed': a.seed, 'test_dataset': a.test_dataset or a.dataset,
            'val': ckpt.get('metrics', {})}
    suffix = eval_suffix(a)
    tta_modes = [a.tta] + (['flip'] if a.tta == 'none' and not a.no_extra_tta_eval and not a.eval_dir else [])
    results = None
    for k, tta in enumerate(tta_modes):
        sfx = suffix if k == 0 else f'{suffix}_tta-{tta}'
        res, probs, labels = LeafInference(model, device, classes, tta).evaluate(test_loader, save_dir, sfx,
                                                                                 a.limit_batches)
        res.update(meta, tta=tta)
        if k == 0:
            results = res
            fig.plot_confusion(res['confusion_matrix'], classes, sfx)
            fig.plot_per_class(res['per_class'], sfx)
            groups = leaf_groups(test_ds)
            fig.plot_errors(test_ds, probs, labels, classes, suffix=sfx, groups=groups)
            if not a.no_gradcam:
                # Grad-CAM + occlusion of the predicted class: figures of fixed images + errors, leaf-focus score
                t_cam = time.time()
                try:
                    sel = select_figure_images(test_ds, probs, a.cam_per_class, groups=groups)
                    # leaf-focus score only where leaves lie on paper (field photos: no reliable leaf mask)
                    test_name = a.test_dataset or a.dataset
                    leaf_focus = test_name in PAPER_BACKGROUND
                    cam_summary, cam_figs = evaluate_cam(model, test_ds, device, a.img_size, a.batch_size,
                                                         sel, a.limit_batches, leaf_focus=leaf_focus)
                    fig.plot_gradcam(cam_figs['gradcam'], classes, sfx, name='fig_gradcam')
                    fig.plot_gradcam(cam_figs['gradcampp'], classes, sfx, name='fig_gradcampp')
                    fig.plot_gradcam(cam_figs['occlusion'], classes, sfx, name='fig_occlusion')
                    res.update(cam_summary)
                    if leaf_focus:
                        nan = float('nan')
                        print(f"[CAM] {cam_summary['cam_method']}: leaf share / focus ratio on "
                              f"{cam_summary['cam_n_uniform_background']}/{cam_summary['cam_n_images']} images "
                              f"with uniform background:")
                        for lab, p in (('Grad-CAM', ''), ('Grad-CAM++', 'pp_'), ('occlusion', 'occ_')):
                            print(f"[CAM]   {lab:<10s} share={cam_summary.get(p + 'leaf_energy_share', nan):.3f} "
                                  f"ratio={cam_summary.get(p + 'leaf_focus_ratio', nan):.2f}")
                    else:
                        print(f"[CAM] {cam_summary['cam_method']}: figures only, leaf-focus score skipped "
                              f"({test_name}: field photos, no leaf mask)")
                    print(f"[CAM] time: Grad-CAM + Grad-CAM++ {cam_summary['cam_seconds']}s, occlusion "
                          f"{cam_summary['occ_seconds']}s, total {time.time() - t_cam:.0f}s")
                except Exception as e:  # never lose a finished run because of a figure
                    print(f"[CAM][WARN] Grad-CAM failed: {type(e).__name__}: {e}")
                    res['cam_error'] = f'{type(e).__name__}: {e}'
        with open(os.path.join(save_dir, f'test_results{sfx}.json'), 'w') as f:
            json.dump(res, f, indent=2)

    cx_path = os.path.join(save_dir, 'complexity.json')
    if cross_keep is not None and os.path.exists(cx_path):
        with open(cx_path) as f:          # cross-dataset test: keep the cost measured with the training run
            complexity = json.load(f)
    else:
        complexity = model_complexity(model, device, a.complexity_size or a.img_size)
        with open(cx_path, 'w') as f:
            json.dump(complexity, f, indent=2)
    ResultsReporter(save_dir).generate(exp, a, logger, results, complexity, info, suffix)

    print(f"\nALL DONE! Results -> {save_dir}")
    for fn in sorted(os.listdir(save_dir)):
        print(f"  {fn} ({os.path.getsize(os.path.join(save_dir, fn)) / 1024:.1f}KB)")
    sec = int(time.time() - t_start)
    d, h, m, s = sec // 86400, (sec % 86400) // 3600, (sec % 3600) // 60, sec % 60
    t_str = f"{d}d {h}h {m}m {s}s" if d else (f"{h}h {m}m {s}s" if h else f"{m}m {s}s")
    print(f"\n{'=' * 60}\nSTART : {dt_start:%Y-%m-%d %H:%M:%S}\nEND   : {datetime.now():%Y-%m-%d %H:%M:%S}\n"
          f"TOTAL : {t_str}\n{'=' * 60}")


if __name__ == '__main__':
    main()
