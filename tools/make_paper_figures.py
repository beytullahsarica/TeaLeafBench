"""
Paper figures from the finished runs -> paper/figures/<name>.png (600 dpi) + .pdf (vector, fonts embedded).

  python tools/make_paper_figures.py                 # all figures
  python tools/make_paper_figures.py --only pareto perclass

Figures (base runs; colour = model category, Okabe-Ito palette, marker shape repeats the category); the name for
--only is given in brackets:
  f1_vs_gflops (pareto)              macro-F1 vs GFLOPs, one panel per dataset (mean +- sd over seeds, marker size: parameters)
  perclass_f1_heatmap (perclass)     per-class F1 of every model, one panel per dataset (mean over seeds)
  confusion_best (confusion)         row-normalised confusion matrix of the best model per dataset (best mean macro-F1, median seed)
  cross_dataset (cross)              macro-F1 on the shared classes: own test set vs the other dataset's test set, per model
                                     (mean and range over seeds; dotted: expected macro-F1 of uniform random guessing)
  cross_confusion (cross_confusion)  confusion matrices of the cross-dataset tests, summed over models and seeds
  reliability (reliability)          reliability diagrams of the reference model and the top-3 ensemble per dataset
  gradcam_compare (gradcam)          Grad-CAM of the best CNN, transformer and hybrid model on the same TeaLeafBD test images (GPU)
  dataset_samples (samples)          one test image per class and dataset
  duplicate_examples (duplicates)    an exact copy and near-duplicate pairs of TeaLeafBD that the duplicate detector grouped
  occlusion_teasickness (occlusion_ts)  occlusion maps of MaxViT-T on Tea Sickness (seed 0), copied from its run folder
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

from make_paper_tables import CROSS, DS_LABEL, SHORT, load_runs, shared_class_f1, stat  # noqa: E402
from models.registry import DEFAULT_BENCHMARK, display_name, family  # noqa: E402

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.dpi': 100,
})
DATASETS = ['TeaLeafBD', 'TeaSickness', 'TeaDisease']
# Okabe-Ito (colour-blind safe), one colour per model category in a fixed order; marker = second encoding of the category
FAMILY_STYLE = {'cnn': ('#0072B2', 'o', 'CNN'), 'light': ('#E69F00', 's', 'Lightweight'),
                'transformer': ('#009E73', '^', 'Transformer'), 'hybrid': ('#CC79A7', 'D', 'Hybrid'),
                'mamba': ('#D55E00', 'v', 'Mamba-style'), 'foundation': ('#56B4E9', 'P', 'Foundation'),
                'segmentation': ('#8C6D1F', 'X', 'Segmentation enc.'), 'yolo': ('#000000', '*', 'YOLO')}
INK, MUTED, GRID = '#222222', '#666666', '#DDDDDD'
WIDTH = 7.2                                   # inches, full text width of a two-column journal page


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext, kw in (('png', {'dpi': 600}), ('pdf', {})):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'), bbox_inches='tight', facecolor='white', **kw)
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def base_groups():
    g = load_runs(os.path.join(ROOT, 'results'), '')
    return {(k[0], k[1]): v for k, v in g.items() if k[3] == 'base' and k[2] == 224}


def class_label(c):
    return c.replace('_', ' ').replace('gray', 'grey')      # British spelling, as in the text


# ---------------------------------------------------------------------------------------------------- figures
def repel_labels(ax, points, gap_pt=8.0, near_x_pt=48.0, near_y_pt=13.0):
    """Direct labels. A point on its own is labelled beside its marker. Points close together form a group whose labels
    go in one column to the right of the group's markers, ordered by height, so that no label covers another label or
    marker of the group; thin leader lines join them. points: (x, y, label, marker area in pt^2)."""
    fig = ax.figure
    fig.canvas.draw()
    to_disp, to_data = ax.transData.transform, ax.transData.inverted().transform
    pt = fig.dpi / 72.0
    items = [(*to_disp((x, y)), x, y, t, np.sqrt(s) / 2 * pt) for x, y, t, s in points]
    # groups: connected components of "close to each other" (in display units)
    group = list(range(len(items)))
    def root(i):
        while group[i] != i:
            i = group[i]
        return i
    for i, a in enumerate(items):
        for j, b in enumerate(items[:i]):
            if abs(a[0] - b[0]) < near_x_pt * pt and abs(a[1] - b[1]) < near_y_pt * pt:
                group[root(i)] = root(j)
    members = {}
    for i, it in enumerate(items):
        members.setdefault(root(i), []).append(it)
    y_low, y_high = ax.get_window_extent().y0 + 4 * pt, ax.get_window_extent().y1 - 4 * pt
    for grp in members.values():
        grp = sorted(grp, key=lambda it: it[1])
        if len(grp) == 1:
            lx = grp[0][0] + grp[0][5] + 3 * pt
            ly = [grp[0][1]]
        else:
            lx = max(it[0] + it[5] for it in grp) + 10 * pt
            ly = []
            for it in grp:      # keep the order by height, at least one line apart
                ly.append(max(it[1], ly[-1] + gap_pt * pt) if ly else it[1])
            shift = np.mean([it[1] for it in grp]) - np.mean(ly)        # centre the column on its points
            ly = [v + shift for v in ly]
            shift = min(0.0, y_high - ly[-1]) + max(0.0, y_low - ly[0])     # stay inside the panel
            ly = [v + shift for v in ly]
        for it, v in zip(grp, ly):
            xy_text = to_data((lx, v))
            ax.annotate(it[4], (it[2], it[3]), xytext=xy_text, textcoords='data', fontsize=5.5, color=INK, va='center',
                        zorder=5, bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.85),
                        arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.35, shrinkA=0, shrinkB=it[5] / pt)
                        if len(grp) > 1 else None)


def fig_pareto(groups, out_dir):
    # one panel per row: the leading models lie close together, and the full text width leaves room for their labels
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 7.0), sharex=True)
    labels_by_ax = {}
    for ax, ds in zip(axes, DATASETS):
        labels = []
        for model in DEFAULT_BENCHMARK:
            runs = groups.get((ds, model))
            if not runs:
                continue
            f1, gf, pm = stat(runs, 'macro_f1'), stat(runs, 'cx_gflops'), stat(runs, 'cx_params_m')
            col, mk, _ = FAMILY_STYLE.get(family(model), ('#999999', 'o', ''))
            size = 12 + 2.2 * pm[0] if pm else 20
            ax.errorbar(gf[0], f1[0] * 100, yerr=f1[1] * 100 if f1[2] > 1 else None, fmt='none', ecolor=col,
                        elinewidth=1, capsize=2, zorder=2)
            ax.scatter(gf[0], f1[0] * 100, s=size, c=col, marker=mk, edgecolors='white', linewidths=0.8, zorder=3)
            labels.append((gf[0], f1[0] * 100, display_name(model).split(' ')[0], size))
        ax.set_xscale('log')
        ax.set_title(DS_LABEL[ds])
        ax.set_ylabel('Macro-F1 (%)')
        ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
        ax.set_xlim(0.7, 70)
        labels_by_ax[ax] = labels
    axes[-1].set_xlabel('GFLOPs (log scale)')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ax, labels in labels_by_ax.items():      # after the layout: label positions are computed in points
        repel_labels(ax, labels)
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle='', markersize=5, label=l)
               for c, m, l in FAMILY_STYLE.values()]
    fig.legend(handles=handles, loc='lower center', ncol=8, bbox_to_anchor=(0.5, 0.0), handletextpad=0.2,
               columnspacing=0.9)
    save(fig, out_dir, 'f1_vs_gflops')


def per_class_f1(runs):
    classes = runs[0]['per_class']['class']
    return classes, np.mean([r['per_class']['f1'] for r in runs], axis=0)


def fig_perclass(groups, out_dir):
    models = [m for m in DEFAULT_BENCHMARK if any((ds, m) in groups for ds in DATASETS)]
    ncls = {ds: len(per_class_f1(groups[(ds, models[0])])[0]) for ds in DATASETS if (ds, models[0]) in groups}
    fig, axes = plt.subplots(1, len(ncls), figsize=(WIDTH, 3.4), gridspec_kw={'width_ratios': list(ncls.values())})
    for i, (ax, ds) in enumerate(zip(axes, ncls)):
        classes, mat = None, []
        for m in models:
            c, f = per_class_f1(groups[(ds, m)]) if (ds, m) in groups else (None, None)
            classes = classes or c
            mat.append(f if f is not None else np.full(ncls[ds], np.nan))
        mat = np.array(mat) * 100
        im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=100, aspect='auto')
        for (r, c), v in np.ndenumerate(mat):
            if not np.isnan(v):
                ax.text(c, r, f'{v:.0f}', ha='center', va='center', fontsize=5.5,
                        color='white' if v > 60 else INK)
        ax.set_xticks(range(len(classes)), [class_label(c) for c in classes], rotation=45, ha='right')
        ax.set_yticks(range(len(models)), [display_name(m) for m in models] if i == 0 else [])
        ax.set_title(DS_LABEL[ds])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=list(axes), fraction=0.015, pad=0.01)
    cb.set_label('F1 (%)')
    cb.outline.set_visible(False)
    save(fig, out_dir, 'perclass_f1_heatmap')


def fig_confusion(groups, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.9))
    for ax, ds in zip(axes, DATASETS):
        cands = [(stat(v, 'macro_f1')[0], m, v) for (d, m), v in groups.items() if d == ds]
        if not cands:
            ax.axis('off')
            continue
        _, model, runs = max(cands, key=lambda c: c[0])
        r = sorted(runs, key=lambda x: x['macro_f1'])[(len(runs) - 1) // 2]     # median seed: a typical run
        seed_txt = f'\nseed {r["_seed"]} (median of {len(runs)})' if len(runs) > 1 else ''    # second line: titles overlap on one
        cm = np.array(r['confusion_matrix'], dtype=float)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        classes = r['per_class']['class']
        ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cmn):
            if cm[i, j] > 0:
                ax.text(j, i, f'{v * 100:.0f}', ha='center', va='center', fontsize=5.5,
                        color='white' if v > 0.6 else INK)
        ax.set_xticks(range(len(classes)), [class_label(c) for c in classes], rotation=45, ha='right')
        ax.set_yticks(range(len(classes)), [class_label(c) for c in classes])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True' if ds == DATASETS[0] else '')
        ax.set_title(f'{DS_LABEL[ds]}: {display_name(model)}{seed_txt}', fontsize=7.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    save(fig, out_dir, 'confusion_best')


def cross_results():
    """(training dataset, test dataset, model) -> [(test_results_on-*.json, run dir)] over the seeds tested"""
    res = {}
    for p in sorted(glob.glob(os.path.join(ROOT, 'results', '*_base_s*', 'test_results_on-*.json'))):
        with open(p) as f:
            r = json.load(f)
        if r.get('tta', 'none') == 'none':
            res.setdefault((r['dataset'], r['test_dataset'], r['model']), []).append((r, os.path.dirname(p)))
    return res


def chance_macro_f1(support):
    """Expected macro-F1 of uniform random guessing among K classes with the given test counts: per class
    precision = prevalence p_k, recall = 1/K, so F1_k = 2 p_k (1/K) / (p_k + 1/K)."""
    p = np.asarray(support, float) / np.sum(support)
    k = len(p)
    return float(np.mean(2 * p / k / (p + 1 / k)))


def fig_cross(out_dir):
    res = cross_results()
    if not res:
        print("  cross_dataset: no cross-dataset results, skipped")
        return
    models = [m for m in DEFAULT_BENCHMARK if any(k[2] == m for k in res)]
    fig, axes = plt.subplots(1, len(CROSS), figsize=(WIDTH, 3.0), sharey=True)
    y = np.arange(len(models))
    for ax, (s, t) in zip(axes, CROSS):
        for i, m in enumerate(models):
            hits = res.get((s, t, m))
            if not hits:
                continue
            own = [x for x in (shared_class_f1(d, r['per_class']['class']) for r, d in hits) if x is not None]
            cross = [r['macro_f1'] for r, _ in hits]
            col = FAMILY_STYLE.get(family(m), ('#999999',))[0]
            if own:
                ax.plot([np.mean(own) * 100, np.mean(cross) * 100], [i, i], color=GRID, linewidth=1.5, zorder=1)
                ax.scatter(np.mean(own) * 100, i, s=16, facecolors='white', edgecolors=col, linewidths=1, zorder=2)
            if len(cross) > 1:
                ax.plot([min(cross) * 100, max(cross) * 100], [i, i], color=col, linewidth=0.8, zorder=2)
            ax.scatter(np.mean(cross) * 100, i, s=16, c=col, zorder=3)
        pc = next(v[0][0] for kk, v in res.items() if kk[:2] == (s, t))['per_class']
        ax.axvline(chance_macro_f1(pc['support']) * 100, color=MUTED, linewidth=0.6, linestyle=':', zorder=0)
        ax.set_xlim(0, 102)
        ax.set_title(f'{SHORT[s]} $\\rightarrow$ {SHORT[t]}\n({len(pc["class"])} shared classes)', fontsize=7.5)
        ax.set_xlabel('Macro-F1 (%)')
        ax.grid(True, axis='x', color=GRID, linewidth=0.5, zorder=0)
    axes[0].set_yticks(y, [display_name(m) for m in models])
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], marker='o', linestyle='', markerfacecolor='white', markeredgecolor=INK,
                          markersize=4, label='own test set (shared classes)'),
               plt.Line2D([], [], marker='o', linestyle='-', color=INK, markersize=4, linewidth=0.8,
                          label='other dataset (mean, range over seeds)'),
               plt.Line2D([], [], color=MUTED, linestyle=':', linewidth=0.8, label='uniform random guessing')]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out_dir, 'cross_dataset')


def fig_cross_confusion(out_dir):
    """Row-normalised confusion matrices of the cross-dataset tests, summed over all models and seeds, one panel
    per direction (outputs restricted to the shared classes, as in the cross-dataset scores)."""
    res = cross_results()
    if not res:
        print("  cross_confusion: no cross-dataset results, skipped")
        return
    fig, axes = plt.subplots(1, len(CROSS), figsize=(WIDTH, 2.4))
    for ax, (s, t) in zip(axes, CROSS):
        hits = [h for k, v in res.items() if k[:2] == (s, t) for h in v]
        if not hits:
            ax.axis('off')
            continue
        cm = np.sum([np.array(r['confusion_matrix'], float) for r, _ in hits], axis=0)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        classes = hits[0][0]['per_class']['class']
        ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cmn):
            ax.text(j, i, f'{v * 100:.0f}', ha='center', va='center', fontsize=5.5, color='white' if v > 0.6 else INK)
        labels = [class_label(c).replace('helopeltis', 'Helopeltis') for c in classes]
        ax.set_xticks(range(len(classes)), labels, rotation=45, ha='right')
        ax.set_yticks(range(len(classes)), labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True' if (s, t) == CROSS[0] else '')
        ax.set_title(f'{SHORT[s]} $\\rightarrow$ {SHORT[t]}', fontsize=7.5)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    save(fig, out_dir, 'cross_confusion')


def fig_reliability(groups, out_dir, n_bins=15):
    """Reliability diagrams (15 equal-width confidence bins, as for the ECE) of the reference model of each dataset
    (best mean validation macro-F1) and of the top-3 soft-voting ensemble of each seed, all seeds pooled."""
    from make_paper_tables import ENSEMBLE_SEEDS, ensemble_candidates, load_probs, mean_val_f1
    items = [((ds, m, 224, 'base'), v) for (ds, m), v in groups.items()]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4), sharey=True)
    edges = np.linspace(0, 1, n_bins + 1)

    def curve(probs, labels):
        conf, ok = probs.max(1), probs.argmax(1) == labels
        b = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
        xs, ys, ns = [], [], []
        for i in range(n_bins):
            if (b == i).any():
                xs.append(conf[b == i].mean())
                ys.append(ok[b == i].mean())
                ns.append((b == i).sum())
        return np.array(xs), np.array(ys), np.array(ns)

    for ax, ds in zip(axes, DATASETS):
        cands = [(mean_val_f1(v), m, v) for (d, m), v in groups.items() if d == ds]
        if not cands:
            ax.axis('off')
            continue
        _, ref, runs = max(cands, key=lambda c: c[0])
        zs = [z for z in (load_probs(r['_dir']) for r in runs) if z]
        ep, el = [], []
        for seed in ENSEMBLE_SEEDS:
            ez = [load_probs(c[2]['_dir']) for c in ensemble_candidates(items, ds, seed)[0][:3]]
            if len(ez) == 3:
                ep.append(np.mean([z[1] for z in ez], 0))
                el.append(ez[0][0])
        ax.plot([0, 1], [0, 1], color=MUTED, linestyle=':', linewidth=0.8)
        for (probs, labels), col, lab in (
                ((np.concatenate([z[1] for z in zs]), np.concatenate([z[0] for z in zs])), '#0072B2',
                 display_name(ref)),
                ((np.concatenate(ep), np.concatenate(el)), '#D55E00', 'top-3 ensemble')):
            xs, ys, ns = curve(probs, labels)
            ax.plot(xs, ys, color=col, linewidth=1.2, marker='o', markersize=2.5, label=lab)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(DS_LABEL[ds])
        ax.set_xlabel('Confidence')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2, handlelength=1.2)    # below the axes: the curves fill them
        ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    axes[0].set_ylabel('Accuracy')
    fig.tight_layout()
    save(fig, out_dir, 'reliability')


def fig_gradcam(groups, out_dir, ds='TeaLeafBD', n_images=6):
    import torch
    from data import load_split
    from data.leaf_datasets import build_transforms
    from models import data_config, get_model
    from utils import load_checkpoint
    from utils.gradcam import GradCAM

    picks = []
    for fam in ('cnn', 'transformer', 'hybrid'):
        c = [(stat(v, 'macro_f1')[0], m) for (d, m), v in groups.items() if d == ds and family(m) == fam]
        if c:
            picks.append(max(c)[1])
    root, classes, splits, _ = load_split(ds, os.path.join(ROOT, 'datasets_prepared'))
    first = {}
    for p, c in sorted(splits['test']):
        first.setdefault(c, p)
    imgs = [(first[c], c) for c in classes if c in first][:n_images]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    maps = {}
    for m in picks:
        r = next(x for x in groups[(ds, m)] if x['_seed'] == 0)
        model = get_model(m, num_classes=len(classes), pretrained=False, img_size=224)
        model.load_state_dict(load_checkpoint(os.path.join(r['_dir'], 'best_model.pth'))['model_state_dict'])
        model = model.to(dev).eval()
        mean, std = data_config(model)
        tf = build_transforms(224, mean, std)[1]
        x = torch.stack([tf(Image.open(os.path.join(root, p)).convert('RGB')) for p, _ in imgs]).to(dev)
        cam = GradCAM(model)
        g, _, logits = cam(x)
        cam.close()
        maps[m] = (g.numpy(), logits.argmax(1).numpy())
        del model
    fig, axes = plt.subplots(len(imgs), 1 + len(picks), figsize=(1.25 * (1 + len(picks)), 1.3 * len(imgs)))
    for i, (p, c) in enumerate(imgs):
        rgb = np.asarray(Image.open(os.path.join(root, p)).convert('RGB').resize((224, 224)))
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_ylabel(class_label(c), fontsize=7)
        for j, m in enumerate(picks):
            g, pred = maps[m]
            axes[i, j + 1].imshow(rgb)
            axes[i, j + 1].imshow(g[i], cmap='jet', alpha=0.45, vmin=0, vmax=1)
            ok = classes[pred[i]] == c
            axes[i, j + 1].set_xlabel(class_label(classes[pred[i]]), fontsize=6,
                                      color=INK if ok else '#D55E00', fontweight='normal' if ok else 'bold')
        if i == 0:
            axes[0, 0].set_title('Image')
            for j, m in enumerate(picks):
                axes[0, j + 1].set_title(display_name(m))
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(h_pad=0.4, w_pad=0.2)
    save(fig, out_dir, 'gradcam_compare')


def fig_samples(out_dir):
    rows = []
    for ds in DATASETS:
        root = os.path.join(ROOT, 'datasets_prepared', ds, 'test')
        rows.append((ds, [(c, os.path.join(root, c, sorted(os.listdir(os.path.join(root, c)))[0]))
                          for c in sorted(os.listdir(root))]))
    ncol = max(len(r[1]) for r in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(WIDTH, WIDTH / ncol * len(rows) * 1.18))
    for i, (ds, items) in enumerate(rows):
        for j in range(ncol):
            ax = axes[i, j]
            ax.axis('off')
            if j < len(items):
                c, p = items[j]
                im = Image.open(p).convert('RGB')
                s = min(im.size)                          # centre square, so all tiles have the same size
                im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2))
                ax.imshow(im.resize((300, 300)))
                ax.set_title(class_label(c), fontsize=6.5, pad=2)
        axes[i, 0].text(-0.08, 0.5, DS_LABEL[ds], transform=axes[i, 0].transAxes, rotation=90, ha='right', va='center',
                        fontsize=8)
    fig.tight_layout(h_pad=0.6, w_pad=0.3)
    save(fig, out_dir, 'dataset_samples')


def fig_duplicates(out_dir, n_near=3):
    raw = os.path.join(ROOT, 'datasets', 'teaLeafBD')
    pairs = []
    with open(os.path.join(ROOT, 'datasets_prepared', 'TeaLeafBD', 'removed.csv')) as f:
        for line in f:
            if 'exact duplicate of' in line:
                src, rest = line.split(',', 1)
                pairs.append(('exact copy', rest.split('exact duplicate of ')[1].strip(), src))
                break
    with open(os.path.join(ROOT, 'datasets_prepared', '_checks', 'TeaLeafBD_duplicates.txt')) as f:
        for line in f:
            parts = [x.strip() for x in line.split('|')]
            if len(parts) == 3 and len(pairs) < 1 + n_near:
                # checked by eye: this pair is below the strictest threshold, but the leaves differ; shown as such, last
                kind = 'grouped, different leaves' if 'helopeltis_00548' in parts[1] and 'helopeltis_00558' in parts[2] else 'same leaf, re-photographed'
                pairs.append((kind, parts[1].rsplit(' [', 1)[0], parts[2].rsplit(' [', 1)[0]))
    pairs.sort(key=lambda x: x[0] == 'grouped, different leaves')
    fig, axes = plt.subplots(len(pairs), 2, figsize=(3.4, 1.55 * len(pairs)))
    for i, (kind, a, b) in enumerate(pairs):
        for j, p in enumerate((a, b)):
            ax = axes[i, j]
            ax.imshow(Image.open(os.path.join(raw, p)).convert('RGB').resize((400, 300)))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(os.path.basename(p), fontsize=5.5, color=MUTED)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[i, 0].set_ylabel(f'({chr(97 + i)}) {kind}', fontsize=6.5)
    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    save(fig, out_dir, 'duplicate_examples')


def fig_occlusion_run(out_dir, run='teasickness_maxvitt_224_base_s0', name='occlusion_teasickness'):
    """Occlusion maps of one run as written by train.py (fixed test images of every class + the most confident
    errors), copied unchanged into the paper figures."""
    import shutil
    src = os.path.join(ROOT, 'results', run)
    done = [ext for ext in ('png', 'pdf') if os.path.exists(os.path.join(src, f'fig_occlusion.{ext}'))]
    for ext in done:
        shutil.copyfile(os.path.join(src, f'fig_occlusion.{ext}'), os.path.join(out_dir, f'{name}.{ext}'))
    print(f"  {name}: {', '.join(done) or 'source missing, skipped'} ({run})")


FIGS = ['pareto', 'perclass', 'confusion', 'cross', 'cross_confusion', 'reliability', 'occlusion_ts', 'gradcam',
        'samples', 'duplicates']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default=os.path.join(ROOT, 'paper', 'figures'))
    p.add_argument('--only', nargs='+', choices=FIGS, default=FIGS)
    a = p.parse_args()
    groups = base_groups()
    print(f"{len(groups)} (dataset, model) configurations -> {a.out_dir}")
    jobs = {'pareto': lambda: fig_pareto(groups, a.out_dir), 'perclass': lambda: fig_perclass(groups, a.out_dir),
            'confusion': lambda: fig_confusion(groups, a.out_dir), 'cross': lambda: fig_cross(a.out_dir),
            'cross_confusion': lambda: fig_cross_confusion(a.out_dir),
            'reliability': lambda: fig_reliability(groups, a.out_dir),
            'occlusion_ts': lambda: fig_occlusion_run(a.out_dir),
            'gradcam': lambda: fig_gradcam(groups, a.out_dir), 'samples': lambda: fig_samples(a.out_dir),
            'duplicates': lambda: fig_duplicates(a.out_dir)}
    failed = []
    for name in a.only:
        try:
            jobs[name]()
        except Exception as e:  # noqa: BLE001  one broken figure must not stop the others
            import traceback
            traceback.print_exc()
            failed.append(name)
            print(f"  FAILED {name}: {type(e).__name__}: {e}")
    print(f"{len(a.only) - len(failed)}/{len(a.only)} figures OK" + (f"; FAILED: {failed}" if failed else ''))


if __name__ == '__main__':
    main()
