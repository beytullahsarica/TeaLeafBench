"""
Per-run figures: training curves, confusion matrices, per-class scores, misclassified test images, Grad-CAM and
occlusion maps.

Publication style (same for every figure, so panels from different runs can be combined in the paper):
  * every figure as .png (300 dpi) + .pdf (vector, TrueType fonts embedded: pdf.fonttype 42, as journals require)
  * no figure titles (the caption is in the paper), 8-9 pt text, light grid, no top/right spines
  * colourblind-safe colours: Okabe-Ito categorical palette for series, perceptually uniform single-hue / 'inferno'
    colour maps with a colour bar for magnitudes (no rainbow 'jet')
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from PIL import Image  # noqa: E402

from data.class_names import display_names  # noqa: E402

# Okabe & Ito (2008) colour-universal-design palette, fixed order
OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#56B4E9', '#CC79A7', '#F0E442', '#000000']
ERROR_COLOR = '#D55E00'          # titles of misclassified images (also marked with 'x', not colour alone)
SALIENCY_CMAP = 'inferno'        # perceptually uniform, readable in greyscale and with colour-vision deficiency
CONFUSION_CMAP = 'Blues'         # single hue, light -> dark

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.prop_cycle': matplotlib.cycler(color=OKABE_ITO), 'lines.linewidth': 1.6,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300, 'figure.dpi': 100,
})


class FigureGenerator:
    def __init__(self, save_dir, dpi=300):
        self.save_dir, self.dpi = save_dir, dpi

    def _save(self, fig, name, tight=True):
        """name.png (dpi) + name.pdf (vector, for the paper)."""
        if tight:
            fig.tight_layout()
        base = os.path.join(self.save_dir, os.path.splitext(name)[0])
        fig.savefig(base + '.png', dpi=self.dpi, bbox_inches='tight')
        fig.savefig(base + '.pdf', bbox_inches='tight')
        plt.close(fig)

    def plot_gradcam(self, figs, classes, suffix='', name='fig_gradcam'):
        """figs: {index: (image, map, true, pred, prob)} -> grid of image / overlay pairs (Grad-CAM, Grad-CAM++ or
        occlusion), correct predictions first, then errors; one shared colour bar.
        Drawn at the printed size (full page width, 7.2 in) so the 7 pt labels stay 7 pt in the paper; the overlay
        is blended into one RGB image (one embedded image per panel instead of two: half the pdf size)."""
        classes = display_names(classes, 'mpl')
        if not figs:
            return
        label = ('Occlusion: drop of predicted-class probability (normalised)' if 'occlusion' in name else
                 'Grad-CAM++ activation (normalised)' if 'gradcampp' in name else 'Grad-CAM activation (normalised)')
        items = [figs[i] for i in sorted(figs, key=lambda i: (figs[i][2] != figs[i][3], figs[i][2], i))]
        cols = 4                                   # 4 pairs (image + overlay) per row, narrow gap between pairs
        rows = int(np.ceil(len(items) / cols))
        W, gap, label_h, cbar_h = 7.2, 0.1, 0.3, 0.5
        panel = (W - gap * (cols - 1)) / (2 * cols)
        H = rows * (panel + label_h) + cbar_h
        fig = plt.figure(figsize=(W, H))
        cmap = plt.get_cmap(SALIENCY_CMAP)
        for k, (img, cam, t, p, prob) in enumerate(items):
            r, c = divmod(k, cols)
            x0 = c * (2 * panel + gap)
            y0 = H - (r + 1) * (panel + label_h)  # bottom of the image row (label above it)
            img = np.asarray(img, dtype=np.float32) / 255
            over = 0.5 * img + 0.5 * cmap(np.clip(cam, 0, 1))[..., :3]
            for j, im in enumerate((img, over)):
                a = fig.add_axes([(x0 + j * panel) / W, y0 / H, panel / W, panel / H])
                a.imshow(np.clip(im, 0, 1), interpolation='antialiased')
                a.axis('off')
            ok = t == p
            fig.text((x0 + panel) / W, (y0 + panel + 0.03) / H,
                     f"True: {classes[t]}\n{'Pred' if ok else 'x Pred'}: {classes[p]} ({prob:.2f})",
                     ha='center', va='bottom', fontsize=7, linespacing=1.15, color='black' if ok else ERROR_COLOR)
        cax = fig.add_axes([0.3, 0.27 / H, 0.4, 0.08 / H])
        cb = fig.colorbar(ScalarMappable(Normalize(0, 1), SALIENCY_CMAP), cax=cax, orientation='horizontal')
        cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        self._save(fig, f'{name}{suffix}.png', tight=False)

    def plot_training_curves(self, logger):
        """One file per panel (easy to arrange or merge in the paper): fig_curve_loss, fig_curve_metrics, fig_curve_lr."""
        rows = logger.rows
        if not rows:
            return
        ep = [r['epoch'] for r in rows]
        best = logger.best_epoch + 1 if logger.best_epoch >= 0 else None

        def panel(name, ylabel, series, log=False):
            fig, ax = plt.subplots(figsize=(3.5, 2.6))   # one journal column
            for values, lab, ls in series:
                ax.plot(ep, values, ls, label=lab)
            if best is not None:
                ax.axvline(best, color='0.5', ls=':', lw=1.0, label=f'Selected ({best})')
            if log:
                ax.set_yscale('log')
            ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel); ax.grid(alpha=0.25, lw=0.5)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2,   # below the axes: never on the curves
                      handlelength=1.8, columnspacing=1.0)
            self._save(fig, name)

        def val(k, w='raw'):
            """Validation metric of one set of weights (val_raw_* / val_ema_*); runs logged before the split: val_*."""
            return [r.get(f'val_{w}_{k}', r.get(f'val_{k}', np.nan) if w == 'raw' else np.nan) for r in rows]

        # training = dashed, validation = solid (identity not by colour alone). Training metrics are measured on the
        # augmented training batches, validation metrics on the model weights (raw) + EMA macro-F1 (model selection
        # picks raw or EMA weights, whichever is better -> 'Selected epoch').
        # short legend labels: two legend columns must fit the 3.5 in column width (definitions go in the caption)
        panel('fig_curve_loss.png', 'Loss', [
            ([r.get('loss_total', np.nan) for r in rows], 'Train (LS-CE)', '--'),
            (val('nll'), 'Val. (NLL)', '-')])
        ema = val('macro_f1', 'ema')
        panel('fig_curve_metrics.png', 'Score', [
            ([r.get('train_acc', np.nan) for r in rows], 'Train accuracy', '--'),
            (val('accuracy'), 'Val. accuracy', '-'), (val('macro_f1'), 'Val. macro-F1', '-'),
            (val('balanced_accuracy'), 'Val. bal. accuracy', '-')]
            + ([(ema, 'Val. macro-F1 (EMA)', '-.')] if not np.all(np.isnan(ema)) else []))
        panel('fig_curve_lr.png', 'Learning rate (head)', [([r['lr'] for r in rows], 'Learning rate', '-')], log=True)

    def plot_confusion(self, cm, classes, suffix=''):
        """fig_confusion (counts) and fig_confusion_norm (row-normalised = recall per class); colour = row share."""
        classes = display_names(classes, 'mpl')
        cm = np.asarray(cm, dtype=float)
        norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
        n = len(classes)
        size = max(3.5, 0.5 * n + 1.8)
        for mat, name, fmt in ((cm, 'fig_confusion', '.0f'), (norm, 'fig_confusion_norm', '.2f')):
            fig, ax = plt.subplots(figsize=(size + 0.6, size))
            im = ax.imshow(norm, cmap=CONFUSION_CMAP, vmin=0, vmax=1)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, format(mat[i, j], fmt), ha='center', va='center', fontsize=7,
                            color='white' if norm[i, j] > 0.5 else 'black')
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(classes, rotation=45, ha='right'); ax.set_yticklabels(classes)
            ax.set_xlabel('Predicted class'); ax.set_ylabel('True class')
            ax.spines[:].set_visible(True)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label('Share of true-class images', fontsize=8)
            self._save(fig, f'{name}{suffix}.png')

    def plot_per_class(self, per_class, suffix=''):
        c = display_names(per_class['class'], 'mpl')
        if 'support' in per_class:                 # test images per class: F1 of a class with 5 images is not stable
            c = [f'{name} ($n$={int(s)})' for name, s in zip(c, per_class['support'])]
        x = np.arange(len(c))
        fig, ax = plt.subplots(figsize=(max(3.5, 0.55 * len(c) + 1.5), 2.8))
        for k, (key, lab) in enumerate((('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1'))):
            ax.bar(x + (k - 1) * 0.27, per_class[key], 0.27, label=lab, color=OKABE_ITO[k], edgecolor='white', lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(c, rotation=45, ha='right'); ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score'); ax.grid(axis='y', alpha=0.25, lw=0.5); ax.set_axisbelow(True)
        ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
        self._save(fig, f'fig_per_class{suffix}.png')

    def plot_errors(self, dataset, probs, labels, classes, max_n=20, suffix='', groups=None):
        """The most confident mistakes on the test set (what the model confuses, for the discussion section).
        groups: same-leaf group per image -> each leaf is shown once. Printed size: full page width (7.2 in)."""
        classes = display_names(classes, 'mpl')
        pred = probs.argmax(1)
        wrong = np.where(pred != labels)[0]
        if not len(wrong):
            return
        wrong = wrong[np.argsort(-probs[wrong, pred[wrong]])]
        if groups is not None:
            used = set()
            wrong = [i for i in wrong if not (groups[i] in used or used.add(groups[i]))]
        # full rows only: up to max_n errors, rounded down to a multiple of 5 (the least confident ones are dropped);
        # fewer than 5 errors -> one row
        cols = min(5, len(wrong))
        wrong = wrong[:max(cols, min(max_n, len(wrong)) // cols * cols)]
        rows = len(wrong) // cols
        fig, ax = plt.subplots(rows, cols, figsize=(7.2 * cols / 5, 1.75 * rows), squeeze=False)
        for a in ax.ravel():
            a.axis('off')
        for a, i in zip(ax.ravel(), wrong):
            path = dataset.samples[i][0]
            img = Image.open(os.path.join(dataset.root, path)).convert('RGB')
            # whole photo resized to a square = what the model sees (no centre crop in the protocol), same as the
            # Grad-CAM panels; 400 px = 300 dpi at the printed panel size (full-resolution photos: 5 MB png)
            a.imshow(img.resize((400, 400), Image.LANCZOS))
            a.set_title(f"True: {classes[labels[i]]}\nPred: {classes[pred[i]]} ({probs[i, pred[i]]:.2f})", fontsize=7,
                        linespacing=1.15, pad=3)
        fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.3 / fig.get_figheight(), bottom=0.01,
                            wspace=0.06, hspace=0.3)
        self._save(fig, f'fig_errors{suffix}.png', tight=False)
