"""Markdown report of one run (config, data, best epoch, test metrics, per-class table, complexity)."""
import os
from datetime import datetime

from data.class_names import display_name


class ResultsReporter:
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def generate(self, exp, args, logger, results, complexity, data_info, suffix=''):
        L = [f"# Experiment report: {exp}", '', f"Generated {datetime.now():%Y-%m-%d %H:%M:%S}", '',
             '## Setup', '', '| key | value |', '|---|---|']
        for k in ('dataset', 'model', 'img_size', 'epochs', 'batch_size', 'lr', 'enc_lr_mult', 'weight_decay',
                  'loss', 'label_smoothing', 'class_weights', 'aug', 'mixup', 'cutmix', 'tta', 'seed'):
            L.append(f"| {k} | {getattr(args, k, '')} |")
        L += ['', '## Data', '', f"root: `{data_info.get('root')}` (split source: {data_info.get('source')})", '',
              '| class | train | val | test |', '|---|---:|---:|---:|']
        cc = data_info.get('class_counts', {})
        for c in data_info.get('classes', []):
            L.append(f"| {display_name(c)} | {cc.get('train', {}).get(c, 0)} | {cc.get('val', {}).get(c, 0)} | "
                     f"{cc.get('test', {}).get(c, 0)} |")
        L.append(f"| **total** | {data_info.get('n_train')} | {data_info.get('n_val')} | {data_info.get('n_test')} |")
        bm = logger.best_metrics or {}
        L += ['', '## Training', '', f"epochs run: {len(logger.rows)}, best epoch: {logger.best_epoch + 1}, "
              f"best val {logger.monitor}: {bm.get(logger.monitor, float('nan')):.4f}", '',
              f'## Test{suffix or ""}', '', '| metric | value |', '|---|---:|']
        for k in ('accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1',
                  'kappa', 'mcc', 'top3_accuracy', 'macro_auc', 'nll', 'ece'):
            if isinstance(results.get(k), float):
                L.append(f"| {k} | {results[k]:.4f} |")
        pc = results.get('per_class')
        if pc:
            L += ['', '| class | precision | recall | F1 | support |', '|---|---:|---:|---:|---:|']
            for i, c in enumerate(pc['class']):
                L.append(f"| {display_name(c)} | {pc['precision'][i]:.4f} | {pc['recall'][i]:.4f} | {pc['f1'][i]:.4f} | "
                         f"{pc['support'][i]} |")
        L += ['', '## Complexity', '']
        L.append(f"params {complexity.get('params_m', 0):.2f} M, GFLOPs {complexity.get('gflops')}, "
                 f"FPS {complexity.get('fps')} ({complexity.get('gpu', 'cpu')}, {complexity.get('input_size')} px)")
        L += ['', '## Figures', '']
        for f in sorted(os.listdir(self.save_dir)):
            if f.startswith('fig_') and f.endswith('.png'):
                L.append(f"![{f}]({f})")
        with open(os.path.join(self.save_dir, f'experiment_report{suffix}.md'), 'w') as f:
            f.write('\n'.join(L) + '\n')
