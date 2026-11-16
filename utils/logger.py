"""Per-epoch history (CSV + JSON), best-epoch tracking by a validation metric, resumable."""
import csv
import json
import os


class TrainLogger:
    def __init__(self, save_dir, experiment_name='exp', monitor='macro_f1'):
        self.save_dir, self.experiment_name, self.monitor = save_dir, experiment_name, monitor
        os.makedirs(save_dir, exist_ok=True)
        self.rows, self.best_metrics, self.best_epoch = [], {}, -1

    def log_epoch(self, epoch, loss_dict, val_dict, lr, epoch_time):
        row = {'epoch': epoch + 1, 'lr': lr, 'epoch_time_sec': epoch_time}
        row.update({f'loss_{k}' if k != 'train_acc' else k: float(v) for k, v in loss_dict.items()})
        is_best = False
        if val_dict is not None:
            row.update({f'val_{k}': float(v) for k, v in val_dict.items()})
            if val_dict[self.monitor] > self.best_metrics.get(self.monitor, -1.0):
                self.best_metrics, self.best_epoch, is_best = dict(val_dict), epoch, True
        self.rows.append(row)
        self._write()
        return is_best

    def _write(self):
        keys = []
        for r in self.rows:
            keys += [k for k in r if k not in keys]
        with open(os.path.join(self.save_dir, f'{self.experiment_name}_log.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
        with open(os.path.join(self.save_dir, 'train_history.json'), 'w') as f:
            json.dump(self.state_dict(), f, indent=1)

    def state_dict(self):
        return {'rows': self.rows, 'best_metrics': self.best_metrics, 'best_epoch': self.best_epoch}

    def load_state_dict(self, s):
        self.rows, self.best_metrics, self.best_epoch = s['rows'], s['best_metrics'], s['best_epoch']

    @classmethod
    def load(cls, save_dir, experiment_name='exp', monitor='macro_f1'):
        lg = cls(save_dir, experiment_name, monitor)
        p = os.path.join(save_dir, 'train_history.json')
        if os.path.exists(p):
            with open(p) as f:
                lg.load_state_dict(json.load(f))
        return lg

    def save_summary(self):
        summary = {'experiment': self.experiment_name, 'epochs_run': len(self.rows),
                   'best_epoch': self.best_epoch + 1, 'best_val': self.best_metrics,
                   'train_time_min': sum(r['epoch_time_sec'] for r in self.rows) / 60}
        with open(os.path.join(self.save_dir, f'{self.experiment_name}_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        return summary
