"""Plot actual matched training histories; never infer test quality from validation."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = ROOT / 'outputs/kaggle_migration/version005/FINet/experiments'
    destination = ROOT / 'outputs/final_camouflage_training'
    destination.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    summary = {}
    for name, color in (('baseline', '#626c77'), ('boundary', '#a35535')):
        rows = [json.loads(line) for line in (root / f'{name}_seed123456_b4_v1/epochs.jsonl').read_text().splitlines()]
        rows = [row for row in rows if 'epoch' in row]
        assert [row['epoch'] for row in rows] == list(range(1, 201))
        assert all(row['validation']['samples'] == 404 for row in rows)
        label = 'Original FINet' if name == 'baseline' else 'Boundary-Aware FINet'
        for axis, values in zip(axes, ([row['training_loss'] for row in rows],
                                       [row['validation']['mae'] for row in rows])):
            axis.plot([row['epoch'] for row in rows], values, color=color, label=label)
            axis.set_xlabel('Completed epoch')
            axis.grid(alpha=0.2)
        best = min(rows, key=lambda row: row['validation']['mae'])
        summary[name] = {'completed_epochs': len(rows), 'last_training_loss': rows[-1]['training_loss'],
                         'final_validation_mae': rows[-1]['validation']['mae'],
                         'selected_epoch': best['epoch'], 'best_validation_mae': best['validation']['mae']}
    axes[0].set(title='Training objective (objectives differ)', ylabel='Mean summed four-head loss')
    axes[1].set(title='Shared 404-image validation split', ylabel='Native-resolution raw-probability MAE')
    axes[1].legend()
    figure.suptitle('Actual FINet training histories · seed 123456 · 200 epochs each')
    figure.savefig(destination / 'training_curves.png', dpi=180)
    figure.savefig(destination / 'training_curves.pdf')
    plt.close(figure)
    summary['interpretation'] = ('Validation only. Training objectives differ by boundary loss. '
                                 'These curves do not establish test improvement or significance.')
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
