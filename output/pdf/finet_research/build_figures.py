"""Build paper figures from recorded measurements and actual inference artifacts."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation_metrics import read_ground_truth, metrics_for_pair
FIGURES = HERE / 'figures'
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42,
})

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

source_root = ROOT / 'outputs/kaggle_migration/version005/FINet/experiments'
histories = {}
fig, ax = plt.subplots(figsize=(3.5, 1.95), constrained_layout=True)
for variant, color, style in [('baseline', '#24577f', '-'), ('boundary', '#a1421a', '--')]:
    source = source_root / f'{variant}_seed123456_b4_v1/epochs.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if 'epoch' in row]
    assert [r['epoch'] for r in rows] == list(range(1, 201))
    assert all(r['validation']['samples'] == 404 for r in rows)
    values = [r['validation']['mae'] for r in rows]
    histories[variant] = {'source': str(source.relative_to(ROOT)), 'sha256': sha(source),
                          'final_validation_mae': values[-1], 'epochs': len(rows),
                          'final_common_structure_loss': rows[-1]['validation']['structure_loss'],
                          'minimum_validation_mae': min(values),
                          'selected_epoch': rows[int(np.argmin(values))]['epoch'],
                          'sampled_history': {str(e): values[e-1] for e in [1,50,100,150,200]}}
    ax.plot(range(1, 201), values, label=variant.capitalize(), color=color, linestyle=style, linewidth=1)
ax.set(xlabel='Epoch', ylabel='Validation MAE', xlim=(1, 200))
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#dddddd', linewidth=.4)
ax.legend(frameon=False, loc='upper right')
fig.savefig(FIGURES / 'validation_mae.pdf')
fig.savefig(FIGURES / 'validation_mae.png', dpi=300)
plt.close(fig)

def pixel_counts(probability, target):
    prediction = probability >= .5
    result = {
        'tp': int((prediction & target).sum()),
        'fp': int((prediction & ~target).sum()),
        'fn': int((~prediction & target).sum()),
        'tn': int((~prediction & ~target).sum()),
        'predicted_pixels': int(prediction.sum()),
        'target_pixels': int(target.sum()),
    }
    assert result['tp'] + result['fp'] == result['predicted_pixels']
    assert result['tp'] + result['fn'] == result['target_pixels']
    assert sum(result[k] for k in ['tp','fp','fn','tn']) == target.size
    return result

ERROR_COLORS = {
    'TP': '#27a970', 'FP': '#d43c32', 'FN': '#366bca', 'TN': '#151515',
}
def error_rgb(probability, target):
    prediction = probability >= .5
    masks = {'TP': prediction & target, 'FP': prediction & ~target,
             'FN': ~prediction & target, 'TN': ~prediction & ~target}
    result = np.zeros((*target.shape, 3), dtype=np.uint8)
    for label, mask in masks.items():
        result[mask] = tuple(int(ERROR_COLORS[label][i:i+2], 16) for i in [1,3,5])
    return result

def draw_panel(ax, data, title=None, probability=False):
    artist = ax.imshow(data, cmap='gray', vmin=0 if probability else None,
                       vmax=1 if probability else None, interpolation='nearest')
    if title:
        ax.set_title(title, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(.35)
    return artist

def output_legend(fig, probability_artist, colorbar_bounds, legend_anchor):
    color_axis = fig.add_axes(colorbar_bounds)
    bar = fig.colorbar(probability_artist, cax=color_axis, orientation='horizontal',
                       ticks=[0, .5, 1])
    bar.set_label('Raw probability', labelpad=1)
    bar.ax.tick_params(labelsize=8, length=2, pad=1)
    handles = [Patch(facecolor=color, label=name) for name, color in ERROR_COLORS.items()]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=legend_anchor,
               ncol=4, frameon=False, handlelength=1, columnspacing=.8,
               handletextpad=.35, fontsize=8)

camo_root = ROOT / 'outputs/final_camouflage_app'
camo_verification = read_json(camo_root / 'verification.json')
sample_id = camo_verification['sample_id']
assert camo_verification['status'] == 'passed' and sample_id == 'camourflage_00012'
image_path = ROOT / f'data/TestDataset/CAMO/Imgs/{sample_id}.jpg'
target_path = ROOT / f'data/TestDataset/CAMO/GT/{sample_id}.png'
original = Image.open(image_path).convert('RGB')
target = read_ground_truth(target_path, (original.height, original.width))
development = read_json(source_root / 'baseline_seed123456_b4_v1/split_manifest.json')
assert sample_id not in {r['id'] for r in development['records']}
camo = {'sample_id': sample_id, 'native_size_wh': list(original.size),
        'image_sha256': sha(image_path), 'ground_truth_sha256': sha(target_path),
        'absent_from_development_manifest': True,
        'scope': 'One previously demonstrated CAMO test example; not an aggregate or random sample.'}
panels = [('Input', original), ('Ground truth', target)]
for variant, folder_name, expected_hash in [
    ('baseline', '473a9caeeccc41b481abc9b6bae86eba', '660a43bc1f5d57cbfd58d5cec7dd452f592932f34e1b8931567a30784af62810'),
    ('boundary', '22f0806f5f04488baf5e3b8f592b88ac', 'c7859e2b2445dbc195dfc1f94908c862f678871e286eb6a1dd99ef45f3caee48'),
]:
    folder = camo_root / folder_name
    saved = read_json(folder / 'prediction.json')
    assert saved['checkpoint_sha256'] == expected_hash
    assert sha(ROOT / f'checkpoints/{variant}/last.pth') == expected_hash
    assert saved['training_progress']['completed_epochs'] == 200 and saved['threshold'] == .5
    probability = np.load(folder / 'probability.npy')
    binary = probability >= .5
    assert np.array_equal(binary, np.asarray(Image.open(folder / 'binary.png')) > 0)
    assert np.array_equal(np.asarray(original), np.asarray(Image.open(folder / 'original.png').convert('RGB')))
    metrics = metrics_for_pair(probability, target, threshold=.5, tolerance_pixels=2)
    assert all(abs(metrics[k] - saved['ground_truth_metrics'][k]) < 1e-12
               for k in ['mae', 'boundary_f1', 's_measure'])
    intersection = np.logical_and(binary, target).sum()
    total = binary.sum() + target.sum()
    union = np.logical_or(binary, target).sum()
    metrics.update(dice=float(2*intersection/total), iou=float(intersection/union))
    camo[variant] = {'checkpoint_sha256': expected_hash, 'metrics': metrics,
                     'pixel_counts': pixel_counts(probability, target),
                     'probability_sha256': sha(folder / 'probability.npy')}
    panels.append((variant.capitalize(), binary))
fig, axes = plt.subplots(2, 2, figsize=(3.5, 2.5), constrained_layout=True)
for ax, (title, data) in zip(axes.flat, panels):
    ax.imshow(data, cmap='gray', interpolation='nearest')
    ax.set_title(title, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(.4)
fig.savefig(FIGURES / 'camouflage_example.pdf', dpi=240)
fig.savefig(FIGURES / 'camouflage_example.png', dpi=300)
plt.close(fig)

camo_probabilities = [
    np.load(camo_root / name / 'probability.npy')
    for name in ['473a9caeeccc41b481abc9b6bae86eba', '22f0806f5f04488baf5e3b8f592b88ac']
]
fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.3))
fig.subplots_adjust(left=.015, right=.99, top=.94, bottom=.17, wspace=.08, hspace=.31)
for ax, data, title in zip(axes[0], [
        original, target, camo_probabilities[0] >= .5, camo_probabilities[1] >= .5],
        ['Input', 'Ground truth', 'A mask', 'B mask']):
    draw_panel(ax, data, title)
probability_artist = draw_panel(axes[1,0], camo_probabilities[0], 'A probability', True)
draw_panel(axes[1,1], camo_probabilities[1], 'B probability', True)
draw_panel(axes[1,2], error_rgb(camo_probabilities[0], target), 'A pixel errors')
draw_panel(axes[1,3], error_rgb(camo_probabilities[1], target), 'B pixel errors')
output_legend(fig, probability_artist, [.15,.075,.22,.023], (.74,.07))
fig.savefig(FIGURES / 'camouflage_outputs.pdf', dpi=240)
fig.savefig(FIGURES / 'camouflage_outputs.png', dpi=240)
plt.close(fig)

samples_root = ROOT / 'outputs/industrial_three_samples_20261116'
report = read_json(samples_root / 'report.json')
assert report['status'] == 'passed' and report['completed_epochs'] == 200
assert report['threshold'] == .5 and len(report['samples']) == 3
fig, axes = plt.subplots(3, 3, figsize=(3.5, 3.05), constrained_layout=True)
for col, title in enumerate(['Input', 'Ground truth', 'Prediction']):
    axes[0, col].set_title(title, pad=4)
for row, record in enumerate(report['samples']):
    folder = samples_root / record['id']
    probability = np.load(folder / 'probability.npy')
    binary = np.array(Image.open(folder / 'binary.png').convert('L')) > 0
    target = np.array(Image.open(folder / 'ground_truth.png').convert('L')) > 0
    assert np.array_equal(binary, probability >= .5)
    union = np.logical_or(binary, target).sum()
    total = binary.sum() + target.sum()
    intersection = np.logical_and(binary, target).sum()
    assert abs((2 * intersection / total if total else 1) - record['dice']) < 1e-12
    assert abs((intersection / union if union else 1) - record['iou']) < 1e-12
    for col, filename in enumerate(['original.png', 'ground_truth.png', 'binary.png']):
        image = Image.open(folder / filename)
        assert image.size == (probability.shape[1], probability.shape[0])
        axes[row, col].imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        for spine in axes[row, col].spines.values():
            spine.set_linewidth(.4)
fig.savefig(FIGURES / 'industrial_examples.pdf', dpi=220)
fig.savefig(FIGURES / 'industrial_examples.png', dpi=300)
plt.close(fig)

industrial_counts = {}
fig, axes = plt.subplots(3, 5, figsize=(7.16, 4.7))
fig.subplots_adjust(left=.075, right=.995, top=.95, bottom=.13, wspace=.10, hspace=.12)
for row, record in enumerate(report['samples']):
    folder = samples_root / record['id']
    image = Image.open(folder / 'original.png').convert('RGB')
    target = np.asarray(Image.open(folder / 'ground_truth.png').convert('L')) > 0
    probability = np.load(folder / 'probability.npy')
    industrial_counts[record['id']] = pixel_counts(probability, target)
    titles = ['Input','Ground truth','Probability','Binary mask','Pixel errors'] if row == 0 else [None]*5
    data = [image, target, probability, probability >= .5, error_rgb(probability, target)]
    for col in range(5):
        artist = draw_panel(axes[row,col], data[col], titles[col], probability=(col == 2))
        if col == 2:
            probability_artist = artist
    axes[row,0].set_ylabel(['Broken\nbottle','Bent\nmetal nut','Normal\ncarpet'][row],
                          rotation=0, labelpad=5, ha='right', va='center')
output_legend(fig, probability_artist, [.28,.055,.21,.018], (.76,.055))
fig.savefig(FIGURES / 'industrial_outputs.pdf', dpi=220)
fig.savefig(FIGURES / 'industrial_outputs.png', dpi=240)
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 2.55))
fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
ax.set(xlim=(0,1), ylim=(0,1))
ax.axis('off')
def node(cx, cy, width, height, label, color='#f4f4f4'):
    box = FancyBboxPatch((cx-width/2,cy-height/2), width, height,
                         boxstyle='round,pad=0.008,rounding_size=0.015',
                         linewidth=.6, edgecolor='#555555', facecolor=color)
    ax.add_patch(box)
    ax.text(cx, cy, label, fontsize=8, ha='center', va='center')
def arrow(start, end, training=False):
    ax.annotate('', xy=end, xytext=start, arrowprops={
        'arrowstyle':'->','lw':.65,'color':'#a1421a' if training else '#444444',
        'linestyle':'--' if training else '-'})
node(.5,.93,.45,.105,'Native RGB photograph')
node(.23,.73,.42,.13,'RGB normalization\n384 x 384')
node(.77,.73,.42,.13,'YCbCr and block DCT\n384 x 384 intensities')
node(.23,.51,.42,.13,'EfficientNet-B0\nfour spatial scales')
node(.77,.51,.42,.13,'High / low tensors\n96 x 48 x 48 each')
node(.5,.30,.77,.13,'Four frequency injection stages\nasymmetric decoder and four heads')
node(.23,.075,.44,.14,'Head 1, native logits, sigmoid\nbinary mask at 0.5', '#eef4f7')
node(.77,.075,.42,.14,'Four-head structure loss\nplus boundary loss in B\n(training with target mask)', '#f7eee8')
for start,end in [
        ((.42,.876),(.23,.802)),((.58,.876),(.77,.802)),
        ((.23,.656),(.23,.585)),((.77,.656),(.77,.585)),
        ((.23,.436),(.37,.373)),((.77,.436),(.63,.373)),
        ((.35,.226),(.23,.153))]:
    arrow(start,end)
arrow((.65,.226),(.77,.153),True)
fig.savefig(FIGURES / 'pipeline.pdf')
fig.savefig(FIGURES / 'pipeline.png', dpi=300)
plt.close(fig)

baseline, boundary = [histories[v]['final_validation_mae'] for v in ['baseline', 'boundary']]
evidence = {
    'format_source': r'C:\Users\hrish\Downloads\grfzhhncsfqn.pdf',
    'format': 'IEEEtran conference, US Letter, 10 pt, two columns, default geometry',
    'camouflage_histories': histories,
    'camouflage_example': camo,
    'final_mae_absolute_decrease': baseline-boundary,
    'final_mae_relative_decrease_percent': 100*(baseline-boundary)/baseline,
    'industrial_source': str((samples_root / 'report.json').relative_to(ROOT)),
    'industrial_report_sha256': sha(samples_root / 'report.json'),
    'industrial_checkpoint_sha256': report['checkpoint_sha256'],
    'industrial_pixel_counts': industrial_counts,
    'error_map_colors': ERROR_COLORS,
    'scope': 'Camouflage validation, one same-image camouflage comparison and three industrial baseline examples; no aggregate test improvement claimed.',
}
(HERE / 'evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
print(json.dumps(evidence, indent=2))
