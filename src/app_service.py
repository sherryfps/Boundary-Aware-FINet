"""Real inference and downloadable artifacts used by the Gradio interface."""

from pathlib import Path
import threading
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image
import torch

from src.boundary_loss import morphological_boundary
from src.finet_adapter import CheckpointError, FINetPredictor
from src.postprocessing import make_visualizations, save_prediction, validate_threshold

BASELINE = 'Original FINet'
BOUNDARY = 'Boundary-Aware FINet'
CAMOUFLAGE = 'Camouflaged Objects'
INDUSTRIAL = 'Factory Quality Inspection'
DOMAINS = (CAMOUFLAGE, INDUSTRIAL)


class ModelRegistry:
    def __init__(self, baseline, boundary, device='auto', industrial_baseline=None, industrial_boundary=None):
        self.paths = {BASELINE: Path(baseline), BOUNDARY: Path(boundary)}
        self.industrial_paths = {BASELINE: Path(industrial_baseline or 'checkpoints/industrial/baseline/last.pth'),
                                 BOUNDARY: Path(industrial_boundary or 'checkpoints/industrial/boundary/last.pth')}
        self.device = device
        self.cache = {}
        self.lock = threading.RLock()

    def get(self, name, domain=CAMOUFLAGE):
        if domain not in DOMAINS:
            raise ValueError('Choose Camouflaged Objects or Factory Quality Inspection.')
        if name not in self.paths:
            raise ValueError('Choose Original FINet or Boundary-Aware FINet.')
        paths = self.paths if domain == CAMOUFLAGE else self.industrial_paths
        path = paths[name].expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'{domain}: {name} is unavailable. A trained checkpoint has not been installed yet.')
        signature = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        with self.lock:
            cache_key = (domain, name)
            cached = self.cache.get(cache_key)
            if cached is None or cached[0] != signature:
                predictor = FINetPredictor(path, self.device)
                if predictor.application_domain != domain:
                    raise CheckpointError(f'Checkpoint domain is {predictor.application_domain}; it cannot be used for {domain}.')
                if domain == INDUSTRIAL and (predictor.dataset_protocol != 'custom_supervised_mvtec_v1'
                                             or not predictor.training_progress
                                             or predictor.training_progress['completed_epochs'] < 1):
                    raise CheckpointError('Industrial prediction requires a genuinely trained checkpoint with supervised MVTec provenance.')
                if name == BOUNDARY and predictor.training_status != 'boundary_training':
                    raise CheckpointError('Boundary model selection requires a checkpoint from boundary-aware training.')
                if name == BASELINE and predictor.training_status == 'boundary_training':
                    raise CheckpointError('The original-model path contains boundary-trained weights.')
                self.cache[cache_key] = (signature, predictor)
            return self.cache[cache_key][1]

    def describe(self, domain=CAMOUFLAGE):
        if domain not in DOMAINS:
            raise ValueError('Unknown application domain.')
        paths = self.paths if domain == CAMOUFLAGE else self.industrial_paths
        dataset = 'COD10K-Train + CAMO-Train' if domain == CAMOUFLAGE else 'MVTec AD, custom supervised split. Not the official anomaly-detection benchmark.'
        lines = [f'**{domain}** · {dataset}']
        for name in paths:
            if not paths[name].is_file():
                lines.append(f'**{name}:** trained checkpoint pending.')
                continue
            try:
                predictor = self.get(name, domain)
                progress = getattr(predictor, 'training_progress', None)
                stage = (f" {progress['completed_epochs']}/{progress['planned_epochs']} epochs; "
                         + ('training budget complete.' if progress['training_complete'] else 'partial training checkpoint.')
                         if progress else '')
                lines.append(f'**{name}:** checkpoint loaded; {predictor.parameter_count:,} parameters; {predictor.device}.{stage}')
            except (ValueError, FileNotFoundError, RuntimeError) as error:
                lines.append(f'**{name}:** unavailable. {error}')
        return '\n\n'.join(lines)

    def available(self, domain=CAMOUFLAGE):
        choices = []
        for name in (BASELINE, BOUNDARY):
            try:
                self.get(name, domain)
                choices.append(name)
            except (ValueError, OSError, RuntimeError):
                pass
        return choices


class AppService:
    def __init__(self, registry, output_root='outputs/web'):
        self.registry = registry
        self.output_root = Path(output_root).resolve()

    def predict(self, image_path, model_name, domain=CAMOUFLAGE):
        if not image_path:
            raise ValueError('Upload a JPG or PNG photograph first.')
        with self.registry.lock:
            prediction = self.registry.get(model_name, domain).predict(image_path)
        return {'original': prediction.original, 'probability': prediction.probability,
                'metadata': prediction.metadata, 'model_name': model_name, 'domain': domain}

    def render(self, state, threshold=0.5, ground_truth=None):
        if state is None:
            raise ValueError('Run segmentation before adjusting the result.')
        validate_threshold(threshold)
        probability, original = state['probability'], state['original']
        visuals = make_visualizations(original, probability, threshold)
        foreground = torch.from_numpy((probability >= threshold).astype(np.float32))[None, None]
        edges = morphological_boundary(foreground)[0, 0].numpy()
        boundary = Image.fromarray((edges * 255).astype(np.uint8))
        metadata = {**state['metadata'], 'boundary_visualization':
                    '3x3 morphological boundary of the thresholded prediction; not ground truth'}
        metric_text = 'Ground-truth metrics are shown only when a matching mask is supplied.'
        if ground_truth:
            from src.evaluation_metrics import metrics_for_pair, read_ground_truth
            target = read_ground_truth(ground_truth, probability.shape)
            if state.get('domain') == INDUSTRIAL:
                from src.industrial_training import segmentation_metrics
                metrics = segmentation_metrics(probability, target, threshold)
            else:
                metrics = metrics_for_pair(probability, target, threshold)
            metadata['ground_truth_metrics'] = metrics
            overlap = (f"Dice {metrics['dice']:.4f} · IoU {metrics['iou']:.4f}" if state.get('domain') == INDUSTRIAL
                       else f"S-measure {metrics['s_measure']:.4f}")
            metric_text = ('User-supplied ground truth: '
                           f"{overlap} · MAE {metrics['mae']:.4f} · Boundary F1 {metrics['boundary_f1']:.4f}.")
        folder = self.output_root / uuid4().hex
        save_prediction(original, probability, folder, metadata, threshold)
        boundary.save(folder / 'boundary.png')
        original.save(folder / 'original.png')
        archive = folder / 'results.zip'
        with ZipFile(archive, 'w', ZIP_DEFLATED) as zipped:
            for path in sorted(folder.iterdir()):
                if path != archive:
                    zipped.write(path, path.name)
        summary = (f"**{state['model_name']}** · threshold {threshold:.2f} · "
                   f"forward {metadata['inference_ms']:.1f} ms · processing {metadata['processing_ms']:.1f} ms\n\n"
                   f"{metric_text}\n\nTiming is measured for this image; model loading and file saving are excluded.")
        if state.get('domain') == INDUSTRIAL:
            summary += ('\n\n**Predicted Defect Region** · MVTec AD custom supervised training. '
                        'An empty mask is allowed. Foreground pixels are candidate regions; '
                        'this prototype does not label a product defective from small mask noise. '
                        'Product acceptance/rejection requires a separately validated decision policy.')
        progress = metadata.get('training_progress')
        if progress:
            summary += (f"\n\nCheckpoint: {progress['completed_epochs']}/{progress['planned_epochs']} training epochs. "
                        + ('The configured training budget is complete.' if progress['training_complete']
                           else 'Partial training diagnostic; final model quality has not been established.'))
        return original, visuals['probability'], visuals['binary'], visuals['overlay'], boundary, summary, str(archive)

    def compare(self, image_path, threshold=0.5, ground_truth=None, domain=CAMOUFLAGE):
        # Validate both checkpoints before producing either side of a comparison.
        first = self.registry.get(BASELINE, domain)
        second = self.registry.get(BOUNDARY, domain)
        if domain == INDUSTRIAL:
            a, b = first.training_progress, second.training_progress
            if (not a or not b or not a['training_complete'] or not b['training_complete']
                    or a['planned_epochs'] != b['planned_epochs']
                    or not all(first.comparison_provenance.get(key) for key in ('initial_model_sha256', 'split_fingerprint'))
                    or first.comparison_provenance != second.comparison_provenance):
                raise ValueError('Fair industrial comparison requires both completed, matched experiments.')
        left = self.render(self.predict(image_path, BASELINE, domain), threshold, ground_truth)
        right = self.render(self.predict(image_path, BOUNDARY, domain), threshold, ground_truth)
        return left, right
