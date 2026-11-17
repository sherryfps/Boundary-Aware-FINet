"""Audit evaluation geometry and explicitly record exclusions without repairing labels."""

from pathlib import Path
from PIL import Image
from src.training_data import audit_pairs, indexed_files


def audit_evaluation_data(root):
    root = Path(root)
    images = indexed_files(root/'Imgs', {'.jpg', '.png'})
    masks = indexed_files(root/'GT', {'.png'})
    if not images or images.keys() != masks.keys():
        raise ValueError('Evaluation requires complete image/mask filename pairing.')
    excluded = []
    for name, path in images.items():
        with Image.open(path) as image, Image.open(masks[name]) as mask:
            if image.size != mask.size:
                excluded.append({'id': name, 'reason': 'image/mask dimension mismatch; original files unchanged',
                                 'image_size_wh': list(image.size), 'mask_size_wh': list(mask.size)})
    records = audit_pairs(root, allow_soft_masks=True, group_duplicates=True,
                          excluded_ids=[entry['id'] for entry in excluded])
    return records, excluded, len(images)
