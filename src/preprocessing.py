"""Match utils.dataloader_freq.TestDataset without requiring a ground-truth mask."""

from dataclasses import dataclass
from pathlib import Path
import pickle

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from utils.dct import dct_2d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class PreparedInput:
    original: Image.Image
    rgb: torch.Tensor
    high: torch.Tensor
    low: torch.Tensor


def read_image(path: str | Path) -> Image.Image:
    """Read real JPEG/PNG content. Preserve the upstream loader's orientation."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("Unsupported image extension. Use JPG, JPEG or PNG.")
    try:
        with Image.open(path) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Image contents must be JPEG or PNG, regardless of extension.")
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Cannot decode image '{path.name}': {exc}") from exc


class FINetPreprocessor:
    def __init__(self, size: int = 384):
        if isinstance(size, bool) or not isinstance(size, int) or size < 64 or size % 64:
            raise ValueError("FINet input size must be a positive multiple of 64; use 384.")
        self.size = size
        self.rgb_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.freq_transform = transforms.Compose([
            transforms.Resize((size, size)), transforms.PILToTensor(),
        ])
        # Trusted, bundled upstream statistics only; never an uploaded pickle.
        with (PROJECT_ROOT / "utils" / "freq_mean_std.pkl").open("rb") as handle:
            stats = pickle.load(handle)
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
        std = torch.as_tensor(stats["std"], dtype=torch.float32)
        if (mean.shape != (192,) or std.shape != (192,)
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
                or not (std > 0).all()):
            raise ValueError("Bundled DCT statistics must contain 192 finite means and positive standard deviations.")
        self.freq_norm = transforms.Normalize(mean=mean, std=std)

    def __call__(self, image: Image.Image) -> PreparedInput:
        if not isinstance(image, Image.Image) or min(image.size) < 1:
            raise ValueError("Preprocessing requires a nonempty Pillow image.")
        original = image.convert("RGB").copy()
        # Keep uint8 0..255 inputs to the exact original YCbCr + block DCT.
        frequency = dct_2d(self.freq_transform(original).unsqueeze(0)).squeeze(0)
        frequency = self.freq_norm(frequency) / 7.0
        y, cb, cr = frequency.split(64, dim=0)
        high = torch.cat([y[32:], cb[32:], cr[32:]], dim=0).unsqueeze(0)
        low = torch.cat([y[:32], cb[:32], cr[:32]], dim=0).unsqueeze(0)
        rgb = self.rgb_transform(original).unsqueeze(0)
        if not all(torch.isfinite(value).all() for value in (rgb, high, low)):
            raise ValueError("Preprocessing produced non-finite values.")
        return PreparedInput(original=original, rgb=rgb, high=high, low=low)
