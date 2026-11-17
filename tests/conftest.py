"""Synthetic fixtures below are unit-test inputs, never pretrained demo results."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session", autouse=True)
def limit_test_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def patterned_image():
    y, x = np.mgrid[:93, :137]
    pixels = np.stack([(3 * x + y) % 256, (x + 5 * y) % 256, (7 * x + y) % 256], axis=-1)
    return Image.fromarray(pixels.astype(np.uint8))
