"""Fetch six real training pairs from a documented COD10K mirror, for smoke tests only.

The mirror declares SINet provenance; it is not the dataset authors' official host.
Full baseline data must be obtained/audited separately. No test split is downloaded.
"""

import argparse
import hashlib
from io import BytesIO, RawIOBase
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

DATASET = "nobg/COD10K"
PARQUET_PATH = "data/train-00000-of-00002.parquet"


class RangeReader(RawIOBase):
    """Read only Parquet footer/selected column chunks, never the full archive."""
    def __init__(self, url, size):
        self.url, self.size, self.position, self.transferred = url, size, 0, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        return self.position

    def read(self, size=-1):
        size = min(self.size - self.position, size if size >= 0 else self.size)
        if size <= 0:
            return b""
        if size > 256 * 1024 * 1024:
            raise ValueError("Refusing a large smoke-data transfer; use an official local dataset instead.")
        request = Request(self.url, headers={"Range": f"bytes={self.position}-{self.position + size - 1}"})
        with urlopen(request, timeout=120) as response:
            if response.status != 206:
                raise ValueError("Server does not support partial downloads; refusing the full dataset file.")
            data = response.read(size + 1)
        if len(data) != size:
            raise ValueError("Unexpected HTTP range length.")
        self.position += size
        self.transferred += size
        return data


def fetch(destination: Path) -> dict:
    if (destination / "provenance.json").exists():
        raise FileExistsError("Smoke data already exists; use it or select a new destination.")
    with urlopen(f"https://huggingface.co/api/datasets/{DATASET}", timeout=60) as response:
        revision = json.load(response)["sha"]
    with urlopen(f"https://huggingface.co/api/datasets/{DATASET}/tree/{revision}/data", timeout=60) as response:
        files = json.load(response)
    size = next(entry["size"] for entry in files if entry["path"] == PARQUET_PATH)
    # Viewer previews are JPEG, including masks. Read original encoded Parquet bytes
    # instead; never threshold a lossy preview and present it as original ground truth.
    import pyarrow.parquet as pq
    source_url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{PARQUET_PATH}"
    reader = RangeReader(source_url, size)
    parquet = pq.ParquetFile(reader)
    print(json.dumps({"parquet_rows": parquet.metadata.num_rows,
                      "first_row_group_rows": parquet.metadata.row_group(0).num_rows}), flush=True)
    rows = next(parquet.iter_batches(batch_size=6, columns=["image", "mask"], use_threads=False)).to_pylist()
    if len(rows) != 6:
        raise ValueError("Expected exactly six real training rows.")
    records = []
    for index, entry in enumerate(rows):
        record = {"row_index": index, "split": "train"}
        sizes = []
        for field, folder in (("image", "Imgs"), ("mask", "GT")):
            payload = entry[field]["bytes"]
            record[f"original_{field}_name"] = entry[field].get("path")
            with Image.open(BytesIO(payload)) as image:
                image.load()
                sizes.append(image.size)
                extension = ".jpg" if image.format == "JPEG" else ".png"
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Unexpected image content from dataset API.")
                if field == "mask":
                    values = np.unique(np.asarray(image.convert("L")))
                    if image.format != "PNG" or not set(values.tolist()).issubset({0, 255}) or len(values) != 2:
                        raise ValueError("Expected a genuine, nonempty binary PNG object mask.")
            path = destination / folder / f"COD10K-smoke-row{index:04d}{extension}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            record[field] = str(path.relative_to(destination))
            record[f"{field}_sha256"] = hashlib.sha256(payload).hexdigest()
        if sizes[0] != sizes[1]:
            raise ValueError("Downloaded real image and mask dimensions differ.")
        record["size_wh"] = list(sizes[0])
        records.append(record)
    if len({record["image_sha256"] for record in records}) != 6:
        raise ValueError("Duplicate smoke images found.")
    report = {"dataset": DATASET, "revision_at_download": revision, "parquet_url": source_url,
              "bytes_transferred": reader.transferred,
              "split": "train", "pairs": records,
              "source": "https://huggingface.co/datasets/nobg/COD10K",
              "declared_original_source": "https://github.com/DengPingFan/SINet",
              "use": "six-pair engineering smoke test only; not a full dataset or quality benchmark",
              "provenance_limit": "Mirror-declared origin; not independently byte-verified against the official archive."}
    (destination / "provenance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/smoke_cod10k"))
    args = parser.parse_args()
    result = fetch(args.output)
    print(json.dumps({"pairs_downloaded": len(result["pairs"]), "destination": str(args.output),
                      "source": result["source"], "use": result["use"]}, indent=2))
