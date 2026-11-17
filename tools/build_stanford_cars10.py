"""Build stanford-cars-10 from the Stanford Cars images (parquet release).

    python build_stanford_cars10.py PARQUET_DIR OUT_DIR

PARQUET_DIR  folder with train-*.parquet and test-*.parquet, the Hugging Face release
             naufalso/stanford_cars (columns image_path, x_min, y_min, x_max, y_max, car_name)
OUT_DIR      receives images/ (the original JPEG bytes, named <split>-<devkit number>.jpg) and
             annotations_boxmask.json (COCO boxes, each also written as a rectangle polygon)

Ten look-alike sedan classes. Train and test images are pooled per class, keys sorted, then
RandomState(2026).permutation per class: the first 12 go to eval, the rest to the train pool.
Class ids follow the alphabetical order of the class names.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image

SEED = 2026
N_EVAL = 12
CLASSES = [
    "Acura TL Sedan 2012", "BMW 3 Series Sedan 2012", "Buick Verano Sedan 2012",
    "Chevrolet Sonic Sedan 2012", "Ford Fiesta Sedan 2012", "Honda Accord Sedan 2012",
    "Hyundai Sonata Sedan 2012", "Mercedes-Benz E-Class Sedan 2012",
    "Mitsubishi Lancer Sedan 2012", "Toyota Camry Sedan 2012",
]


def read_rows(parquet_dir: Path) -> dict[str, list[dict]]:
    """{class name: rows}. Devkit numbers repeat across train and test, so the key carries the split."""
    rows: dict[str, list[dict]] = {n: [] for n in CLASSES}
    for f in sorted(parquet_dir.glob("*.parquet")):
        split = "train" if f.name.startswith("train-") else "test"
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["image_path", "x_min", "y_min", "x_max", "y_max", "car_name"])
            t = t.filter(pc.is_in(t["car_name"], value_set=pa.array(CLASSES)))
            for r in t.to_pylist():
                stem = Path(r["image_path"]["path"]).stem
                rows[r["car_name"]].append({"key": f"{split}-{stem}", "bytes": r["image_path"]["bytes"],
                                            "x0": r["x_min"], "y0": r["y_min"], "x1": r["x_max"], "y1": r["y_max"]})
    return rows


def main(parquet_dir: Path, out_root: Path) -> None:
    images_out = out_root / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(parquet_dir)
    images, anns = [], []
    n_eval = n_train = 0
    for cid, name in enumerate(CLASSES, start=1):
        pool = sorted(rows[name], key=lambda r: r["key"])
        perm = np.random.RandomState(SEED).permutation(len(pool))
        eval_rows = [pool[i] for i in perm[:N_EVAL]]
        train_rows = [pool[i] for i in perm[N_EVAL:]]
        n_eval += len(eval_rows)
        n_train += len(train_rows)
        for r in eval_rows + train_rows:
            fname = f"{r['key']}.jpg"
            (images_out / fname).write_bytes(r["bytes"])
            w, h = Image.open(io.BytesIO(r["bytes"])).size
            # The source boxes are 1-based inclusive pixel corners (MATLAB); COCO is 0-based x, y, w, h.
            x, y = r["x0"] - 1, r["y0"] - 1
            bw, bh = r["x1"] - r["x0"] + 1, r["y1"] - r["y0"] + 1
            i = len(images) + 1
            images.append({"id": i, "file_name": fname, "width": w, "height": h})
            anns.append({"id": i, "image_id": i, "category_id": cid, "bbox": [x, y, bw, bh],
                         "area": bw * bh, "iscrowd": 0,
                         "segmentation": [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]]})
        print(f"{name}: {len(pool)} images")
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES, start=1)]}
    (out_root / "annotations_boxmask.json").write_text(json.dumps(coco))
    print(f"wrote {len(images)} images ({n_eval} eval, {n_train} train pool) to {out_root}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
