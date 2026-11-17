"""Cache PMT's output on COCO val2017 images: the input of the retention guard.

    python tools/build_coco_guard_cache.py [--limit 20] [--device cuda] [--out DIR]
                                           [--coco-root DIR] [--pmt-root DIR]

The guard scores the original COCO head, with the edited rows appended, on the first 20 val2017
image ids (sorted). `--limit 20` is enough for it. `--limit 0` caches all 5000 images, which the
full retention report (`agent-edit retention`) reads. Needs a GPU; 20 images take about a minute.

Reads:  --coco-root (default $AGENT_EDIT_COCO_ROOT, else $AGENT_EDIT_DATA_RAW/coco): annotations/instances_val2017.json
        and val2017/*.jpg. --pmt-root (default $AGENT_EDIT_PMT_ROOT, else $AGENT_EDIT_CACHE/pmt): the PMT source and
        weights, see tools/fetch_pmt.py.
Writes: --out (default <pmt-root>/cache/coco_val2017): one <image id>.pt per image. Cached images
        are skipped, so a stopped run resumes.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=20, help="first N val2017 ids, sorted; 0 = all (default 20)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, help="cache folder; default <pmt-root>/cache/coco_val2017")
    ap.add_argument("--coco-root", type=Path, help="folder with annotations/ and val2017/")
    ap.add_argument("--pmt-root", type=Path, help="folder with src/, weights/ and cache/")
    a = ap.parse_args(argv)

    # config reads these when it first needs a path, so they are set before the import below
    if a.coco_root:
        os.environ["AGENT_EDIT_COCO_ROOT"] = str(a.coco_root)
    if a.pmt_root:
        os.environ["AGENT_EDIT_PMT_ROOT"] = str(a.pmt_root)

    from agent_edit.analysis import retention

    done = retention.build_cache(limit=a.limit or None, root=a.out, device=a.device)
    print(f"wrote {len(done)} entries to {a.out or retention.guard_root()}")


if __name__ == "__main__":
    main()
