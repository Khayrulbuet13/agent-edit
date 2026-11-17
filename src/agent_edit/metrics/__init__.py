"""Model-agnostic scoring: matcher, macro-F1, bootstrap, mask AP, prediction docs."""

from .bootstrap import BOOT, bootstrap_diff, f1_ci
from .docs import encode_masks, make_doc, make_instances, write_pred_file
from .f1 import (
    METRIC,
    counts,
    f1_from_counts,
    image_stats,
    per_image_f1,
    score,
    score_train,
    summarize,
    train_report,
)
from .match import IOU_THR, greedy_match, gt_by_class, prf, rle_of

__all__ = [
    "BOOT", "IOU_THR", "METRIC", "bootstrap_diff", "counts", "encode_masks", "f1_ci", "f1_from_counts",
    "greedy_match", "gt_by_class", "image_stats", "make_doc", "make_instances", "per_image_f1", "prf",
    "rle_of", "score", "score_train", "summarize", "train_report", "write_pred_file",
]
