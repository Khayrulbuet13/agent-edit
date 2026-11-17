"""Projects, splits and records."""

from .loader import image_hw, open_rgb
from .registry import (
    BRIDGE_PROJECTS,
    EXTRA_PROJECTS,
    N_TRAIN,
    PER_CLASS_SHOT,
    PROJECTS,
    PROMPT_VARIANT,
    check_project,
    class_prompts,
    class_table,
    manifest,
    manifest_path,
    prompt_variants,
)
from .splits import (
    eval_image_keys,
    eval_image_paths,
    extra_pool,
    gt_for_agent,
    load_split,
    pool_keys,
    records,
    seed_order,
    train_keys,
    write_agent_gt,
)

__all__ = [
    "BRIDGE_PROJECTS", "EXTRA_PROJECTS", "N_TRAIN", "PER_CLASS_SHOT", "PROJECTS", "PROMPT_VARIANT",
    "check_project", "class_prompts", "class_table", "eval_image_keys", "eval_image_paths", "extra_pool",
    "gt_for_agent", "image_hw", "load_split", "manifest", "manifest_path", "open_rgb", "pool_keys",
    "prompt_variants", "records", "seed_order", "train_keys", "write_agent_gt",
]
