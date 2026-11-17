"""The interface every model backend implements for the harness."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Backend(ABC):
    """`weights` args are full name -> float32 CPU tensor dicts (read-only), or None for the
    original head. Nothing here may return raw GT masks or eval images.

    Keys of a generate() result that start with '_' are private: the harness drops them before the
    agent sees the result, but passes the full result to render_errors()."""

    # One line that explains the colors and marks of the render_errors() pictures.
    overlay_legend: str = ""
    # render_errors() honours sheet_k (--overlay-sheet); the harness refuses the flag otherwise.
    supports_sheet: bool = False
    # Set by the harness when class names are hidden: {real name: shown name}. Text drawn on
    # pictures must go through shown_name(), since the harness can only rewrite text, not pixels.
    class_labels: dict | None = None

    @abstractmethod
    def initial_weights(self) -> dict:
        """name -> float32 CPU tensor of every editable weight."""

    @abstractmethod
    def generate(self, weights, n_images: int, which: str = "train") -> dict:
        """Run the head on n_images train images and score them. Return a dict with
        'macro_f1', 'per_image_f1' (list or dict), 'n_pred', 'n_gt', 'per_class_f1'.
        Extra JSON-able keys are fine."""

    @abstractmethod
    def read(self, weights, spec: dict) -> dict:
        """Read internals. spec always has which='train'. Big arrays end up as .npy files in the run's work folder."""

    @abstractmethod
    def describe(self) -> str:
        """Plain-text description of the head, its tensor names and the read() spec."""

    def frozen_patterns(self) -> list:
        """fnmatch patterns of tensor names save_edit must refuse."""
        return []

    def project_info(self) -> dict:
        """Optional: {'classes': [...], 'task': '...', 'n_shot': N, 'model_intro': '...'} for the prompt."""
        return {}

    def render_errors(self, result: dict, out_dir, sheet_k: int | None = None) -> list[str] | None:
        """Pictures of the train errors in one generate() result, for runs served with --overlays.

        `result` is what generate() returned, private '_' keys included. Write PNG files under
        `out_dir` (it may not exist yet) and return their paths. Return None if this backend has
        no pictures; the harness then refuses --overlays at start.

        `sheet_k` is set (--overlay-sheet) only when `supports_sheet` is True: also write
        `out_dir/sheet.png`, tiling the `sheet_k` worst-scored images of this result, worst first."""
        return None

    def retention(self, weights) -> dict | None:
        """Score of `weights` (None = the original head) on a small fixed held-out set of the
        model's ORIGINAL task, for the harness's fwd {"which": "retain"} (runs served with --retention).

        Return {'score': points (0-100, higher is better), 'metric': short name, 'n_images': int,
        'about': one sentence for the agent on what is scored}. Same images every call.
        Return None if this backend has no such set; the harness then refuses --retention at start."""
        return None

    def shown_name(self, name: str) -> str:
        return (self.class_labels or {}).get(name, name)

    def secret_words(self) -> list[str]:
        """Extra words to scrub from agent-visible text when names are hidden or permuted, beyond
        the class list (e.g. a merged project's parent species names, spelled out in its image
        keys). Empty by default."""
        return []

    def reading_task(self) -> dict | None:
        """For runs served with --reading: name-shuffled class rows plus a labelled pool to tell
        them apart, instead of a head to edit.

        Return {'rows': (K, D) float array already shuffled, 'key': {row index: true class name}
        (never shown to the agent), 'pool_features': (N, D) float array, 'pool_names': [str] * N
        real names of the pool, in the same feature space as the rows}.
        Return None if this backend has no reading task; the harness then refuses --reading."""
        return None
