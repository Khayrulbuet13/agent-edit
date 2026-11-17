"""Toy backend for tests and dry runs:  --backend agent_edit.backends.fake:FakeBackend"""

from __future__ import annotations

from .base import Backend

CLASSES = ["a", "b", "c"]
MODEL_INTRO = ("You are given the weights of a toy network, a 2-layer MLP that labels a 4-number point "
               "with one of its classes.")


class FakeBackend(Backend):
    """2-layer MLP; the task is to classify 4-d points into 3 classes made by a hidden teacher.
    `seed` / `n_shot` draw another train pool from the same teacher (for recipe replay)."""

    overlay_legend = "one square per wrong point: left = true class, right = predicted class"
    supports_sheet = True

    def __init__(self, seed=None, n_shot=None, classes=None, **_):
        import torch

        g = torch.Generator().manual_seed(0)
        self._w = {"fc1.weight": torch.randn(16, 4, generator=g), "fc1.bias": torch.zeros(16),
                   "fc2.weight": torch.randn(3, 16, generator=g), "fc2.bias": torch.zeros(3)}
        self.x = torch.randn(512, 4, generator=g)
        t1, t2 = torch.randn(16, 4, generator=g), torch.randn(3, 16, generator=g)
        self.teacher = lambda x: (torch.relu(x @ t1.T) @ t2.T).argmax(1)
        self.y = self.teacher(self.x)
        # drawn after the default train pool, so the pool is the same whether or not they exist
        self.x_eval, self.x_retain = torch.randn(256, 4, generator=g), torch.randn(64, 4, generator=g)
        if seed is not None or n_shot is not None:
            gs = torch.Generator().manual_seed(1000 + int(seed or 0))
            self.x = torch.randn(int(n_shot or 512), 4, generator=gs)
            self.y = self.teacher(self.x)
        self.classes = classes.split(",") if isinstance(classes, str) else (classes or CLASSES)
        self.drawn_labels: list[str] = []

    def initial_weights(self):
        return {k: v.clone() for k, v in self._w.items()}

    def n_images(self) -> int:
        return len(self.x)

    def _fwd(self, w, n, x=None):
        w = w or self._w
        x = self.x[:n] if x is None else x
        h = (x @ w["fc1.weight"].T + w["fc1.bias"]).relu()
        return h, h @ w["fc2.weight"].T + w["fc2.bias"]

    def generate(self, weights, n_images, which="train"):
        _, lg = self._fwd(weights, n_images)
        pred = lg.argmax(1)
        ok = (pred == self.y[:n_images]).float()
        per_class = {self.classes[c]: float(ok[self.y[:n_images] == c].mean()) for c in range(3)}
        return {"macro_f1": float(ok.mean()), "per_image_f1": ok.tolist(),
                "n_pred": [1] * n_images, "n_gt": [1] * n_images, "per_class_f1": per_class,
                "_pred": pred.tolist(), "_true": self.y[:n_images].tolist()}

    def read(self, weights, spec):
        if spec.get("fail"):
            raise RuntimeError("toy failure")
        if spec.get("leak"):
            return {"gt_masks": [1]}
        n = int(spec.get("n_images", 32))
        h, lg = self._fwd(weights, n)
        return {"hidden": h.numpy(), "logits": lg.numpy(), "matched_class": self.y[:n].numpy(),
                "classes": [{"class": k, "class_name": c} for k, c in enumerate(self.classes)]}

    def render_errors(self, result, out_dir, sheet_k=None):
        from pathlib import Path

        import numpy as np
        from PIL import Image, ImageDraw

        from ..harness.overlays import draw_sheet, worst_k

        wrong = [(t, p) for t, p in zip(result["_true"], result["_pred"]) if t != p]
        im = Image.new("RGB", (64, 16 * max(1, len(wrong))), "white")
        d = ImageDraw.Draw(im)
        self.drawn_labels = []
        for r, (t, p) in enumerate(wrong):
            self.drawn_labels.append(f"{self.shown_name(self.classes[t])}->{self.shown_name(self.classes[p])}")
            d.rectangle([0, 16 * r, 15, 16 * r + 15], fill=(40 * t, 0, 0))
            d.rectangle([32, 16 * r, 47, 16 * r + 15], fill=(0, 40 * p, 0))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        p = Path(out_dir) / "errors.png"
        im.save(p)
        if sheet_k:
            f1s = result.get("per_image_f1") or []
            order = worst_k(f1s, sheet_k)
            tiles = []
            for i in order:
                t, pr = result["_true"][i], result["_pred"][i]
                sq = np.zeros((32, 32, 3), np.uint8)
                sq[:, :16] = (40 * t, 0, 0)
                sq[:, 16:] = (0, 40 * pr, 0)
                label = f"#{i} f1={f1s[i]:.2f} true:{self.shown_name(self.classes[t])} pred:{self.shown_name(self.classes[pr])}"
                tiles.append((sq, label))
            draw_sheet(tiles, out_dir)
        return [str(p)]

    def retention(self, weights):
        """Agreement (%) with the original head on 64 other points."""
        _, lo = self._fwd(None, 0, self.x_retain)
        _, le = self._fwd(weights, 0, self.x_retain)
        return {"score": 100.0 * float((lo.argmax(1) == le.argmax(1)).float().mean()), "metric": "agreement",
                "n_images": len(self.x_retain), "about": "agreement (%) with the original head on 64 held-out points."}

    def reading_task(self):
        """Smoke-test stand-in for the real reading task: fc2.weight's 3 rows, name-shuffled, and
        the hidden-layer activations of 60 train points as the labelled pool."""
        import numpy as np

        names = list(self.classes)
        perm = np.random.RandomState(1).permutation(len(names))
        rows = self._w["fc2.weight"].numpy()
        h, _ = self._fwd(None, min(60, len(self.x)))
        pool_names = [names[int(c)] for c in self.y[: h.shape[0]].tolist()]
        return {"rows": rows[perm], "key": {int(i): names[int(p)] for i, p in enumerate(perm)},
                "pool_features": h.numpy(), "pool_names": pool_names}

    def evaluate(self, weights) -> dict:
        """Held-out accuracy, the toy's hidden eval (used by replay)."""
        _, lg = self._fwd(weights, 0, self.x_eval)
        return {"macro_f1": float((lg.argmax(1) == self.teacher(self.x_eval)).float().mean()), "n_images": len(self.x_eval)}

    def describe(self):
        return ("Toy MLP: fc1.weight (16x4), fc1.bias (16), fc2.weight (3x16), fc2.bias (3). "
                "read options: n_images -> hidden (n,16), logits (n,3), matched_class (n,).")

    def frozen_patterns(self):
        return ["fc2.bias"]

    def project_info(self):
        return {"classes": list(self.classes), "model_intro": MODEL_INTRO}
