"""PMT-L 1280 (COCO instance) editable head on a frozen DINOv3 ViT-L/16.

Cut point: after the lateral BatchNorms. Frozen and cached: DINOv3 backbone (layers
5/11/17/23, final norm) and the 4 lateral BNs. Editable: the Plain Mask Decoder
(queries, lateral projections, 6 blocks, decoder norm, upscale, mask head, class head).
See HEAD_MAP.md.

The repo's own PlainMaskDecoder runs the head; weights are overridden with
torch.func.functional_call, so the shared module is never mutated. Needs the `pmt` extra.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from ... import config

RES = 1280
PATCH = 16
GRID = RES // PATCH
EMBED = 1024
NUM_Q = 200
NUM_BLOCKS = 6
HIDDEN = 1024  # decoder MLP width (config hidden_dim), not the backbone's 4096
NUM_HEADS = 16
NUM_PREFIX = 5  # CLS + 4 registers
NUM_CLASSES = 80
NO_OBJECT = NUM_CLASSES
LAYERS = (5, 11, 17, 23)
TOP_K = 100  # eval_top_k_instances


def _repo():
    src = str(config.pmt_root() / "src" / "image")
    if not Path(src).is_dir():
        raise FileNotFoundError(f"PMT source not found at {src}; run tools/fetch_pmt.py")
    if src not in sys.path:
        sys.path.insert(0, src)
    from models import pmd, vit

    return pmd, vit


def _checkpoint() -> dict:
    f = config.pmt_root() / "weights" / "model.safetensors"
    if not f.is_file():
        raise FileNotFoundError(f"PMT weights not found at {f}; run tools/fetch_pmt.py")
    from safetensors.torch import load_file

    return load_file(str(f))


def load_head_weights(device="cpu") -> dict:
    """Editable tensors, fp32, named as inside PlainMaskDecoder (e.g. 'class_head.weight')."""
    sd = _checkpoint()
    p = "network.decoder."
    return {k[len(p):]: v.to(device=device, dtype=torch.float32)
            for k, v in sd.items() if k.startswith(p) and k != p + "attn_mask_probs"}


def build_decoder(device="cpu"):
    """Repo decoder with the checkpoint loaded strictly; strict load catches config drift."""
    pmd, _ = _repo()
    dec = pmd.PlainMaskDecoder(
        embed_dim=EMBED, hidden_dim=HIDDEN, num_prefix_tokens=NUM_PREFIX,
        grid_size=(GRID, GRID), patch_size=(PATCH, PATCH), num_classes=NUM_CLASSES,
        num_q=NUM_Q, num_blocks=NUM_BLOCKS, masked_attn_enabled=True,
        interaction_indices=list(LAYERS), lateral_projection="mlp",
        residual_projection=True, num_heads=NUM_HEADS,
    )
    sd = _checkpoint()
    p = "network.decoder."
    dec.load_state_dict({k[len(p):]: v for k, v in sd.items() if k.startswith(p)}, strict=True)
    return dec.to(device).eval().requires_grad_(False)


class Backbone:
    """Frozen part: DINOv3 + lateral BN. Runs like the repo's eval (16-mixed autocast)."""

    def __init__(self, device="cuda"):
        _, vit = _repo()
        self.device = torch.device(device)
        self.vit = vit.ViT(img_size=(RES, RES), patch_size=PATCH, backbone_name=config.pmt_backbone(),
                           frozen_encoder=True).to(self.device).eval()
        sd = _checkpoint()
        self.bn = torch.nn.ModuleList([torch.nn.BatchNorm1d(EMBED) for _ in LAYERS])
        self.bn.load_state_dict({k[len("network.bn."):]: v for k, v in sd.items()
                                 if k.startswith("network.bn.")}, strict=True)
        self.bn = self.bn.to(self.device).eval()
        assert self.vit.backbone.num_prefix_tokens == NUM_PREFIX

    @staticmethod
    def scaled_size(hw) -> list[int]:
        f = min(RES / hw[0], RES / hw[1])
        return [round(s * f) for s in hw]

    def pixels(self, rgb: np.ndarray) -> torch.Tensor:
        """uint8 HxWx3 -> 1x3xRESxRES in [0, 1]: resize_and_pad_imgs_instance_panoptic + `/ 255`."""
        from PIL import Image

        h, w = self.scaled_size(rgb.shape[:2])
        img = np.array(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))
        x = torch.from_numpy(img).permute(2, 0, 1).to(self.device)
        return F.pad(x, [0, RES - w, 0, RES - h])[None] / 255.0

    def preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        # the ViT's own buffers, so the result matches PMT.forward bit for bit
        return (self.pixels(rgb) - self.vit.pixel_mean) / self.vit.pixel_std

    @torch.no_grad()
    def features(self, rgb: np.ndarray) -> tuple[torch.Tensor, tuple]:
        """-> (feats [4, NUM_PREFIX + GRID*GRID, EMBED] fp32, rope (cos, sin))."""
        x = self.preprocess(rgb)
        bb = self.vit.backbone
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            rope = bb.rope_embeddings(x)
            h = bb.patch_embed(x)
            feats = []
            for i, blk in enumerate(bb.blocks):
                h = blk(h, attention_mask=None, position_embeddings=rope)
                if i in LAYERS:
                    f = bb.norm(h)
                    feats.append(self.bn[len(feats)](f.permute(0, 2, 1)).permute(0, 2, 1))
                if i == LAYERS[-1]:
                    break
        return torch.cat(feats).float(), tuple(r.float() for r in rope)


def head_forward(dec, weights: dict, feats: torch.Tensor, rope, amp: bool = False):
    """Last-layer (mask_logits [B,Q,4*GRID,4*GRID], class_logits [B,Q,81]).

    feats: [4, T, EMBED] for one image or [B, 4, T, EMBED]. amp=True reproduces the repo's
    16-mixed eval; fp32 keeps small edits from being rounded away.
    """
    if feats.dim() == 3:
        feats = feats[None]
    # PMT's lateral BN hands the decoder transposed views; the same strides give the same
    # fp16 kernels, which is what makes amp=True bit-exact with the repo.
    outs = [feats[:, i].transpose(1, 2).contiguous().transpose(1, 2) for i in range(feats.shape[1])]
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        masks, classes = functional_call(dec, weights, (outs, rope), strict=False)
    return masks[-1], classes[-1]


if config.has_module("backends.pmt.head_lab"):  # names of an optional sibling module, when the tree has it
    from .head_lab import *  # noqa: F403
