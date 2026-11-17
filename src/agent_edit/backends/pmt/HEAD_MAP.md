# PMT head map

PMT-L 1280, COCO instance (`tue-mps/coco_instance_pmt_large_1280_dinov3`, MIT), on frozen DINOv3 ViT-L/16
(`facebook/dinov3-vitl16-pretrain-lvd1689m`, loaded directly; the caches were built from the ungated timm `vit_large_patch16_dinov3_qkvb.lvd1689m` re-host converted to HF format, which has identical weights). Source: `tue-mps/pmt` (MIT), commit `0e803722aa5737a242b383dec1b90c2c66b86baa` (`image/models/{pmt,pmd,layers,vit}.py`,
`image/training/mask_classification_instance.py`). A local copy sits in `src/` under the PMT root (`AGENT_EDIT_PMT_ROOT`).

## Cut point

| Part | Status | Where |
|---|---|---|
| DINOv3 ViT-L (24 blocks) | frozen, cached | `PMT.dinov3_forward` |
| Final backbone norm on layers 5, 11, 17, 23 | frozen, cached | `backbone.norm(x)` per level |
| Lateral BatchNorm x4 (`network.bn.*`, eval stats) | frozen, cached | `PMT.normalize_features`. Outside the decoder |
| RoPE (cos, sin) | fixed, computed once | depends only on the 1280x1280 input size |
| Plain Mask Decoder (143 tensors, 53.87M params) | **editable** | `PlainMaskDecoder` |

Cache per image: `[4, 6405, 1024]` fp32 (5 prefix tokens = CLS + 4 registers, then 80x80 patches), about 105 MB.

## Head facts

1. **Class head.** `class_head.weight` `(81, 1024)`, `class_head.bias` `(81,)`. Rows 0..79 = COCO classes in category-id order
   (`datasets/coco_instance.py` `CLASS_MAPPING`). **Row 80 = no object.**
2. **Backbone layers read.** Outputs of blocks 5, 11, 17, 23, each through the final backbone norm and then its lateral BN.
   All 6405 tokens of each level are used (the prefix tokens too: they flow through the decoder blocks).
3. **Preprocessing.** Scale so the long side is 1280 (`factor = min(1280/h, 1280/w)`, `round`), PIL bilinear on uint8,
   zero-pad bottom/right to 1280x1280, `/255`, mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`.
4. **Instance post-processing** (`eval_step`, last decoder layer only):
   - `softmax` over the 81 logits, drop the no-object column.
   - top-100 over the flattened (200 queries x 80 classes) scores. One query can give several instances.
   - Mask logits `(200, 320, 320)` → bilinear to 1280 → crop to the scaled size → bilinear to the image size → `> 0`.
   - score = class prob x mean sigmoid inside the mask.
   - This backend adds: softmax over the K slot rows + row 80 only, then keep score >= 0.5.
5. **Dtypes.** The checkpoint is fp32. The repo trains and evaluates with Lightning `16-mixed` (fp16 autocast) and
   `set_float32_matmul_precision("medium")`. This backend runs the backbone like the repo (fp16 autocast; LayerNorm and BN outputs
   are fp32) and caches fp32. The editable head runs in **fp32**. `head_forward(..., amp=True)` reproduces the repo's eval
   dtype for the parity check.

Other facts:
- `attn_mask_probs` = 0 for all 6 blocks, so masked attention is fully off at inference. Each block still computes an
  intermediate prediction to build the mask, and all of it is then disabled.
- Decoder MLP width is 1024 (`hidden_dim`), not 4096. `k_proj` has no bias.

## Editable tensors

| Name (`<i>` = index) | Count | Shape | Params | Role |
|---|---|---|---|---|
| `lateral_projections.<i>.0.weight` / `.0.bias` | 4 | (512, 1024) / (512,) | 2.10M | per-level residual MLP on backbone features, in |
| `lateral_projections.<i>.2.weight` / `.2.bias` | 4 | (1024, 512) / (1024,) | 2.10M | per-level residual MLP, out |
| `scalers.0..3` | 4 | (1024,) | 4K | per-level channel scale before the 4 levels are summed |
| `q.weight` | 1 | (200, 1024) | 0.20M | learned queries, prepended to the patch tokens |
| `blocks.<i>.norm1.*`, `norm2.*` | 6 | (1024,) | 25K | pre-norms |
| `blocks.<i>.attention.{q,k,v,o}_proj.weight` | 6 | (1024, 1024) | 25.2M | joint self-attention over queries + patches (RoPE on patches) |
| `blocks.<i>.attention.{q,v,o}_proj.bias` | 6 | (1024,) | 18K | |
| `blocks.<i>.layer_scale{1,2}.lambda1` | 6 | (1024,) | 12K | residual branch scales |
| `blocks.<i>.mlp.{up,down}_proj.*` | 6 | (1024, 1024) | 12.6M | block MLP |
| `decoder_norm.*` | 1 | (1024,) | 2K | final LayerNorm, applied before both heads |
| `class_head.weight` / `.bias` | 1 | (81, 1024) / (81,) | 83K | class logits; row 80 = no object |
| `mask_head.<i>.weight` / `.bias` (i = 0, 2, 4) | 3 | (1024, 1024) | 3.15M | 3-layer MLP on queries → mask embedding |
| `upscale.<i>.conv1.*` (ConvTranspose 2x2) | 2 | (1024, 1024, 2, 2) | 8.39M | patch features 80 → 320 |
| `upscale.<i>.conv2.weight` (depthwise 3x3), `upscale.<i>.norm.*` | 2 | (1024, 1, 3, 3) | 22K | |

Mask logit for query q = `mask_head(q) · upscale(patches)`, so masks depend on the queries, the patch path and the mask head.

## Parity status

| Check | Result |
|---|---|
| Strict load of `network.decoder.*` into the repo's `PlainMaskDecoder` | pass |
| `head_forward` (functional_call) vs the module forward, CPU fp32, synthetic features | max abs diff 0.0 |
| Cached path vs repo `PMT.forward` on 10 COCO val images, repo dtype (fp16) | bit-exact (max diff 0.0), after passing features with the repo's strides |
| fp32 head vs fp16 head, same images | class logits differ <= 0.17, mask sign flips <= 1e-4 of pixels |
| Official gated `facebook/dinov3-vitl16-pretrain-lvd1689m` vs the converted timm copy | every weight equal (max diff 0.0), forward equal (0.0); only the unused init `layerscale_value` differs |
| COCO val2017 mask mAP, cached fp32 path, 5000 images (ref 49.0) | **49.01** (AP50 73.48) |
| Cached `predict` vs live postprocess | same queries on 5/5 images, score diff <= 1.3e-6 |
