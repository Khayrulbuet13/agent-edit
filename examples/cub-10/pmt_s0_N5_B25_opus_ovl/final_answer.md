# Final answer

# cub-10 edit summary

## What was wrong
The 10 class rows were unrelated COCO rows (person, bicycle, ..., boat), so the original head finds no birds:
macro-F1 is 0.0 on the 50 train images, and its only 4 predictions are boat false positives.

## Evidence used
- A read of all 200 queries on all 50 images (original head) shows the decoder already localises each bird.
  One "primary" query matches the labelled bird (mask IoU >= 0.72, mask score >= 0.96) and has a low
  no-object logit (5.8-8.4). Its many duplicates and part masks have no-object logits of about 10-12.
- The direction d = mean(primary) - mean(all other queries with mask score > 0.5) separates primaries from
  everything else by a wide margin. Relative to the primary mean, primaries fall in [-2.2, 0.9],
  duplicates are <= -15 and background is <= -8.4. The same holds when d is built from either half of the data.
- Species information is weak: leave-one-out nearest-class-mean accuracy on primary embeddings is 36%
  (terns, Heermann and Ivory Gull are separable; the large white-headed gulls are confused).
  Abstaining on low-confidence cases did not raise the estimated F1, so the head should always name its best class.

## Edit (class rows 0-9 only; the no-object row is unchanged)
For class c: w_c = 2*d + 2*n_c, where n_c = unit(mean primary of class c - mean primary).
b_c is set so that an average primary query gets a logit of 10 before the species term
(b_c = -2*t + 10 - 2*n_c.g, with t = mean primary projection on d and g = mean primary).
Because the no-object logit of primaries is about 6-8 and of duplicates about 10-12, the existing
no-object row still suppresses duplicates.
recipe.py computes all of this from train.read; the constants are a=2, beta=2 and Tb=10.

## Results
- Train (in-sample, inflated by prototypes built on the same images): macro-F1 0.000 -> 0.705.
  Per-image F1 is 33 better and 0 worse (delta 0.643 +- 0.067).
- Honest estimate from a split half (rows built on images 25-49, scored on images 0-24):
  macro-F1 0.254 vs 0.0 for the original. A variant with a=3, beta=3, Tb=11 gave 0.244 (no real difference).
- Retention: COCO mask mAP 63.11 -> 62.98 (-0.13, within 1 point).

## Expected effect and limits
Detection of the labelled bird should carry over well, because it uses the decoder's own primary-query signal.
Species accuracy is the bottleneck (about 30-36%), so expect a held-out macro-F1 of about 0.25-0.35.
Remaining false positives: other, unlabelled birds in the image; occasionally non-bird objects such as gravestones.

---
edited tensors: 2, changed weights: 10250 of 11275 (90.9091%), passes 13/25, code runs 33/300, finish: agent
