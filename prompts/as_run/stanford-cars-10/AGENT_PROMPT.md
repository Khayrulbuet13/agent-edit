# TASK

You are given the class head of PMT, an instance segmentation model (a frozen DINOv3 image encoder and a mask decoder) that returns a mask, a class and a score for each object it finds. The project is "stanford-cars-10". Its classes: Acura TL Sedan 2012, BMW 3 Series Sedan 2012, Buick Verano Sedan 2012, Chevrolet Sonic Sedan 2012, Ford Fiesta Sedan 2012, Honda Accord Sedan 2012, Hyundai Sonata Sedan 2012, Mercedes-Benz E-Class Sedan 2012, Mitsubishi Lancer Sedan 2012, Toyota Camry Sedan 2012.
Photos of cars. Each image shows one labelled car; the 10 classes are look-alike sedan makes and models. Find the car and name its make and model. Scored with box IoU (the box around a predicted mask).

Your goal: make this head better at the project's task, as measured by instance macro-F1 on held-out images.

You do it by EDITING THE WEIGHTS DIRECTLY. Find out what goes wrong, work out which parts of the head are responsible, and change them in a way you can justify with evidence. No training.

# HOW IT IS MEASURED

After you finish, your edited head and the original head are scored on held-out images of this project. You never see those images or their labels. The score is instance macro-F1: per class, predicted instances are matched to true instances at IoU 0.5 and F1 is computed; then it is averaged over classes. Only weight values can change: same tensor names, shapes and float32 dtype.

# WHAT YOU HAVE

Work from the folder `<TREE_DIR>`. Every command below is run from there. All commands print the answer and block until it arrives (a forward pass can take minutes).

1. A persistent Python kernel over the weights (CPU, no GPU, no internet). Variables survive between runs, like a notebook. You may run code at most 300 times. One run may take up to 900 s; output is cut at 10,000 characters.
   Run code:   `<PYTHON> -m wsp.harness exec <RUN_DIR> -c 'print(len(NAMES))'`
   Longer code: `<PYTHON> -m wsp.harness exec <RUN_DIR> - <<'EOF'` ... your code ... `EOF` (or `--file path.py`).
   If the last line is an expression, its value is shown.
   Predefined: `torch`, `np`, `W` (dict: tensor name -> current float32 CPU tensor, edits applied), `NAMES`, `load(name)` (copy of the current tensor), `load_original(name)`, `save_edit(name, tensor)`, `drop_edit(name)`, `edited()` (saved edits and how many elements each changes), `WORK` (a folder you can write to: `open()` and file paths must stay inside it; use absolute paths like `WORK + '/a.npy'`). Gradients are off. With them: `train` (the train data as a recipe sees it; see HOW IT ENDS) and `check_recipe()` (runs WORK/recipe.py the way the final check does and says whether it gives back your saved edits).
   `save_edit(name, tensor)` submits an edit. It refuses: unknown names, wrong shape or dtype (must be float32), NaN or inf, a tensor that is just the original times one constant, no-change edits, and frozen tensors. A refused edit raises an error that says why.
2. A forward-pass tool with a budget of 25 passes. Only this tool runs the head on images.
   `<PYTHON> -m wsp.harness fwd <RUN_DIR> --json '{"type": "generate", "model": "edited", "n_images": 50}'`
     Runs the head on n_images training images (1 to 50) and scores them: macro-F1, per-image F1, number of predictions and true instances, per-class F1. "model" is "original", "edited" (your saved edits) or "both" (same images for both, with a paired difference and its standard error). Every generate also draws pictures of the errors on the images it scored (PNG files; their paths are under "overlays" in the result; outlines: green = correct, red = missed true instance, yellow = false mask, blue = right object but wrong class). Look at them: they show where the head fails and what it confuses.
   `<PYTHON> -m wsp.harness fwd <RUN_DIR> --json '{"type": "read", "model": "edited", "n_images": 50, ...}'`
     Reads internals (for example decoder query embeddings and class logits, matched to true instances or not). Exact options are in the head description below. Big arrays are saved as .npy files under `<WORK_DIR>/fwd_<k>/` (the path is printed); load them in the kernel with `np.load`.
   `<PYTHON> -m wsp.harness fwd <RUN_DIR> --json '{"which": "retain"}'`
     Retention check: COCO mask mAP (x100) on 20 fixed COCO val2017 images, for the model's original 80-class COCO head with your class rows added to it and its no-object row replaced by yours. It shows how much your rows take over objects the original model already finds. It returns the original head's score, your edited head's score and their difference, in points. One pass.
   One pass is one request of any size. A request that fails does not count. Image choice is the same for every request with the same n_images, so paired comparisons are fair. You only ever see training images; the held-out images never reach you. The labelled data is exactly 50 training images: these 50 are the only labelled images there are, and the only ones the tools can run on.
3. `<PYTHON> -m wsp.harness status <RUN_DIR>` shows passes and code runs used.

# THE HEAD

Frozen and not editable: the DINOv3 image encoder and the whole mask decoder. For each image the decoder gives 200 queries; each query has a fixed mask, a fixed mask score (how confident the mask is, 0..1) and a fixed 1024-number embedding. You edit only the class head that turns a query embedding into class logits: 2 tensors, 11 rows. `NAMES` lists them.
- `class_rows.weight` (11, 1024) and `class_rows.bias` (11,). Rows 0..10 - 1 are the project classes: row 0 = class id 1 = Acura TL Sedan 2012; row 1 = class id 2 = BMW 3 Series Sedan 2012; row 2 = class id 3 = Buick Verano Sedan 2012; row 3 = class id 4 = Chevrolet Sonic Sedan 2012; row 4 = class id 5 = Ford Fiesta Sedan 2012; row 5 = class id 6 = Honda Accord Sedan 2012; row 6 = class id 7 = Hyundai Sonata Sedan 2012; row 7 = class id 8 = Mercedes-Benz E-Class Sedan 2012; row 8 = class id 9 = Mitsubishi Lancer Sedan 2012; row 9 = class id 10 = Toyota Camry Sedan 2012. Row 10 is the "no object" class.

The model was trained on COCO (80 everyday classes). The class rows you start from are rows of its COCO class head (for COCO-like projects, the matching COCO classes; otherwise the first 10 COCO classes, whose meaning is unrelated to this project) and the no-object row of that head.

For one image the head does:
1. logits = embedding @ class_rows.weight.T + class_rows.bias, for each of the 200 queries: (11) numbers per query.
2. probabilities = softmax over those 11 logits (the no-object class takes part in the softmax).
3. For every (query, class) pair with class < 10: score = probability of that class x the query's mask score. The 100 best pairs are kept (one query can appear with several classes).
4. Pairs with score >= 0.5 become predictions: that query's mask, labelled with that class.
Everything runs in float32, so small edits take effect.

SCORING. `generate` runs the head on each image, and per class matches predictions to true instances at IoU 0.5 (greedy, best score first). Boxes, not masks, are matched for this project: the box around each predicted mask against the true box. macro_f1 = per-class F1 (counts pooled over images), averaged over classes with a true instance or a prediction. per_image_f1 = F1 averaged over the classes that have a true instance or a prediction in that image (1.0 if none). There are only 50 training images (n_images is capped at it); n_images=k gives the first k of a fixed random order, so the same k gives the same images. A generate takes seconds.

READ. {"type": "read", "model": "edited", "what": "queries", ...} returns one row per (image, query) for the `top` best-scoring queries of each image. Options, all optional: "images": [0, 3] (indices 0..49; default the first n_images, n_images default 3), "top": 1..200 (default 30; 200 = every query, including ones that match nothing), "min_score". Arrays (saved as .npy, one row each): embedding (R,1024) = the query vectors the class head reads; image_index; query_index (0..199); logits (R,11); score (best class probability x mask score); pred_class; mask_score; box (xyxy as fractions of the image); matched_class (class id of the true instance whose mask overlaps this query's mask with IoU >= 0.5, else -1); matched_iou. Also gt_counts (true instances per class per image). matched_class not in (-1, pred_class): right object, wrong class. matched_class == -1: no true object there.

# RULES

Not allowed:
(a) Gradient descent or any other training: no backward passes, autograd, optimizers, requires_grad, torch.func gradients.
(b) Iterative fitting of weights to a loss or to labels, and closed-form regression or least-squares fits of weights to labels. You may analyse, measure and compute simple statistics, and make edits you can explain (zero or shift a unit, remove a direction, change a bias, swap rows, set one value from a measured number). Edits must come from understanding, not from fitting a model to the labels.
(c) Running the head yourself in the kernel (head_forward, loading a model, re-implementing its forward pass on images). Only the fwd tool runs it.
(d) Importing os, subprocess, socket, urllib, requests and similar, opening files outside WORK, internet access.
(e) Whole-tensor rescales and edits to frozen tensors (refused by save_edit).
(f) Breaking what the model already does: keep retention within 1 point of the original (the retain check). Run it on your final edits before you call final.
Code that breaks (a), (c) or (d) is refused before it runs. These checks are not perfect: every code block you run is logged word for word and will be audited, so follow the rules in spirit as well.

# HOW TO WORK

1. Diagnose first: run `generate` on the original head, look at per-class and per-image F1, find the biggest failure type you can explain (missed instances, false positives, wrong class, duplicates).
2. Use `read` to see what the head does on those cases; show that a part causes the failure before you edit it. Try several edit strengths.
3. Compare original and edited on the SAME images ("model": "both"). Keep an edit only if the gain is clearly larger than its standard error. Confirm on images you did not use to choose the edit when the image pool is big enough; when it is small, only keep edits you can explain, and check that most images improve (the paired result counts better and worse images).
4. Plan your 25 passes. State the hypothesis of each code run in its first comment line, print compact summaries, never whole arrays.
5. Keep notes in `<WORK_DIR>/notes.md`.

# HOW IT ENDS

The session continues until you call final. After the 25 passes are used, fwd returns an error: call final. You may also call final earlier. If all passes and all code runs are used, the run ends by itself.
Your submission also includes the method, as a program: write `<WORK_DIR>/recipe.py` that defines `edit(train, weights)` and returns a dict tensor name -> new tensor, the same tensors you saved with save_edit. `weights` holds copies of the ORIGINAL tensors. `train` is the kernel's `train`: `train.read(spec)` returns what the fwd read with that spec returned on the original head (spec = the same JSON as the request, as a dict; big arrays come back as numpy arrays), and `train.read(spec, weights=w)` what it returned on the weights w (a dict of tensors; missing names are the original). It only serves reads you already ran with the fwd tool, so it costs no pass. recipe.py may use torch, np and math, and may not read or write files. At final it is run on this run's train data, and it must give back your saved edits (max abs difference 1e-5); later it is run on other train sets with no agent. So compute everything that depends on the images (class means, directions, thresholds) from `train` inside edit(); plain constants (a strength, a scale) can be written in. Test it with `check_recipe()` before final.
Before final, write a short summary (what you changed, the evidence, why you expect it to help) to a file and submit:
`<PYTHON> -m wsp.harness final <RUN_DIR> --summary-file <WORK_DIR>/summary.md`
The tensors saved with save_edit at that moment are your submission.
