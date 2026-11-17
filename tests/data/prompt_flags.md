# TASK

Custom intro. The project is "toy". Its classes: (see the head description).
Toy task.
The presence gate is already open: the head you start from has its presence logit pushed high for every image and prompt, so presence is about 1 and every query is scored by its class logit alone. The failures that remain are about which instances are the project's objects and classes, not about whether the class is present.


Your goal: make this head better at the project's task, as measured by instance macro-F1 on held-out images.

You do it by EDITING THE WEIGHTS DIRECTLY. Find out what goes wrong, work out which parts of the head are responsible, and change them in a way you can justify with evidence. No training.

# HOW IT IS MEASURED

After you finish, your edited head and the original head are scored on held-out images of this project. You never see those images or their labels. The score is instance macro-F1: per class, predicted instances are matched to true instances at IoU 0.5 and F1 is computed; then it is averaged over classes. Only weight values can change: same tensor names, shapes and float32 dtype.

# WHAT YOU HAVE

Work from the folder `/REPO`. Every command below is run from there. All commands print the answer and block until it arrives (a forward pass can take minutes).

1. A persistent Python kernel over the weights (CPU, no GPU, no internet). Variables survive between runs, like a notebook. You may run code at most 300 times. One run may take up to 600 s; output is cut at 12,345 characters.
   Run code:   `PY -m agent_edit.harness exec /RUN/x -c 'print(len(NAMES))'`
   Longer code: `PY -m agent_edit.harness exec /RUN/x - <<'EOF'` ... your code ... `EOF` (or `--file path.py`).
   If the last line is an expression, its value is shown.
   Predefined: `torch`, `np`, `W` (dict: tensor name -> current float32 CPU tensor, edits applied), `NAMES`, `load(name)` (copy of the current tensor), `load_original(name)`, `save_edit(name, tensor)`, `drop_edit(name)`, `edited()` (saved edits and how many elements each changes), `WORK` (a folder you can write to: `open()` and file paths must stay inside it; use absolute paths like `WORK + '/a.npy'`). Gradients are off.
   `save_edit(name, tensor)` submits an edit. It refuses: unknown names, wrong shape or dtype (must be float32), NaN or inf, a tensor that is just the original times one constant, no-change edits, and frozen tensors. A refused edit raises an error that says why.
2. A forward-pass tool with a budget of 25 passes. Only this tool runs the head on images.
   `PY -m agent_edit.harness fwd /RUN/x --json '{"type": "generate", "model": "edited", "n_images": 20}'`
     Runs the head on n_images training images (1 to 20) and scores them: macro-F1, per-image F1, number of predictions and true instances, per-class F1. "model" is "original", "edited" (your saved edits) or "both" (same images for both, with a paired difference and its standard error).
   `PY -m agent_edit.harness fwd /RUN/x --json '{"type": "read", "model": "edited", "n_images": 20, ...}'`
     Reads internals (for example decoder query embeddings and class logits, matched to true instances or not). Exact options are in the head description below. Big arrays are saved as .npy files under `/RUN/x/work/fwd_<k>/` (the path is printed); load them in the kernel with `np.load`.
   One pass is one request of any size. A request that fails does not count. Image choice is the same for every request with the same n_images, so paired comparisons are fair. You only ever see training images; the held-out images never reach you. The labelled data is exactly 1 training image: these 1 are the only labelled images there are, and the only ones the tools can run on.
3. `PY -m agent_edit.harness status /RUN/x` shows passes and code runs used.

# THE HEAD

Toy MLP: fc1.weight (16x4), fc1.bias (16), fc2.weight (3x16), fc2.bias (3). read options: n_images -> hidden (n,16), logits (n,3), matched_class (n,).

# RULES

Not allowed:
(a) Gradient descent or any other training: no backward passes, autograd, optimizers, requires_grad, torch.func gradients.
(b) Iterative fitting of weights to a loss or to labels, and closed-form regression or least-squares fits of weights to labels. You may analyse, measure and compute simple statistics, and make edits you can explain (zero or shift a unit, remove a direction, change a bias, swap rows, set one value from a measured number). Edits must come from understanding, not from fitting a model to the labels.
(c) Running the head yourself in the kernel (head_forward, loading a model, re-implementing its forward pass on images). Only the fwd tool runs it.
(d) Importing os, subprocess, socket, urllib, requests and similar, opening files outside WORK, internet access.
(e) Whole-tensor rescales and edits to frozen tensors (refused by save_edit).
Code that breaks (a), (c) or (d) is refused before it runs. These checks are not perfect: every code block you run is logged word for word and will be audited, so follow the rules in spirit as well.

# HOW TO WORK

1. Diagnose first: run `generate` on the original head, look at per-class and per-image F1, find the biggest failure type you can explain (missed instances, false positives, wrong class, duplicates).
2. Use `read` to see what the head does on those cases; show that a part causes the failure before you edit it. Try several edit strengths.
3. Compare original and edited on the SAME images ("model": "both"). Keep an edit only if the gain is clearly larger than its standard error. Confirm on images you did not use to choose the edit when the image pool is big enough; when it is small, only keep edits you can explain, and check that most images improve (the paired result counts better and worse images).
4. Plan your 25 passes. State the hypothesis of each code run in its first comment line, print compact summaries, never whole arrays.
5. Keep notes in `/RUN/x/work/notes.md`.

# HOW IT ENDS

The session continues until you call final. After the 25 passes are used, fwd returns an error: call final. You may also call final earlier. If all passes and all code runs are used, the run ends by itself.
Before final, write a short summary (what you changed, the evidence, why you expect it to help) to a file and submit:
`PY -m agent_edit.harness final /RUN/x --summary-file /RUN/x/work/summary.md`
The tensors saved with save_edit at that moment are your submission.
