# agent-edit

A coding agent adapts a vision model to new classes from a few labeled images, without gradients.
The backbone of the model stays frozen. The agent works in a Python kernel over the editable
tensors of the model's class head. It has a small budget of forward passes on the training
images, it can look at pictures of its own errors, and it writes its weight edits as code. When it
calls `final`, the saved tensors are its answer. They are scored on held-out images it never
sees. It can also hand in a `recipe.py` that rebuilds the edit with no agent.

This repository holds the system that runs the agent (the harness), the scorer, a fixed
agent-free formula distilled from the agents' programs, and gradient baselines trained on the
same labels. The real backend is PMT, a mask transformer on a frozen DINOv3 backbone. A toy
backend lets you try everything on a CPU.

Two names are used for the methods:

- **Agent edit**: the weights the agent returns.
- **Agent formula**: a fixed formula, no agent. It is a nearest-class-mean head with a gate term
  and removal of within-class directions. Its hyperparameters are picked by cross-validation on
  the training images. A guard checks that the original COCO classes are kept.

## What is in the box

| Folder | What |
| --- | --- |
| `src/agent_edit/harness/` | the agent's tools: a kernel with guards, a file-queue server and client, the prompt |
| `src/agent_edit/backends/` | `pmt/` is PMT on frozen DINOv3 (the head is split into cached frozen features and an editable head); `fake.py` is the toy backend |
| `src/agent_edit/metrics/` | instance macro-F1 at IoU 0.5, paired bootstrap, mask AP |
| `src/agent_edit/baselines/` | the Agent formula and the gradient baselines |
| `src/agent_edit/analysis/` | recipe replay, and the check that the original COCO classes are kept |
| `src/agent_edit/data/` | the sets and their train and eval splits |
| `configs/datasets/` | one small file per set |
| `data/` | manifests and annotation files of the sets built for this work. No images. See `data/README.md` |
| `tools/` | scripts that fetch PMT, build a cache, and build two image sets |
| `prompts/as_run/` | the prompts of the reported runs |
| `examples/` | one finished agent run, with its edit and `recipe.py` |
| `tests/` | CPU tests |

## Install

The package is managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev                                                   # CPU: harness, scorer, toy backend, tests
UV_PROJECT_ENVIRONMENT=.venv-pmt uv sync --extra pmt --extra dev --python 3.13   # PMT backend
```

The `pmt` extra pins the torch, transformers and numpy versions the reported runs used, so it
lives in its own environment. Commands below that start with `.venv-pmt/bin/` need it. The
others run in the CPU environment.

## Quickstart on a CPU

This runs one agent session by hand on the toy backend. You play the agent. Nothing to download.

```bash
RUN=/tmp/agent-edit-demo
uv run agent-edit serve --run-dir $RUN --budget 3 --max-exec 10 \
    --backend agent_edit.backends.fake:FakeBackend --project toy &
while [ ! -d $RUN/queue ]; do sleep 1; done; sleep 3          # wait for the server

uv run agent-edit exec   $RUN -c 'print(NAMES)'
uv run agent-edit exec   $RUN -c "w = W['fc1.weight'].clone(); w[0] *= -1; save_edit('fc1.weight', w)"
uv run agent-edit fwd    $RUN --json '{"type": "generate", "model": "both", "n_images": 64}'
uv run agent-edit status $RUN
uv run agent-edit final  $RUN --summary 'flipped unit 0'
wait                                                          # the server stops after final
cat $RUN/AGENT_PROMPT.md $RUN/final_answer.md
```

`serve` writes `AGENT_PROMPT.md`, the task an agent gets. Every code block the agent runs is
logged word for word in `log/turns.jsonl`.

The tests run on a CPU with no data:

```bash
uv run pytest -m "not slow"
```

## Data setup

Data, caches, runs and results live outside the repository. The code finds them through
environment variables, and stops with a clear message when one it needs is missing.

| Variable | What it points to |
| --- | --- |
| `AGENT_EDIT_DATA` | the `data/` folder of this repository: `manifests/<set>.json` and the annotation files |
| `AGENT_EDIT_DATA_RAW` | the folder of source images, one subfolder per dataset (layout in `data/README.md`) |
| `AGENT_EDIT_CACHE` | frozen-feature caches and model weights |
| `AGENT_EDIT_RUNS` | agent runs, `<set>/<run>/` |
| `AGENT_EDIT_RESULTS` | scores and baseline outputs |
| `AGENT_EDIT_PMT_ROOT` | PMT source code, weights and cache. Default `$AGENT_EDIT_CACHE/pmt` |

Optional: `AGENT_EDIT_COCO_ROOT` (COCO 2017, default `$AGENT_EDIT_DATA_RAW/coco`) and
`AGENT_EDIT_PMT_BACKBONE` (DINOv3 Hugging Face id or local path). `agent-edit env` lists the four
variables every command needs (`_DATA`, `_CACHE`, `_RUNS`, `_RESULTS`) and says which are missing.

```bash
export AGENT_EDIT_DATA=$PWD/data
export AGENT_EDIT_DATA_RAW=/path/to/raw
export AGENT_EDIT_CACHE=/path/to/cache
export AGENT_EDIT_RUNS=/path/to/runs
export AGENT_EDIT_RESULTS=/path/to/results
```

The images are not in this repository. `data/README.md` lists where to get them, the terms of
each source, and how to build the two sets that are cut from larger downloads.

## Get PMT and build the caches

```bash
.venv-pmt/bin/python tools/fetch_pmt.py                            # PMT source and weights
.venv-pmt/bin/agent-edit cache pmt cub-10                             # features of one set (GPU)
.venv-pmt/bin/python tools/build_coco_guard_cache.py --limit 20    # COCO guard images (GPU)
```

- `tools/fetch_pmt.py` clones the PMT source at a fixed commit and downloads the COCO instance
  checkpoint (215 MB, checked against its sha256) into `$AGENT_EDIT_PMT_ROOT`, or `$AGENT_EDIT_CACHE/pmt`
  when that is unset. The DINOv3 backbone weights are gated on Hugging Face. The tool prints how to
  get access.
- `agent-edit cache pmt SET` runs the frozen backbone once over the images of a set and stores the
  features, so the editable head can run later on a CPU.
- `tools/build_coco_guard_cache.py` stores PMT's output on the first 20 COCO val2017 images. The
  COCO guard uses them to check that the original classes are kept. It reads
  `annotations/instances_val2017.json` and `val2017/` under `AGENT_EDIT_COCO_ROOT`. Both come from
  the COCO 2017 download at [cocodataset.org](https://cocodataset.org/#download).

Do these three steps once, in this order, before any command below that uses PMT.

## Run the Agent formula on one cell

A cell is one set, one number of shots per class, and one seed.

```bash
.venv-pmt/bin/agent-edit formula cub-10 --n 5 --seed 0 --out $PWD/out/formula
```

It picks its hyperparameters by cross-validation on the training images and checks that the
original COCO classes are kept. It needs the feature cache of the set and the guard cache, and no
GPU. A cell takes from under a minute to about ten minutes on one CPU thread. It writes two files
into `--out`, named like `cub-10_s0_N5`, with the endings `.json` (the pick and the scores) and
`.pt` (the new rows). On a CPU with one thread the result repeats exactly.

## Run the gradient baselines

They train on the same labels as the Agent formula and the agent.

```bash
.venv-pmt/bin/agent-edit baselines c2 cub-10 --n 5 --seed 0 --out-dir $PWD/out/baselines
```

The first argument picks the method. The tag is the prefix of the result files.

| Argument | Tag | Method |
| --- | --- | --- |
| `c2` | `C2` | linear probe on the query embeddings |
| `c5r` | `C5R` | cosine head, trained with Adam |
| `c5f` | `C5F` | full decoder fine-tuning (GPU) |
| `lwf` | `LWF` | full decoder fine-tuning with a learning-without-forgetting term. Needs `--kd-weight` and a finished `c5f` cell, whose learning rate and epochs it reuses |

Result files go into a folder per set under `--out-dir`. On a CPU, `c2` takes a few minutes and `c5r` about twenty.
A `c5f` cell is five cross-validation folds and a final fit, about 14 minutes per fold on one H100
GPU. `--fold J` runs one fold, so each fold can be its own GPU job. `c5f` starts from the linear
probe and runs `c2` first when that cell is not in `--out-dir` yet. `agent-edit baselines -h` lists all
options.

How well a rerun of the same cell repeats:

| Method | Rerun |
| --- | --- |
| `c5r` | repeats the F1 exactly on a CPU; the fitted rows differ in late digits |
| `c2` | can pick a neighbouring weight decay, which moves the score a little (L-BFGS is not exact run to run) |
| `c5f`, `lwf` | does not repeat on a GPU, even with the same seed and batch order. On a five-shot cell the score of one fit can move by a large margin, so report it over several repeats or batch orders (`--order`) |

## Serve a run for an agent

The server and the agent only share a folder. The agent's shell can be on another machine that
sees the same filesystem.

```bash
RUN=$PWD/out/runs/cub-10/my_run
.venv-pmt/bin/agent-edit serve --run-dir $RUN --budget 25 --max-exec 300 --max-images 50 \
    --backend agent_edit.backends.pmt.backend:make_backend \
    --backend-arg project=cub-10 --backend-arg seed=0 --backend-arg n_shot=5 \
    --project cub-10 --overlays --retention --recipe
```

Give `$RUN/AGENT_PROMPT.md` to a coding agent that can run shell commands. This is the whole task
description. It is built from `src/agent_edit/harness/prompts/agent.md`, the description of the
head from the backend, and the flags below. The agent then uses four commands:

| Command | What it does |
| --- | --- |
| `agent-edit exec RUN -c CODE` | run code in the kernel (`W`, `NAMES`, `save_edit` and others are predefined) |
| `agent-edit fwd RUN --json SPEC` | spend one forward pass: run the head on training images and score them, or read internals |
| `agent-edit status RUN` | passes and code runs used |
| `agent-edit final RUN --summary TEXT` | submit the saved edits and end the run |

`agent-edit prompt` writes `AGENT_PROMPT.md` without starting a server. The flags of `serve`
are off by default:

| Flag | What it adds |
| --- | --- |
| `--overlays` | every `generate` also returns pictures of the training errors |
| `--overlay-sheet` | needs `--overlays`. Also tiles the worst-scored images into one labelled `sheet.png` |
| `--retention` | `fwd '{"which": "retain"}'` scores the original COCO task. The prompt asks to keep it within 1 point |
| `--recipe` | the agent also writes `work/recipe.py`. At `final` the harness checks that it rebuilds the saved edits (max abs diff 1e-5) |
| `--hide-names` | class names become `class_<k>` everywhere the agent looks |
| `--permute-names` | class names are shuffled among the classes, with a fixed seed. Not together with `--hide-names` |

The flags a run started with are kept in `RUN/harness.json`. A restart with other flags is refused.
After `final`, score the run on the held-out images:

```bash
.venv-pmt/bin/agent-edit score $RUN        # writes $RUN/eval.json
```

`eval.json` also has the fields `ceiling` and `recall_over_ceiling`. They are null here: they need
a mask-recall file that this repository does not build.

## Replay the example run

`examples/cub-10/pmt_s0_N5_B25_opus_ovl/` is one finished run: the edit it saved (`edit/`), its
`recipe.py`, `final.json`, `eval.json`, and replays of its recipe on other sets (`replay/`).
`replay` runs `recipe.py` on another training pool with no agent, then scores the result.

```bash
.venv-pmt/bin/agent-edit replay examples/cub-10/pmt_s0_N5_B25_opus_ovl \
    --project pets-10 --seed 0 --n 5 --out $PWD/out/replay_pets-10.json
```

It needs the PMT cache of the target set (see above). Without `--project`, it replays on the
run's own pool, and the result must match the saved edit. Without `--out`, the result is written
into the run folder, under `replay/`.

## Prompts of the reported runs

`prompts/as_run/` has one prompt per fine-grained set (`stanford-cars-10`, `cub-10`, `pets-10`,
`aircraft-10`). They are word for word what the agents got, with machine paths replaced by
placeholders. The package had another name when the runs were made, so the commands inside them
differ from this tree in the package name, and in one phrase of the first line. Read
`prompts/as_run/README.md` first.

## License

Code: MIT, see `LICENSE`. The files in `data/` are derived from the source datasets, and each
source's terms apply to them. PMT, DINOv3 and the PMT weights are not part of this repository and
come with their own terms.
