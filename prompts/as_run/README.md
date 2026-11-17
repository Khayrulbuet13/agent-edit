# As-run prompts

These are the prompts of four real runs, one per fine-grained set, as the agent saw them. They are
here to read, not to run. Each run is `pmt_s0_N5_B25_opus_ovl`: seed 0, 5 shots per class, a budget of
25 forward passes, the overlay on, agent `opus`.

```
<set>/AGENT_PROMPT.md   the task text of the run
<set>/prompt.md         the full text the agent got: a short wrapper, then AGENT_PROMPT.md
```

Sets: `stanford-cars-10`, `cub-10`, `pets-10`, `aircraft-10`.

## What was changed

Only absolute paths. Each one is replaced by a placeholder. Nothing else is touched, not the first
line and not the commands.

| placeholder | replaces | per file |
| --- | --- | --- |
| `<WORK_DIR>` | the `work/` folder of the run | 4 in `AGENT_PROMPT.md`, 7 in `prompt.md` |
| `<RUN_DIR>` | the run folder (`runs/<set>/<run>`) | 7 in `AGENT_PROMPT.md`, 10 in `prompt.md` |
| `<PYTHON>` | the Python of the PMT environment | 7 in `AGENT_PROMPT.md` |
| `<WSP_CLI>` | the `wsp` command of the PMT environment | 8 in `prompt.md` |
| `<TREE_DIR>` | the folder with the code of the run (a checkout at the run's commit) | 1 in each file |

The counts are the same for the four sets. `<WORK_DIR>` is applied before `<RUN_DIR>`, so a path
inside `work/` becomes `<WORK_DIR>/...`.

## Package name

The package was named `wsp` when the runs were made. The text still says `wsp`, for example
`<PYTHON> -m wsp.harness exec <RUN_DIR> ...` and `<WSP_CLI> status <RUN_DIR>`. In this repo the
package is `agent_edit`, so use `agent_edit.harness` where a prompt says `wsp.harness`.
