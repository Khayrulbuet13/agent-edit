import subprocess
import sys

import pytest

from agent_edit import config
from agent_edit.cli import active, main


def test_help_lists_every_registered_command(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for cmd in ("serve", "exec", "fwd", "status", "final", "prompt", "score", "baselines", "cache"):
        assert cmd in out
    assert all(c.name in out for c in active())


def test_client_commands_do_not_import_torch(tmp_path):
    code = ("import sys; from agent_edit.cli import main; "
            f"main(['status', '{tmp_path}']); assert 'torch' not in sys.modules, 'torch imported'")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "server never started" in r.stdout


def test_missing_env_var_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("AGENT_EDIT_RUNS", raising=False)
    with pytest.raises(config.ConfigError, match="AGENT_EDIT_RUNS is not set"):
        config.runs_dir()


def test_prompt_command_with_fake_backend(tmp_path, capsys):
    assert main(["prompt", "--project", "toy", "--budget", "5", "--backend", "agent_edit.backends.fake:FakeBackend",
                 "--run-dir", str(tmp_path)]) == 0
    text = (tmp_path / "AGENT_PROMPT.md").read_text()
    assert 'The project is "toy". Its classes: a, b, c.' in text
    assert "budget of 5 passes" in text


@pytest.mark.parametrize("args", [["exec", "RUN", "--file", "OUT"], ["fwd", "RUN", "--json-file", "OUT"],
                                  ["final", "RUN", "--summary-file", "OUT"]])
def test_client_sends_only_files_inside_the_run(tmp_path, capsys, args):
    run, out = tmp_path / "run", tmp_path / "secret.txt"
    (run / "queue").mkdir(parents=True)
    out.write_text("secret")
    assert main([str(run) if a == "RUN" else str(out) if a == "OUT" else a for a in args]) == 2
    assert "not inside the run folder" in capsys.readouterr().out
