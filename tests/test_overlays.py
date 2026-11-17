"""--overlays: the harness asks the backend for error pictures and attaches their paths to the fwd answer."""

from pathlib import Path

import pytest

from agent_edit.backends.base import Backend
from agent_edit.backends.fake import FakeBackend
from conftest import fwd, result


def test_overlays_attached_to_generate(make_server):
    srv = make_server(overlays=True)
    ok, text, _ = fwd(srv, type="generate", model="original", n_images=32)
    assert ok, text
    res = result(srv, 1)
    ov = res["overlays"]
    assert ov["legend"] == FakeBackend.overlay_legend
    assert ov["files"] and all(Path(f).is_file() for f in ov["files"])
    assert Path(ov["dir"]).parent == srv.run / "work" / "fwd_1"
    assert "_pred" not in res and "_true" not in res
    assert "overlays" in text


def test_both_draws_both_heads(make_server):
    srv = make_server(overlays=True)
    ok, text, _ = fwd(srv, type="generate", model="both", n_images=16)
    assert ok, text
    res = result(srv, 1)
    assert Path(res["original"]["overlays"]["dir"]).name == "overlays_original"
    assert Path(res["edited"]["overlays"]["dir"]).name == "overlays_edited"
    assert res["paired"]["n_images"] == 16


def test_no_flag_no_overlays_and_no_private_keys(make_server):
    srv = make_server()
    assert fwd(srv, type="generate", model="edited", n_images=8)[0]
    res = result(srv, 1)
    assert "overlays" not in res and not any(k.startswith("_") for k in res)
    assert not list((srv.run / "work" / "fwd_1").glob("overlays*"))


def test_backend_without_renderer_is_refused(make_server):
    class NoPictures(FakeBackend):
        render_errors = Backend.render_errors

    with pytest.raises(SystemExit, match="render_errors"):
        make_server(NoPictures(), overlays=True)


def test_prompt_mentions_overlays_only_with_the_flag(make_server):
    on = (make_server(name="on", overlays=True).run / "AGENT_PROMPT.md").read_text()
    off = (make_server(name="off").run / "AGENT_PROMPT.md").read_text()
    assert "pictures of the errors" in on and FakeBackend.overlay_legend in on
    assert "pictures of the errors" not in off


def test_picture_folders_survive_a_cut_answer(make_server):
    srv = make_server(overlays=True, max_chars=200)
    ok, text, _ = fwd(srv, type="generate", model="both", n_images=16)
    assert ok, text
    assert "[cut;" in text
    head = text.split("\n{", 1)[0]
    for label in ("original", "edited"):
        assert str(srv.run / "work" / "fwd_1" / f"overlays_{label}") in head


def test_no_picture_note_without_the_flag(make_server):
    srv = make_server()
    ok, text, _ = fwd(srv, type="generate", model="both", n_images=16)
    assert ok and "Error pictures" not in text


# --overlay-sheet (needs --overlays): a sheet.png tiling the worst-scored images of the pass.


def test_overlay_sheet_needs_overlays(make_server):
    with pytest.raises(SystemExit, match="overlay-sheet"):
        make_server(overlay_sheet=True)


def test_overlay_sheet_needs_backend_support(make_server):
    class NoSheet(FakeBackend):
        supports_sheet = False

    with pytest.raises(SystemExit, match="sheet"):
        make_server(NoSheet(), overlays=True, overlay_sheet=True)


def test_overlay_sheet_draws_a_labelled_grid(make_server):
    srv = make_server(overlays=True, overlay_sheet=True)
    ok, text, _ = fwd(srv, type="generate", model="original", n_images=32)
    assert ok, text
    res = result(srv, 1)
    ov = res["overlays"]
    sheet = Path(ov["dir"]) / "sheet.png"
    assert sheet.is_file()
    assert f"Open {ov['dir']}/sheet.png first" in text
    assert "Error pictures" in text  # per-image PNGs are still there and still mentioned


def test_overlay_sheet_off_mentions_no_sheet(make_server):
    srv = make_server(overlays=True)
    ok, text, _ = fwd(srv, type="generate", model="original", n_images=8)
    assert ok and "sheet.png" not in text


def test_overlay_sheet_prompt_sentence_only_with_the_flag(make_server):
    on = (make_server(name="on", overlays=True, overlay_sheet=True).run / "AGENT_PROMPT.md").read_text()
    off = (make_server(name="off", overlays=True).run / "AGENT_PROMPT.md").read_text()
    assert "sheet.png" in on and "sheet.png" not in off
