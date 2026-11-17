"""--hide-names: class_<k> replaces class names in the prompt, describe(), fwd answers and overlay labels."""

import json

import pytest

from agent_edit.backends.fake import FakeBackend
from agent_edit.harness.names import Names
from conftest import fwd, result

CLASSES = ["Black-footed Albatross", "California Gull", "Abyssinian"]


class Described(FakeBackend):
    def describe(self):
        return super().describe() + " Rows: " + "; ".join(f"row {k} = {c}" for k, c in enumerate(self.classes))

    def project_info(self):
        return {"classes": list(self.classes), "task": "Photos of birds and cats, e.g. a california gull."}


def test_names_text():
    nm = Names(CLASSES)
    assert nm.text("California_Gull_0100_41088.jpg") == "class_1_0100_41088.jpg"
    assert nm.text("a black footed albatross and an Abyssinian") == "a class_0 and an class_2"
    assert nm.text("Abyssinians and XAbyssinian") == "Abyssinians and XAbyssinian"
    out = nm.out({"California Gull": {"f1": 1.0}, "keys": ["Abyssinian_184"], "n": 3})
    assert out == {"class_1": {"f1": 1.0}, "keys": ["class_2_184"], "n": 3}
    assert nm.inp({"prompts": ["class_1", 2], "what": "queries"}) == {"prompts": ["California Gull", 2], "what": "queries"}


def test_names_hidden_everywhere_the_agent_looks(make_server):
    srv = make_server(Described(classes=CLASSES), overlays=True, hide_names=True)
    prompt = (srv.run / "AGENT_PROMPT.md").read_text()
    assert "Its classes: class_0, class_1, class_2." in prompt and "row 1 = class_1" in prompt
    assert "a class_1." in prompt and "class names are hidden" in prompt

    ok, text, _ = fwd(srv, type="generate", model="original", n_images=64)
    assert ok
    ok, text2, _ = fwd(srv, type="read", model="original", n_images=8)
    assert ok
    seen = prompt + text + text2 + json.dumps(result(srv, 1)) + json.dumps(result(srv, 2))
    for c in CLASSES:
        for word in c.replace("-", " ").split():
            assert word.lower() not in seen.lower(), word
    assert set(result(srv, 1)["per_class_f1"]) == {"class_0", "class_1", "class_2"}
    assert [c["class_name"] for c in result(srv, 2)["classes"]] == ["class_0", "class_1", "class_2"]
    assert srv.backend.drawn_labels and all(lab.startswith("class_") for lab in srv.backend.drawn_labels)
    assert json.loads((srv.run / "harness.json").read_text())["names"]["California Gull"] == "class_1"


def test_names_shown_without_the_flag(make_server):
    srv = make_server(Described(classes=CLASSES), overlays=True)
    assert "California Gull" in (srv.run / "AGENT_PROMPT.md").read_text()
    fwd(srv, type="generate", model="original", n_images=64)
    assert "California Gull" in result(srv, 1)["per_class_f1"]
    assert any("California Gull" in lab for lab in srv.backend.drawn_labels)


def test_hiding_needs_a_class_list(make_server):
    class NoClasses(FakeBackend):
        def project_info(self):
            return {}

    with pytest.raises(ValueError, match="class list"):
        make_server(NoClasses(), hide_names=True)


# --- permuted names (--permute-names) -----------------------------------------------------


def test_derangement_has_no_fixed_point():
    from agent_edit.harness.names import derangement

    for seed in range(20):
        order = derangement(10, seed)
        assert sorted(order) == list(range(10))
        assert all(order[i] != i for i in range(10))


def test_derangement_trivial_sizes():
    from agent_edit.harness.names import derangement

    assert derangement(0, 0) == []
    assert derangement(1, 0) == [0]  # nothing to derange


def test_names_permuted_is_a_bijection_over_real_names():
    nm = Names(CLASSES, permute=True, seed=0)
    assert set(nm.shown.values()) == set(CLASSES)  # every shown name is still a real class name
    assert all(nm.shown[c] != c for c in CLASSES)  # no class keeps its own name
    for c in CLASSES:
        assert nm.real[nm.shown[c]] == c
    assert nm.inp(nm.out({"prompts": CLASSES})) == {"prompts": CLASSES}


def test_names_permuted_needs_a_seed():
    with pytest.raises(ValueError, match="seed"):
        Names(CLASSES, permute=True)


def test_names_permuted_rewrites_image_keys_to_another_real_name():
    nm = Names(CLASSES, permute=True, seed=1)
    shown = nm.shown["California Gull"]
    assert nm.text("California_Gull_0100_41088.jpg") == f"{shown}_0100_41088.jpg"


def test_names_permuted_prompt_has_no_hint(make_server):
    """The point of permuting is to say nothing: no 'names are hidden' note, and only real class
    names appear (just not next to their own class)."""
    srv = make_server(Described(classes=CLASSES), permute_names=True, permute_seed=0)
    prompt = (srv.run / "AGENT_PROMPT.md").read_text()
    assert "class names are hidden" not in prompt.lower()
    body = prompt.replace(str(srv.run), "<run>")  # the run's own tmp_path can contain "permut"/"derange"
    assert "permut" not in body.lower() and "derange" not in body.lower()
    for c in CLASSES:
        assert c in prompt  # every real name still shows up, just reassigned


def test_names_hide_and_permute_are_mutually_exclusive(make_server):
    with pytest.raises(SystemExit, match="mutually exclusive"):
        make_server(Described(classes=CLASSES), hide_names=True, permute_names=True, permute_seed=0)


def test_permute_names_needs_a_seed_at_serve_time(make_server):
    with pytest.raises(SystemExit, match="seed"):
        make_server(Described(classes=CLASSES), permute_names=True)


def test_names_shown_dict_override_skips_the_derangement():
    """replay reuses the run's own stored mapping instead of recomputing it."""
    shown = {"Black-footed Albatross": "Abyssinian", "California Gull": "Black-footed Albatross",
             "Abyssinian": "California Gull"}
    nm = Names(CLASSES, shown=shown)
    assert nm.shown == shown
    assert nm.real["Abyssinian"] == "Black-footed Albatross"


# --- secret words (invented classes over a parent set's own image keys) ------------------


def test_secret_words_get_a_one_way_token():
    nm = Names(["group_1", "group_2"], secret=["California Gull", "Abyssinian"])
    assert nm.text("California_Gull_0100_41088.jpg") == "secret_1_0100_41088.jpg"
    assert nm.text("an Abyssinian cat") == "an secret_0 cat"
    # a secret token has no class, so it isn't echoed back by inp()
    assert nm.inp({"a": "secret_0"}) == {"a": "secret_0"}


def test_secret_words_do_not_collide_with_class_names():
    nm = Names(["group_1", "group_2"], secret=["group_1", "California Gull"])
    assert nm.text("California Gull") == "secret_0"
    assert nm.text("group_1") == "class_0"  # a project's own class wins over a same-spelled secret


def test_secret_words_also_apply_in_permuted_mode():
    nm = Names(["group_1", "group_2"], secret=["California Gull"], permute=True, seed=0)
    assert nm.text("California_Gull_0100_41088.jpg").startswith("secret_0_0100")
