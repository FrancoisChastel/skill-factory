"""Examples, honest splits and the seal."""

from __future__ import annotations

import json

import pytest

from skill_factory.classifier.dataset import (
    Example,
    examples_from_records,
    load_examples,
    parse_label,
    save_examples,
)
from skill_factory.classifier.splits import Seal, SealedError, SplitPlan


def test_parse_label_accepts_named_classes_bools_and_ints():
    assert parse_label("malicious", "malicious", "benign") is True
    assert parse_label("benign", "malicious", "benign") is False
    assert parse_label(True, "a", "b") is True
    assert parse_label(0, "a", "b") is False
    assert parse_label("Yes", "a", "b") is True
    with pytest.raises(ValueError, match="neither"):
        parse_label("Malicious", "malicious", "benign")


def test_example_hashes_state_and_defaults_group():
    ex = Example(id="a", label=True, state="hello")
    assert ex.state_hash and ex.chars == 5 and ex.group == "a"
    with pytest.raises(ValueError):
        Example(id=" ", label=True)


def test_example_status_names_why_it_was_not_judged():
    assert Example(id="a", label=True, state="x" * 10).status(100) == "judged"
    assert Example(id="a", label=True, state="x" * 10).status(5) == "over budget"
    assert Example(id="a", label=True).status(100) == "no state"
    assert Example(id="a", label=True, state="x", skipped="truncated").status(100) == "skipped: truncated"
    assert Example(id="a", label=True, state_hash="abc", chars=10).status(100) == "judged"
    # A size without text (an imported lab) is still over budget, not merely stateless.
    assert Example(id="a", label=True, chars=500).status(100) == "over budget"


def test_records_validation():
    with pytest.raises(ValueError, match="no 'id'"):
        examples_from_records([{"label": 1}])
    with pytest.raises(ValueError, match="no 'label'"):
        examples_from_records([{"id": "a"}])
    with pytest.raises(ValueError, match="duplicate"):
        examples_from_records([{"id": "a", "label": 1}, {"id": "a", "label": 0}])
    with pytest.raises(ValueError, match="empty"):
        examples_from_records([])
    ex = examples_from_records([{"id": "a", "label": 1, "input": {"k": "v"}}])[0]
    assert '"k"' in (ex.state or "")


def test_jsonl_round_trip_keeps_hash_only_examples(tmp_path):
    exs = [
        Example(id="a", label=True, state="text", group="g", source="s", metadata={"m": 1}),
        Example(id="b", label=False, state_hash="h" * 24, chars=99, split="val", skipped="too long"),
    ]
    p = save_examples(exs, tmp_path / "d.jsonl", positive="bad", negative="good")
    back = load_examples(p, positive="bad", negative="good")
    assert [e.id for e in back] == ["a", "b"]
    assert back[0].state == "text" and back[0].metadata == {"m": 1}
    assert back[1].state is None and back[1].state_hash == "h" * 24 and back[1].split == "val"


def test_load_examples_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_examples(tmp_path / "nope.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a", "label": 1}\nnot json\n')
    with pytest.raises(ValueError, match=":2:"):
        load_examples(bad)


# --- splits ------------------------------------------------------------------

def test_split_is_group_aware_and_deterministic(examples):
    plan = SplitPlan()
    a, b = plan.assign(examples), plan.assign(list(reversed(examples)))
    assert {s: sorted(e.id for e in v) for s, v in a.items()} == {s: sorted(e.id for e in v) for s, v in b.items()}
    where = {}
    for split, exs in a.items():
        for ex in exs:
            where.setdefault(ex.group, set()).add(split)
    assert all(len(s) == 1 for s in where.values())
    assert a["train"] and a["val"] and a["test"]


def test_salt_changes_the_split(examples):
    a = SplitPlan().assign(examples)["train"]
    b = SplitPlan(salt="other").assign(examples)["train"]
    assert {e.id for e in a} != {e.id for e in b}


def test_held_out_sources_and_groups(examples):
    plan = SplitPlan(held_out_sources=frozenset({"src-b"}), held_out_groups=frozenset({"g0"}))
    held = plan.assign(examples)["held-out"]
    assert {e.source for e in held if e.group != "g0"} == {"src-b"}
    assert any(e.group == "g0" for e in held)


def test_read_before_split_moves_to_train():
    exs = [Example(id="x", label=True, state="s", source="held", group="gx")]
    plan = SplitPlan(held_out_sources=frozenset({"held"}), read_before_split=frozenset({"gx"}))
    assert plan.split_of(exs[0]) == "train"


def test_explicit_split_wins_and_is_validated():
    assert SplitPlan().split_of(Example(id="a", label=True, split="test")) == "test"
    with pytest.raises(ValueError, match="unknown split"):
        SplitPlan().split_of(Example(id="a", label=True, split="dev"))


def test_a_group_straddling_splits_is_refused():
    exs = [Example(id="a", label=True, group="g", split="train"), Example(id="b", label=True, group="g", split="test")]
    with pytest.raises(ValueError, match="span several splits"):
        SplitPlan().assign(exs)


def test_split_plan_from_dict_and_validation():
    plan = SplitPlan.from_dict({"fractions": {"train": 0.6, "val": 0.2}, "held_out_sources": ["s"], "salt": "x"})
    assert plan.train == 0.6 and "s" in plan.held_out_sources
    with pytest.raises(ValueError):
        SplitPlan(train=0.8, val=0.5)


# --- seal ----------------------------------------------------------------------

def test_sealed_splits_refuse_until_frozen(tmp_path):
    seal = Seal.load(tmp_path)
    assert seal.open("h1", ["train", "val"]) == 0  # tuning splits are never sealed
    with pytest.raises(SealedError, match="freeze"):
        seal.open("h1", ["test"])
    seal.freeze("h1")
    assert Seal.load(tmp_path).open("h1", ["test", "held-out"]) == 1
    with pytest.raises(SealedError, match="not the frozen one"):
        Seal.load(tmp_path).open("h2", ["test"])


def test_refreezing_after_a_look_needs_force_and_is_reported(tmp_path):
    seal = Seal.load(tmp_path)
    seal.freeze("h1")
    seal.open("h1", ["test"])
    with pytest.raises(SealedError, match="--force"):
        seal.freeze("h2")
    seal.freeze("h2", force=True)
    reloaded = Seal.load(tmp_path)
    assert reloaded.contaminated
    assert "WARNING" in reloaded.summary()
    data = json.loads((tmp_path / "seal.json").read_text())
    assert data["frozen"]["hash"] == "h2" and len(data["views"]) == 1


def test_seal_summary_before_freezing(tmp_path):
    assert "sealed" in Seal.load(tmp_path).summary()


def test_the_baseline_is_part_of_the_seal(tmp_path):
    seal = Seal.load(tmp_path)
    seal.freeze("after", baseline="before")
    assert seal.open("after", ["test"], baseline="before") == 1
    assert seal.open("after", ["test"], baseline="after") == 2  # comparing with itself is not a second model
    with pytest.raises(SealedError, match="baseline"):
        seal.open("after", ["test"], baseline="another-candidate")
    with pytest.raises(SealedError, match="--force"):
        seal.freeze("after", baseline="another-candidate")  # swapping the baseline after a look
    seal.freeze("after", baseline="another-candidate", force=True)
    assert Seal.load(tmp_path).contaminated
