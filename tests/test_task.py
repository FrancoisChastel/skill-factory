"""Tests for Task and Dataset."""

from __future__ import annotations

import pytest

from skill_factory.core.task import Dataset, Task


def _ds(n: int) -> Dataset:
    return Dataset([Task(id=f"t{i}", input=f"in{i}") for i in range(n)])


def test_split_is_deterministic():
    ds = _ds(10)
    train_a, val_a = ds.split(0.3, seed=0)
    train_b, val_b = ds.split(0.3, seed=0)
    assert [t.id for t in val_a] == [t.id for t in val_b]
    assert [t.id for t in train_a] == [t.id for t in train_b]


def test_split_sizes_and_disjoint():
    ds = _ds(10)
    train, val = ds.split(0.3, seed=1)
    assert len(val) == 3
    assert len(train) == 7
    assert set(t.id for t in train).isdisjoint(t.id for t in val)


def test_split_always_keeps_one_each():
    ds = _ds(2)
    train, val = ds.split(0.9, seed=0)
    assert len(train) >= 1 and len(val) >= 1


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        Dataset([Task(id="dup", input="a"), Task(id="dup", input="b")])


def test_empty_dataset_rejected():
    with pytest.raises(ValueError):
        Dataset([])


def test_from_records_requires_input():
    with pytest.raises(ValueError, match="input"):
        Dataset.from_records([{"id": "x"}])


def test_from_records_coerces_dict_input_to_json():
    ds = Dataset.from_records([{"id": "x", "input": {"a": 1}, "expected": {"b": 2}}])
    task = ds[0]
    assert '"a": 1' in task.input
    assert task.expected is not None and '"b": 2' in task.expected


def test_from_jsonl(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"id":"1","input":"hi","expected":"yo"}\n\n{"id":"2","input":"ho"}\n')
    ds = Dataset.from_jsonl(p)
    assert len(ds) == 2
    assert ds[0].expected == "yo"
    assert ds[1].expected is None


def test_from_jsonl_bad_line(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id":"1","input":"hi"}\n{not json}\n')
    with pytest.raises(ValueError, match="invalid JSON"):
        Dataset.from_jsonl(p)
