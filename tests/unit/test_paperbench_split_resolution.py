from __future__ import annotations

from pathlib import Path

import pytest

from paperbench.nano.eval import _load_paper_ids_for_split


def test_load_paper_ids_for_split_reads_explicit_file_path(tmp_path: Path) -> None:
    split_path = tmp_path / "custom_split.txt"
    split_path.write_text("paper-a\npaper-b\n", encoding="utf-8")

    resolved_path, paper_ids = _load_paper_ids_for_split(str(split_path))

    assert resolved_path == split_path
    assert paper_ids == ["paper-a", "paper-b"]


def test_load_paper_ids_for_split_falls_back_to_single_paper_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "paperbench.nano.eval.paper_registry.get_paper",
        lambda paper_id: object(),
    )

    resolved_path, paper_ids = _load_paper_ids_for_split("toy-paper")

    assert resolved_path is None
    assert paper_ids == ["toy-paper"]
