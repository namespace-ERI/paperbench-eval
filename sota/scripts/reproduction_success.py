from __future__ import annotations

from pathlib import Path

from paperbench.nano.reproduction_success import (
    EXPLICIT_FAILURE_MARKERS,
    SUCCESS_MARKERS,
    TERMINAL_FAILURE_MARKERS,
    normalize_repro_log_text,
    reproduction_log_has_terminal_failure,
    reproduction_payload_succeeded,
)


def reproduction_metadata_path_succeeded(path: Path) -> bool:
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return reproduction_payload_succeeded(payload)
