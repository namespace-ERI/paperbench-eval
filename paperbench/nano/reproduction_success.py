from __future__ import annotations

from typing import Any


EXPLICIT_FAILURE_MARKERS = (
    "full gpu reproduction failed with status",
    "full reproduction failed with status",
    "running explicit sanity fallback",
    "explicit sanity fallback",
    "full_run_failed_fallback_used",
)

TERMINAL_FAILURE_MARKERS = (
    "traceback (most recent call last):",
    "runtimeerror:",
    "attributeerror:",
    "modulenotfounderror:",
    "datasetnotfounderror:",
    "filenotfounderror:",
    "calledprocesserror:",
    "assertionerror:",
    "permissionerror:",
    "oserror:",
    "cuda error:",
    "device-side assert triggered",
    "core dumped",
    "aborted                 (core dumped)",
    "assertion `t >= 0 && t < n_classes` failed",
)

SUCCESS_MARKERS = (
    "reproduction complete.",
    "reproduction complete",
    "foa reproduction complete.",
    "foa reproduction complete",
    "finished. outputs are in",
    "outputs written to",
    "results are in",
    "results written to",
)


def normalize_repro_log_text(repro_log: str) -> str:
    return (repro_log or "").replace("\r", "\n").lower()


def reproduction_log_has_terminal_failure(repro_log: str) -> bool:
    text = normalize_repro_log_text(repro_log)
    if not text:
        return False

    if any(marker in text for marker in EXPLICIT_FAILURE_MARKERS):
        return True

    last_success = max((text.rfind(marker) for marker in SUCCESS_MARKERS), default=-1)
    last_failure = max((text.rfind(marker) for marker in TERMINAL_FAILURE_MARKERS), default=-1)
    if last_failure < 0:
        return False
    if last_success >= 0:
        return last_failure > last_success
    return last_failure >= max(0, len(text) - 20000)


def reproduction_payload_succeeded(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False

    repro_script_exists = payload.get("repro_script_exists")
    if repro_script_exists is None:
        repro_script_exists = payload.get("reproduce_script_exists")
    if repro_script_exists is None:
        repro_script_exists = payload.get("reproduce_sh_exists")
    if repro_script_exists is None and payload.get("executed_submission"):
        # Some official metadata payloads do not preserve an explicit script-exists
        # boolean even when reproduce.sh ran successfully and produced an executed
        # submission. Treat that shape as script-present unless a later failure
        # signal contradicts it.
        repro_script_exists = True
    if not bool(repro_script_exists):
        return False

    if bool(payload.get("timedout")):
        return False

    if not payload.get("executed_submission"):
        return False

    exit_code = payload.get("repro_exit_code")
    if exit_code is not None and exit_code != 0:
        return False

    repro_log = str(payload.get("repro_log") or "")
    if "reproduce.sh not found" in repro_log.lower():
        return False

    if reproduction_log_has_terminal_failure(repro_log):
        return False

    return True
