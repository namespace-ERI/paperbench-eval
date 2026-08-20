#!/usr/bin/env python3
"""Schedule official PaperBench reruns using already-extracted SOTA skills.

This scheduler intentionally starts from the official PaperBench rollout stage.
It does not relaunch Paper2Skills distillation. Case order is read directly from
`sota/docs/case_result_status.md`.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sota.scripts.reproduction_success import reproduction_payload_succeeded

ROOT = REPO_ROOT
STATUS_DOC = ROOT / "sota" / "docs" / "case_result_status.md"
STATE_ROOT = ROOT / "sota" / "state" / "official_case_scheduler"
CASES_ROOT = ROOT / "sota" / "cases"
SCHEDULER_TICK_LOCK = STATE_ROOT / "scheduler_tick.lock"
DEFERRED_CASES_PATH = STATE_ROOT / "deferred_cases.json"
DEFAULT_MAX_ACTIVE_CASES = 1
DEFAULT_INTERVAL_SECONDS = 300
DEFAULT_LAUNCHES_PER_TICK = 1
DEFAULT_GPU_IDS = [str(i) for i in range(4, 8)]
ALLOWED_GPU_IDS = set(DEFAULT_GPU_IDS)
DEFAULT_MAX_CASES_PER_GPU = 1
CASE_LINE_RE = re.compile(r"^- `(case\d+)` \| `([^`]+)`(?: \| .*|)$")
DEFAULT_HEALTH_STALL_SECONDS = 3600
DEFAULT_MIN_RUN_AGE_BEFORE_STALL_SECONDS = 1800
BACK_HALF_LOG_STALE_SECONDS = 1800
RESPONSES_CALL_DANGLING_SECONDS = 900
BACK_HALF_ARTIFACT_STALE_SECONDS = 5400
BACK_HALF_ACTIVE_PHASES = {
    "launching",
    "running",
    "retrying_after_transient_network_failure",
}

TRANSIENT_NETWORK_MARKERS = (
    "requests.exceptions.ProxyError",
    "urllib3.exceptions.ProxyError",
    "RemoteDisconnected",
    "Remote end closed connection without response",
    "ReadTimeoutError",
    "SSLEOFError",
    "SSLError",
    "gnutls_handshake() failed",
    "The TLS connection was non-properly terminated",
    "fatal: unable to access",
    "UNEXPECTED_EOF_WHILE_READING",
    "IncompleteRead",
    "ProtocolError",
    "Connection broken:",
    "Connection refused",
    "Connection reset by peer",
    "Temporary failure in name resolution",
    "Max retries exceeded with url",
    "HTTPSConnectionPool(host='huggingface.co'",
    "HTTPSConnectionPool(host=\"huggingface.co\"",
    "HTTPSConnectionPool(host='download.pytorch.org'",
    "HTTPSConnectionPool(host='download-r2.pytorch.org'",
    "HTTPSConnectionPool(host='pypi.org'",
    "HTTPSConnectionPool(host='files.pythonhosted.org'",
    "HTTP Error 502",
    "502 Bad Gateway",
    "Retryable HTTP 502",
    "HTTP Error 520",
    "Retryable HTTP 520",
    "520",
    "Read timed out",
    "Could not install packages due to an OSError",
    "cas-bridge.xethub.hf.co",
    "xet-bridge-us",
    "Transient judge computer failure while reading selected submission files",
    "No such container",
    "404 Client Error for http+docker",
    "container is not running",
    "docker.errors.NotFound",
)

RESPONSES_STATUS_RE = re.compile(
    r"(?:Responses|Chat completions) call returned.*status_code=(\d+)"
)
RESPONSES_STARTED_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[info\s+\] "
    r"(?:Responses|Chat completions) call started"
)
RESPONSES_RETURNED_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[info\s+\] "
    r"(?:Responses|Chat completions) call returned"
)
LEGACY_BACK_HALF_MIN_RETRY_ATTEMPTS = 100


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_deferred_cases() -> set[str]:
    payload = load_json(DEFERRED_CASES_PATH)
    if isinstance(payload, list):
        return {str(item).strip() for item in payload if str(item).strip()}
    if isinstance(payload, dict):
        values = payload.get("cases")
        if isinstance(values, list):
            return {str(item).strip() for item in values if str(item).strip()}
    return set()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def event(name: str, details: dict[str, Any]) -> None:
    append_jsonl(
        STATE_ROOT / "events.jsonl",
        {"timestamp_utc": utc_now(), "event": name, "details": details},
    )


class SchedulerTickLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"{os.getpid()}\n")
        self.handle.flush()

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is None:
            return
        try:
            self.handle.seek(0)
            self.handle.truncate()
            self.handle.flush()
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except Exception:
        return False
    if value <= 0:
        return False
    proc_path = Path(f"/proc/{value}")
    if not proc_path.exists():
        return False
    try:
        status_text = (proc_path / "status").read_text(encoding="utf-8", errors="replace")
    except Exception:
        return True
    for line in status_text.splitlines():
        if line.startswith("State:"):
            return "\tZ" not in line and "(zombie)" not in line.lower()
    return True


def process_group_id(pid: Any) -> int | None:
    try:
        value = int(pid)
    except Exception:
        return None
    if value <= 0 or not pid_alive(value):
        return None
    try:
        pgid = os.getpgid(value)
    except Exception:
        return None
    return pgid if pgid > 0 else None


def parse_utc_to_epoch(text: str) -> float | None:
    value = text.strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc).timestamp()
        except Exception:
            pass
    return None


def parse_queue() -> list[dict[str, str]]:
    if not STATUS_DOC.exists():
        raise RuntimeError(f"missing status doc: {STATUS_DOC}")

    queue: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in STATUS_DOC.read_text(encoding="utf-8").splitlines():
        match = CASE_LINE_RE.match(line.strip())
        if not match:
            continue
        case_id, paper_id = match.groups()
        if case_id in seen:
            raise RuntimeError(f"duplicate case in status doc: {case_id}")
        queue.append({"case_id": case_id, "paper_id": paper_id})
        seen.add(case_id)

    case_dirs = sorted(
        [path for path in CASES_ROOT.glob("case*") if path.is_dir()],
        key=lambda path: int(path.name.removeprefix("case")) if path.name.removeprefix("case").isdigit() else math.inf,
    )
    expected_cases = [path.name for path in case_dirs]
    if len(queue) != len(expected_cases):
        for case_id in expected_cases:
            if case_id in seen:
                continue
            meta = metadata(case_id)
            paper_id = str(meta.get("paper_id") or "").strip()
            if not paper_id:
                raise RuntimeError(
                    f"status doc missing {case_id}, and fallback metadata has no paper_id in "
                    f"{case_root(case_id) / 'CASE_METADATA.json'}"
                )
            queue.append({"case_id": case_id, "paper_id": paper_id})
            seen.add(case_id)
    if len(queue) != len(expected_cases):
        raise RuntimeError(
            f"expected {len(expected_cases)} ordered cases after fallback merge for {STATUS_DOC}, "
            f"found {len(queue)}"
        )
    return queue


def launch_priority(kind: str | None) -> int:
    value = str(kind or "").strip()
    if value == "grading_only":
        return 0
    if value == "reproduction_grading":
        return 1
    return 2


def case_root(case_id: str) -> Path:
    return CASES_ROOT / case_id


def metadata(case_id: str) -> dict[str, Any]:
    payload = load_json(case_root(case_id) / "CASE_METADATA.json")
    return payload if isinstance(payload, dict) else {}


def prompt_mode(meta: dict[str, Any]) -> str:
    value = str(meta.get("official_prompt_mode") or "").strip()
    return value or "official_skill_recovery_strict"


def skills_dir(meta: dict[str, Any], case_id: str) -> str:
    value = str(meta.get("official_skills_dir") or "").strip()
    if value:
        return value
    paper_id = str(meta.get("paper_id") or "").strip() or case_id
    return f"sota/skills/{paper_id}/skill"


def launch_record_path(case_id: str) -> Path:
    return STATE_ROOT / "launches" / f"{case_id}.json"


def recovery_state_path(case_id: str) -> Path:
    return case_root(case_id) / "monitoring" / "official_recovery_state.json"


def continuation_state_path(case_id: str) -> Path:
    return case_root(case_id) / "monitoring" / "official_continuation_state.json"


def back_half_rerun_state_path(case_id: str) -> Path:
    return case_root(case_id) / "monitoring" / "back_half_rerun_state.json"


def load_launch_record(case_id: str) -> dict[str, Any]:
    payload = load_json(launch_record_path(case_id))
    return payload if isinstance(payload, dict) else {}


def launch_record_is_back_half_rerun(case_id: str) -> bool:
    launch_record = load_launch_record(case_id)
    if not isinstance(launch_record, dict):
        return False
    mode = str(launch_record.get("mode") or "").strip()
    return mode in {"reproduction_grading_rerun", "grading_only_rerun"}


def supervisor_pids_should_count_as_live_rollout(case_id: str, *, now: float | None = None) -> bool:
    if not launch_record_is_back_half_rerun(case_id):
        return True
    active_state, _ = active_back_half_state_details(case_id, now=now)
    return active_state is not None


def load_recovery_state(case_id: str) -> dict[str, Any]:
    payload = load_json(recovery_state_path(case_id))
    return payload if isinstance(payload, dict) else {}


def load_continuation_state(case_id: str) -> dict[str, Any]:
    payload = load_json(continuation_state_path(case_id))
    return payload if isinstance(payload, dict) else {}


def refresh_continuation_state_checkpoint(case_id: str, *, dry_run: bool) -> dict[str, Any] | None:
    payload = load_continuation_state(case_id)
    if not isinstance(payload, dict) or not payload:
        return None

    current_tar = continuation_submission_tar(case_id)
    if current_tar is not None and current_tar.exists():
        return None

    best_tar = best_continuation_submission_tar(case_id)
    if best_tar is None or not best_tar.exists():
        return None

    best_run_root = run_root_for_submission_tar(best_tar)
    update = {
        "case_id": case_id,
        "previous_submission_checkpoint": str(best_tar.relative_to(ROOT)),
        "current_submission_checkpoint": "",
        "previous_score": -1,
        "new_score": -1,
        "updated_at_utc": utc_now(),
    }
    if best_run_root is not None:
        update["run_root"] = str(best_run_root.relative_to(ROOT))

    if dry_run:
        return update

    payload["previous_submission_checkpoint"] = update["previous_submission_checkpoint"]
    if "run_root" in update:
        payload["run_root"] = update["run_root"]
    payload["last_updated_utc"] = update["updated_at_utc"]
    write_json(continuation_state_path(case_id), payload)
    event("scheduler_refreshed_continuation_checkpoint", update)
    return update


def load_back_half_rerun_state(case_id: str) -> dict[str, Any]:
    payload = load_json(back_half_rerun_state_path(case_id))
    return payload if isinstance(payload, dict) else {}


def active_back_half_state_details(
    case_id: str,
    *,
    now: float | None = None,
) -> tuple[dict[str, Any] | None, str]:
    state = load_back_half_rerun_state(case_id)
    if not isinstance(state, dict) or not state:
        return None, "missing_state"

    phase = str(state.get("phase") or "").strip()
    if phase not in BACK_HALF_ACTIVE_PHASES:
        return None, f"phase_not_active:{phase or 'missing'}"

    pid = state.get("pid")
    if not pid_alive(pid):
        return None, "pid_not_alive"

    preferred_root = preferred_run_root(case_id)
    state_root = normalize_candidate_run_root(case_id, str(state.get("run_root") or ""))
    # A live back-half worker may legitimately continue from an older run root when the
    # continuation checkpoint comes from an earlier official run. Treat that as active
    # work and let later reconciliation logic decide whether the lineage should be
    # refreshed; do not drop the worker from the active set just because supervisor state
    # now points at a newer run root.

    now = now if now is not None else time.time()
    heartbeat_epoch = parse_utc_to_epoch(str(state.get("last_updated_utc") or ""))
    latest_log = latest_back_half_launch_log(
        case_id,
        str(state.get("kind") or "").strip() or None,
    )
    latest_log_mtime = safe_stat_mtime(latest_log) if latest_log is not None else None
    freshest_epoch = max(
        [value for value in (heartbeat_epoch, latest_log_mtime) if value is not None],
        default=None,
    )
    if freshest_epoch is not None and (now - freshest_epoch) >= BACK_HALF_LOG_STALE_SECONDS:
        run_root = preferred_root or state_root or latest_run_root(case_id)
        task_run_dir = latest_task_run_dir_for_run_root(run_root)
        run_log_mtime = safe_stat_mtime(task_run_dir / "run.log") if task_run_dir is not None else None
        group_log_mtime = safe_stat_mtime(task_run_dir / "group.log") if task_run_dir is not None else None
        freshest_runtime_epoch = max(
            [value for value in (run_log_mtime, group_log_mtime) if value is not None],
            default=None,
        )
        if freshest_runtime_epoch is None or (now - freshest_runtime_epoch) >= BACK_HALF_LOG_STALE_SECONDS:
            return None, "heartbeat_and_logs_stale"

    return state, ""


def supervisor_status_path(case_id: str) -> Path:
    return case_root(case_id) / "monitoring" / "supervisor_status.json"


def process_events_path(case_id: str) -> Path:
    return case_root(case_id) / "monitoring" / "process_events.jsonl"


def latest_run_root(case_id: str) -> Path | None:
    runs_root = case_root(case_id) / "official_runs"
    candidates = sorted(
        [path for path in runs_root.glob("*") if path.is_dir()],
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    return candidates[-1] if candidates else None


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def normalize_candidate_run_root(case_id: str, raw: str) -> Path | None:
    value = str(raw or "").strip()
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    runs_root = case_root(case_id) / "official_runs"
    if not path.exists() or not path_is_within(path, runs_root):
        return None
    return path


def supervisor_preferred_run_root(case_id: str) -> Path | None:
    status = load_json(supervisor_status_path(case_id))
    if not isinstance(status, dict):
        return None

    for key in ("official_current_run_root", "official_run_group_path"):
        candidate = normalize_candidate_run_root(case_id, str(status.get(key) or ""))
        if candidate is not None:
            return candidate

    task_runs_path = normalize_candidate_run_root(case_id, str(status.get("official_task_runs_path") or ""))
    if task_runs_path is not None:
        try:
            return task_runs_path.parent
        except Exception:
            return None
    return None


def preferred_run_root(case_id: str) -> Path | None:
    return supervisor_preferred_run_root(case_id) or latest_run_root(case_id)


def parse_run_root_started_at(run_root: Path | None) -> float | None:
    if run_root is None:
        return None
    token = run_root.name.split("_", 1)[0]
    try:
        return datetime.strptime(token, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return None


def latest_task_run_dir_for_run_root(run_root: Path | None) -> Path | None:
    if run_root is None:
        return None
    run_groups_root = run_root / "runs"
    if not run_groups_root.exists():
        return None
    run_groups = sorted(
        [path for path in run_groups_root.glob("*") if path.is_dir()],
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    if not run_groups:
        return None
    task_runs = sorted(
        [path for path in run_groups[-1].iterdir() if path.is_dir()],
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    return task_runs[-1] if task_runs else None


def latest_task_run_dir(case_id: str) -> Path | None:
    return latest_task_run_dir_for_run_root(preferred_run_root(case_id))


def continuation_submission_tar(case_id: str) -> Path | None:
    continuation = load_continuation_state(case_id)
    raw = str(continuation.get("previous_submission_checkpoint") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_dir():
        candidate = path / "submission.tar.gz"
        return candidate if candidate.exists() else None
    if path.name == "submission.tar.gz" and path.exists():
        return path
    return None


def latest_submission_tar_under_task_run(task_run_dir: Path | None) -> Path | None:
    if task_run_dir is None:
        return None
    candidates = sorted(
        task_run_dir.glob("submissions/*/submission.tar.gz"),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    return candidates[-1] if candidates else None


def latest_submission_tar_under_run_root(run_root: Path | None) -> Path | None:
    return latest_submission_tar_under_task_run(latest_task_run_dir_for_run_root(run_root))


def continuation_submission_tar_for_scheduler(case_id: str) -> Path | None:
    recorded = continuation_submission_tar(case_id)
    preferred_root = preferred_run_root(case_id)
    preferred_tar = latest_submission_tar_under_run_root(preferred_root)

    if recorded is not None and recorded.exists():
        recorded_root = run_root_for_submission_tar(recorded)
        if preferred_root is not None and recorded_root is not None and preferred_root != recorded_root:
            if preferred_tar is not None and preferred_tar.exists():
                return preferred_tar
        return recorded

    if preferred_tar is not None and preferred_tar.exists():
        return preferred_tar

    runs_root = case_root(case_id) / "official_runs"
    if not runs_root.exists():
        return None

    candidates = sorted(
        runs_root.glob("**/submission.tar.gz"),
        key=lambda candidate: (candidate.stat().st_mtime, str(candidate)),
    )
    return candidates[-1] if candidates else None


def normalize_archive_member_name(name: str) -> str:
    normalized = name.lstrip("/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


_SUBMISSION_PAYLOAD_COUNT_CACHE: dict[tuple[str, int, int], int] = {}


def submission_payload_file_count(tar_path: Path) -> int:
    import tarfile

    try:
        stat = tar_path.stat()
    except Exception:
        return -1

    cache_key = (str(tar_path), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _SUBMISSION_PAYLOAD_COUNT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        with tarfile.open(tar_path, "r:gz") as archive:
            count = 0
            for member in archive:
                normalized = normalize_archive_member_name(member.name)
                if not normalized.startswith("submission/"):
                    continue
                rel = normalized[len("submission/") :]
                if not rel or rel == ".git" or rel.startswith(".git/"):
                    continue
                if member.isfile() or member.islnk() or member.issym():
                    count += 1
            _SUBMISSION_PAYLOAD_COUNT_CACHE[cache_key] = count
            return count
    except Exception:
        return -1


def run_root_for_submission_tar(tar_path: Path) -> Path | None:
    for parent in tar_path.parents:
        try:
            if parent.parent.name == "official_runs":
                return parent
        except Exception:
            continue
    return None


def best_continuation_submission_tar(case_id: str) -> Path | None:
    current = continuation_submission_tar(case_id)
    if current is not None and current.exists():
        return current

    runs_root = case_root(case_id) / "official_runs"
    if not runs_root.exists():
        return None

    candidates = sorted(
        runs_root.glob("**/submission.tar.gz"),
        key=lambda candidate: (candidate.stat().st_mtime, str(candidate)),
    )
    return candidates[-1] if candidates else None


def continuation_submission_dir(case_id: str) -> Path | None:
    submission_tar = continuation_submission_tar_for_scheduler(case_id)
    return submission_tar.parent if submission_tar is not None else None


def task_run_dir_for_submission_dir(submission_dir: Path) -> Path | None:
    try:
        if submission_dir.parent.name != "submissions":
            return None
        return submission_dir.parent.parent
    except Exception:
        return None


def run_root_for_task_run_dir(task_run_dir: Path) -> Path | None:
    try:
        return task_run_dir.parents[2]
    except Exception:
        return None


def safe_stat_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except Exception:
        return None


def latest_mtime(paths: list[Path]) -> float | None:
    mtimes = [value for value in (safe_stat_mtime(path) for path in paths) if value is not None]
    return max(mtimes) if mtimes else None


def file_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def tail_text(path: Path, max_bytes: int = 262144) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def looks_like_transient_network_failure(*texts: str) -> bool:
    haystack = "\n".join(texts)
    return any(marker in haystack for marker in TRANSIENT_NETWORK_MARKERS)


def reproduction_metadata_succeeded(payload: Any) -> bool:
    return reproduction_payload_succeeded(payload)


def judge_payload_succeeded(payload: Any) -> bool:
    return bool(payload) and (
        bool(payload.get("success"))
        or (
            payload.get("num_leaf_nodes") is not None
            and payload.get("num_invalid_leaf_nodes") is not None
            and int(payload.get("num_invalid_leaf_nodes")) < int(payload.get("num_leaf_nodes"))
        )
    )


def back_half_log_has_trailing_judge_502s(log_path: Path, *, min_count: int) -> bool:
    if not log_path.exists():
        return False
    text = tail_text(log_path, max_bytes=4 * 1024 * 1024)
    if not text:
        return False
    recent_statuses = [
        int(match.group(1))
        for match in RESPONSES_STATUS_RE.finditer(text)
    ]
    if len(recent_statuses) < min_count:
        return False
    trailing = recent_statuses[-min_count:]
    return all(code >= 500 for code in trailing)


def back_half_log_stuck_on_judge_api_502(log_path: Path) -> bool:
    return back_half_log_has_trailing_judge_502s(log_path, min_count=6)


def _parse_scheduler_log_timestamp(text: str) -> float | None:
    value = text.strip()
    if not value:
        return None
    try:
        local_tz = datetime.now().astimezone().tzinfo
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=local_tz).timestamp()
    except Exception:
        return None


def back_half_log_has_dangling_responses_call(
    log_path: Path,
    *,
    now: float,
    max_dangling_seconds: int = RESPONSES_CALL_DANGLING_SECONDS,
) -> bool:
    if not log_path.exists():
        return False
    text = tail_text(log_path, max_bytes=4 * 1024 * 1024)
    if not text:
        return False
    last_started_epoch: float | None = None
    last_returned_epoch: float | None = None
    for line in text.splitlines():
        match = RESPONSES_STARTED_RE.match(line)
        if match:
            ts = _parse_scheduler_log_timestamp(match.group(1))
            if ts is not None:
                last_started_epoch = ts
            continue
        match = RESPONSES_RETURNED_RE.match(line)
        if match:
            ts = _parse_scheduler_log_timestamp(match.group(1))
            if ts is not None:
                last_returned_epoch = ts
    if last_started_epoch is None:
        return False
    if last_returned_epoch is not None and last_returned_epoch >= last_started_epoch:
        return False
    return (now - last_started_epoch) >= max_dangling_seconds


def back_half_state_has_legacy_retry_budget(back_half_state: dict[str, Any] | None) -> bool:
    if not isinstance(back_half_state, dict):
        return False
    try:
        max_attempts = int(back_half_state.get("max_attempts") or 0)
    except Exception:
        return False
    return 0 < max_attempts < LEGACY_BACK_HALF_MIN_RETRY_ATTEMPTS


def back_half_finished_state(case_id: str, kind: str) -> dict[str, Any] | None:
    state = load_back_half_rerun_state(case_id)
    if not isinstance(state, dict) or not state:
        return None
    if str(state.get("phase") or "").strip() != "finished":
        return None
    if str(state.get("kind") or "").strip() != kind:
        return None
    return state


def latest_submission_executed_metadata(case_id: str) -> Path | None:
    exact_submission_dir = continuation_submission_dir(case_id)
    if exact_submission_dir is not None:
        candidate = exact_submission_dir / "submission_executed_metadata.json"
        if candidate.exists():
            return candidate
    task_run_dir = latest_task_run_dir(case_id)
    if task_run_dir is None:
        return None
    candidates = sorted(
        task_run_dir.glob("submissions/*/submission_executed_metadata.json"),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    return candidates[-1] if candidates else None


def back_half_failure_looks_transient(case_id: str, kind: str, state: dict[str, Any]) -> bool:
    if bool(state.get("transient_network_failure")):
        return True
    texts: list[str] = []
    launch_log = latest_back_half_launch_log(case_id, kind)
    if launch_log is not None:
        texts.append(tail_text(launch_log, max_bytes=256 * 1024))
    task_run_dir = latest_task_run_dir(case_id)
    if task_run_dir is not None:
        texts.append(tail_text(task_run_dir / "run.log", max_bytes=256 * 1024))
        texts.append(tail_text(task_run_dir / "group.log", max_bytes=256 * 1024))
    executed_meta = latest_submission_executed_metadata(case_id)
    if executed_meta is not None:
        texts.append(tail_text(executed_meta, max_bytes=256 * 1024))
    return looks_like_transient_network_failure(*texts)


def back_half_failure_looks_retryable_infra(case_id: str, kind: str, state: dict[str, Any]) -> bool:
    texts: list[str] = []
    launch_log = latest_back_half_launch_log(case_id, kind)
    if launch_log is not None:
        texts.append(tail_text(launch_log, max_bytes=256 * 1024))
    task_run_dir = latest_task_run_dir(case_id)
    if task_run_dir is not None:
        texts.append(tail_text(task_run_dir / "run.log", max_bytes=256 * 1024))
        texts.append(tail_text(task_run_dir / "group.log", max_bytes=256 * 1024))
    executed_meta = latest_submission_executed_metadata(case_id)
    if executed_meta is not None:
        texts.append(tail_text(executed_meta, max_bytes=256 * 256 * 1024))
    texts.append(json.dumps(state, ensure_ascii=False))
    haystack = "\n".join(texts)
    markers = (
        "could not find an available, non-overlapping IPv4 address pool among the defaults to assign to the network",
        "RuntimeError(\"Kernel didn't respond in 60 seconds\")",
        "Kernel didn't respond in 60 seconds",
        "http+docker://localhost/v1.43/networks/create",
    )
    return any(marker in haystack for marker in markers)


def back_half_finished_state_retryable(case_id: str, kind: str) -> dict[str, Any] | None:
    state = back_half_finished_state(case_id, kind)
    if not state:
        return None
    if bool(state.get("transient_network_failure")):
        return state
    if back_half_failure_looks_transient(case_id, kind, state):
        return state
    if back_half_failure_looks_retryable_infra(case_id, kind, state):
        return state
    return None


def latest_back_half_launch_log(case_id: str, kind: str | None = None) -> Path | None:
    monitoring_root = case_root(case_id) / "monitoring"
    if kind:
        pattern = f"official_scheduler_{case_id}_{kind}_*.log"
    else:
        pattern = f"official_scheduler_{case_id}_*.log"
    candidates = sorted(
        monitoring_root.glob(pattern),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    return candidates[-1] if candidates else None


def latest_executed_metadata_mtime(case_id: str) -> float | None:
    exact_submission_dir = continuation_submission_dir(case_id)
    if exact_submission_dir is not None:
        exact_meta = exact_submission_dir / "submission_executed_metadata.json"
        if exact_meta.exists():
            return safe_stat_mtime(exact_meta)
    task_run_dir = latest_task_run_dir(case_id)
    if task_run_dir is None:
        return None
    executed_meta = sorted(
        task_run_dir.glob("submissions/*/submission_executed_metadata.json"),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    if not executed_meta:
        return None
    return safe_stat_mtime(executed_meta[-1])


def back_half_artifact_freshness_epoch(case_id: str) -> float | None:
    rerun_plan = back_half_rerun_plan(case_id)
    task_run_dir_raw = str(rerun_plan.get("task_run_dir") or "").strip()
    if not task_run_dir_raw:
        return None
    task_run_dir = Path(task_run_dir_raw)
    if not task_run_dir.is_absolute():
        task_run_dir = ROOT / task_run_dir

    candidates: list[Path] = []
    grade_path = task_run_dir / "grade.json"
    if grade_path.exists():
        candidates.append(grade_path)

    for path in sorted(
        task_run_dir.glob("submissions/*/submission_executed_grader_output_0.json"),
        key=lambda item: (item.stat().st_mtime, str(item)),
    ):
        candidates.append(path)

    for path in sorted(
        task_run_dir.glob("submissions/*/submission_executed_metadata.json"),
        key=lambda item: (item.stat().st_mtime, str(item)),
    ):
        candidates.append(path)

    for relative in ("run.log", "../group.log"):
        path = (task_run_dir / relative).resolve()
        if path.exists():
            candidates.append(path)

    mtimes = [value for value in (safe_stat_mtime(path) for path in candidates) if value is not None]
    return max(mtimes) if mtimes else None


def back_half_rerun_plan(case_id: str) -> dict[str, Any]:
    """Infer whether a case should only rerun reproduction/grading or grading.

    Prefer the exact checkpoint recorded in official_continuation_state when it
    exists. Falling back to the latest task run can pick up a different
    submission than the one the official wrapper asked us to continue from.
    """

    exact_submission_dir = continuation_submission_dir(case_id)
    exact_submission_tar = continuation_submission_tar_for_scheduler(case_id)
    task_run_dir = (
        task_run_dir_for_submission_dir(exact_submission_dir)
        if exact_submission_dir is not None
        else latest_task_run_dir(case_id)
    )
    payload: dict[str, Any] = {
        "case_id": case_id,
        "kind": None,
        "run_root": "",
        "task_run_dir": "",
        "task_done": False,
        "metadata_ok": False,
        "submission_exists": False,
        "executed_submission_exists": False,
        "executed_metadata_exists": False,
        "has_grade": False,
        "rollout_succeeded": False,
        "reproduction_succeeded": False,
        "judge_success": False,
        "monitor_violation_count": 0,
    }
    if task_run_dir is None:
        return payload

    run_root = run_root_for_task_run_dir(task_run_dir) or latest_run_root(case_id)
    if run_root is not None:
        payload["run_root"] = str(run_root.relative_to(ROOT))
    payload["task_run_dir"] = str(task_run_dir.relative_to(ROOT))

    status = load_json(task_run_dir / "status.json")
    metadata = load_json(task_run_dir / "metadata.json")
    payload["task_done"] = isinstance(status, dict) and str(status.get("status") or "") == "done"
    payload["metadata_ok"] = isinstance(metadata, dict) and not bool(
        str(metadata.get("error_msg") or "").strip()
    )

    if exact_submission_tar is not None and exact_submission_dir is not None:
        submission_tars = [exact_submission_tar] if exact_submission_tar.exists() else []
        executed_tars = (
            [exact_submission_dir / "submission_executed.tar.gz"]
            if (exact_submission_dir / "submission_executed.tar.gz").exists()
            else []
        )
        executed_meta = (
            [exact_submission_dir / "submission_executed_metadata.json"]
            if (exact_submission_dir / "submission_executed_metadata.json").exists()
            else []
        )
        if not executed_tars:
            executed_tars = sorted(
                task_run_dir.glob("submissions/*/submission_executed.tar.gz"),
                key=lambda path: (path.stat().st_mtime, str(path)),
            )
        if not executed_meta:
            executed_meta = sorted(
                task_run_dir.glob("submissions/*/submission_executed_metadata.json"),
                key=lambda path: (path.stat().st_mtime, str(path)),
            )
    else:
        submission_tars = sorted(
            task_run_dir.glob("submissions/*/submission.tar.gz"),
            key=lambda path: (path.stat().st_mtime, str(path)),
        )
        executed_tars = sorted(
            task_run_dir.glob("submissions/*/submission_executed.tar.gz"),
            key=lambda path: (path.stat().st_mtime, str(path)),
        )
        executed_meta = sorted(
            task_run_dir.glob("submissions/*/submission_executed_metadata.json"),
            key=lambda path: (path.stat().st_mtime, str(path)),
        )
    payload["submission_exists"] = bool(submission_tars)
    payload["executed_submission_exists"] = bool(executed_tars)
    payload["executed_metadata_exists"] = bool(executed_meta)

    grade_path = task_run_dir / "grade.json"
    if grade_path.exists():
        data = load_json(grade_path)
        pb = data.get("paperbench_result") or {} if isinstance(data, dict) else {}
        agent = pb.get("agent_output") or {} if isinstance(pb, dict) else {}
        repro = pb.get("reproduction_metadata") or {} if isinstance(pb, dict) else {}
        judge = pb.get("judge_output") or {} if isinstance(pb, dict) else {}
        monitor = pb.get("monitor_result") or {} if isinstance(pb, dict) else {}
        violations = monitor.get("violations") or [] if isinstance(monitor, dict) else []

        judge_success = judge_payload_succeeded(judge)
        reproduction_succeeded = reproduction_metadata_succeeded(repro)
        rollout_succeeded = bool(agent) and not bool(str(agent.get("error_msg") or "").strip())

        payload.update(
            {
                "has_grade": True,
                "submission_exists": bool(pb.get("submission_exists")) or payload["submission_exists"],
                "rollout_succeeded": rollout_succeeded,
                "reproduction_succeeded": reproduction_succeeded,
                "judge_success": judge_success,
                "monitor_violation_count": len(violations) if isinstance(violations, list) else 0,
            }
        )
        if payload["submission_exists"] and rollout_succeeded and payload["monitor_violation_count"] == 0:
            if not reproduction_succeeded:
                payload["kind"] = "reproduction_grading"
            elif not judge_success:
                payload["kind"] = "grading_only"
        return payload

    payload["rollout_succeeded"] = payload["task_done"] and payload["metadata_ok"] and payload["submission_exists"]
    if payload["executed_metadata_exists"]:
        executed_meta_payload = load_json(executed_meta[-1])
        payload["reproduction_succeeded"] = reproduction_metadata_succeeded(executed_meta_payload)
    if payload["rollout_succeeded"]:
        if payload["reproduction_succeeded"]:
            payload["kind"] = "grading_only"
        else:
            payload["kind"] = "reproduction_grading"
    return payload


def wrapper_can_yield_to_back_half(case_id: str, rerun_plan: dict[str, Any]) -> bool:
    """Whether a live official wrapper can be safely replaced by back-half rerun.

    There are two cases where the wrapper should yield:

    1. The wrapper already recorded an explicit recovery handoff
       (`needs_continuation` / `salvaged_*`) and a concrete back-half plan is
       available. In that state the wrapper is only blocking scheduler progress.
    2. The latest task run has already finished rollout and reproduction
       successfully and only grading remains.
    """

    rerun_kind = str(rerun_plan.get("kind") or "").strip()
    if not rerun_kind:
        return False

    recovery = load_recovery_state(case_id)
    recovery_phase = str(recovery.get("phase") or "").strip()
    if recovery_phase in {"needs_continuation", "salvaged_timeout", "salvaged_late_exit"}:
        return True

    return (
        rerun_kind == "grading_only"
        and bool(rerun_plan.get("task_done"))
        and bool(rerun_plan.get("metadata_ok"))
        and bool(rerun_plan.get("rollout_succeeded"))
        and bool(rerun_plan.get("reproduction_succeeded"))
        and not bool(rerun_plan.get("has_grade"))
    )


def active_back_half_processes() -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    queue = parse_queue()
    now = time.time()
    for item in queue:
        case_id = item["case_id"]
        state, invalid_reason = active_back_half_state_details(case_id, now=now)
        if state is None:
            continue
        payload[case_id] = {
            "pid": int(state["pid"]),
            "kind": str(state.get("kind") or "").strip() or "reproduction_grading",
            "gpu_id": str(state.get("gpu_id") or "").strip(),
            "cmd": "from back_half_rerun_state",
            "state_source": "back_half_rerun_state",
        }
    try:
        ps_text = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    except Exception:
        return payload

    for line in ps_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            continue
        pid_text, cmd_text = parts
        has_reproduction_rerun = (
            "rerun_reproduction_and_grading.py" in cmd_text
            or "sota.scripts.rerun_reproduction_and_grading" in cmd_text
        )
        has_grading_rerun = (
            "rerun_grading_only.py" in cmd_text
            or "sota.scripts.rerun_grading_only" in cmd_text
        )
        if not has_reproduction_rerun and not has_grading_rerun:
            continue
        if "rg rerun_reproduction_and_grading" in cmd_text or "rg rerun_grading_only" in cmd_text:
            continue
        try:
            pid = int(pid_text)
        except Exception:
            continue
        if not pid_alive(pid):
            continue
        try:
            tokens = shlex.split(cmd_text)
        except Exception:
            tokens = cmd_text.split()
        kind = ""
        if any(
            "rerun_reproduction_and_grading.py" in token
            or token == "sota.scripts.rerun_reproduction_and_grading"
            for token in tokens
        ):
            kind = "reproduction_grading"
        elif any(
            "rerun_grading_only.py" in token
            or token == "sota.scripts.rerun_grading_only"
            for token in tokens
        ):
            kind = "grading_only"
        if not kind:
            continue
        gpu_id = ""
        cases: list[str] = []
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token == "--case" and idx + 1 < len(tokens):
                case_id = str(tokens[idx + 1]).strip()
                if case_id:
                    cases.append(case_id)
                idx += 2
                continue
            if token == "--gpu-id" and idx + 1 < len(tokens):
                gpu_id = str(tokens[idx + 1]).strip()
                idx += 2
                continue
            idx += 1
        for case_id in cases:
            inferred_gpu_id = gpu_id
            if not inferred_gpu_id and kind == "reproduction_grading":
                status = load_json(supervisor_status_path(case_id))
                if isinstance(status, dict):
                    inferred_gpu_id = str(status.get("official_gpu_id") or "").strip()
            existing = payload.get(case_id)
            if existing is None or str(existing.get("state_source") or "") != "back_half_rerun_state":
                payload[case_id] = {
                    "pid": pid,
                    "kind": kind,
                    "gpu_id": inferred_gpu_id,
                    "cmd": cmd_text,
                    "state_source": "ps_scan",
                }
    return payload


def health_check_targets(
    queue: list[dict[str, str]],
    *,
    active_back_half: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    targets: list[str] = []
    active_back_half = active_back_half or {}
    for item in queue:
        case_id = item["case_id"]
        status = load_json(supervisor_status_path(case_id))
        recovery = load_recovery_state(case_id)
        latest_run = latest_run_root(case_id)
        if case_id in active_back_half:
            targets.append(case_id)
            continue
        if isinstance(status, dict) and not bool(status.get("official_finished")):
            targets.append(case_id)
            continue
        if isinstance(recovery, dict) and recovery.get("phase") in {
            "running",
            "needs_continuation",
            "salvaged_timeout",
            "salvaged_late_exit",
        }:
            targets.append(case_id)
            continue
        if back_half_rerun_plan(case_id).get("kind"):
            targets.append(case_id)
            continue
        if latest_run is not None and launch_record_path(case_id).exists():
            launch_record = load_launch_record(case_id)
            if isinstance(launch_record, dict) and launch_record.get("run_root") == str(
                latest_run.relative_to(ROOT)
            ):
                targets.append(case_id)
    return sorted(set(targets))


def pid_candidates_for_case(case_id: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()

    def add_pid(raw: Any) -> None:
        try:
            value = int(raw)
        except Exception:
            return
        if value > 0 and value not in seen:
            values.append(value)
            seen.add(value)

    status = load_json(supervisor_status_path(case_id))
    if isinstance(status, dict):
        add_pid(status.get("official_wrapper_pid"))
        pid_path = str(status.get("official_pid_path") or "").strip()
        if pid_path:
            pid_file = ROOT / pid_path if not Path(pid_path).is_absolute() else Path(pid_path)
            if pid_file.exists():
                add_pid(pid_file.read_text(encoding="utf-8").strip())

    pid_file = case_root(case_id) / "monitoring" / "official.pid"
    if pid_file.exists():
        add_pid(pid_file.read_text(encoding="utf-8").strip())

    launch_record = load_launch_record(case_id)
    if isinstance(launch_record, dict):
        add_pid(launch_record.get("launcher_pid"))

    return values


def live_rollout_pids_for_case(case_id: str, *, now: float | None = None) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()

    def add_pid(raw: Any) -> None:
        try:
            value = int(raw)
        except Exception:
            return
        if value > 0 and pid_alive(value) and value not in seen:
            values.append(value)
            seen.add(value)

    now = now if now is not None else time.time()
    status = load_json(supervisor_status_path(case_id))
    launch_record = load_launch_record(case_id)
    back_half_launch_record = launch_record_is_back_half_rerun(case_id)
    active_state, _ = active_back_half_state_details(case_id, now=now)
    supervisor_counts = supervisor_pids_should_count_as_live_rollout(case_id, now=now)

    if isinstance(status, dict) and supervisor_counts:
        add_pid(status.get("official_wrapper_pid"))
        pid_path = str(status.get("official_pid_path") or "").strip()
        if pid_path:
            pid_file = ROOT / pid_path if not Path(pid_path).is_absolute() else Path(pid_path)
            if pid_file.exists():
                add_pid(pid_file.read_text(encoding="utf-8").strip())

    pid_file = case_root(case_id) / "monitoring" / "official.pid"
    if supervisor_counts and pid_file.exists():
        add_pid(pid_file.read_text(encoding="utf-8").strip())

    # A back-half launch record does not represent a rollout wrapper. Counting it
    # as rollout here makes handoff logic kill the newly launched back-half worker
    # when it only intends to stop stale rollout processes.
    if isinstance(launch_record, dict) and not back_half_launch_record:
        add_pid(launch_record.get("launcher_pid"))

    return values


def _stop_pid_values(
    case_id: str,
    pid_values: list[int],
    *,
    mark_back_half_state: bool,
    event_name: str,
) -> dict[str, Any]:
    pid_values = [int(pid) for pid in pid_values if int(pid) > 0]

    terminated_pids: list[int] = []
    killed_pids: list[int] = []
    already_dead: list[int] = []

    pgid_values: list[int] = []
    seen_pgids: set[int] = set()
    for pid in pid_values:
        pgid = process_group_id(pid)
        if pgid is not None and pgid not in seen_pgids:
            pgid_values.append(pgid)
            seen_pgids.add(pgid)

    terminated_pgids: list[int] = []
    killed_pgids: list[int] = []

    for pgid in pgid_values:
        try:
            os.killpg(pgid, signal.SIGTERM)
            terminated_pgids.append(pgid)
        except ProcessLookupError:
            pass
        except Exception:
            pass

    for pid in pid_values:
        if not pid_alive(pid):
            already_dead.append(pid)
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            terminated_pids.append(pid)
        except ProcessLookupError:
            already_dead.append(pid)
        except Exception:
            pass

    deadline = time.time() + 20
    while time.time() < deadline:
        alive = [pid for pid in pid_values if pid_alive(pid)]
        if not alive:
            break
        time.sleep(1)

    for pgid in pgid_values:
        survivors = [pid for pid in pid_values if process_group_id(pid) == pgid and pid_alive(pid)]
        if not survivors:
            continue
        try:
            os.killpg(pgid, signal.SIGKILL)
            killed_pgids.append(pgid)
        except ProcessLookupError:
            pass
        except Exception:
            pass

    for pid in pid_values:
        if not pid_alive(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed_pids.append(pid)
        except ProcessLookupError:
            pass
        except Exception:
            pass

    payload = {
        "case_id": case_id,
        "pid_candidates": pid_values,
        "pgid_candidates": pgid_values,
        "terminated_pids": terminated_pids,
        "killed_pids": killed_pids,
        "terminated_pgids": terminated_pgids,
        "killed_pgids": killed_pgids,
        "already_dead": already_dead,
        "stopped_at_utc": utc_now(),
    }

    if mark_back_half_state:
        state_path = back_half_rerun_state_path(case_id)
        state = load_json(state_path)
        if isinstance(state, dict) and state:
            terminal_keys = (
                "score",
                "judge_success",
                "reproduction_succeeded",
                "repro_exit_code",
                "transient_network_failure",
            )
            keep_finished = str(state.get("phase") or "").strip() == "finished" or any(
                key in state for key in terminal_keys
            )
            state.update(
                {
                    "phase": "finished" if keep_finished else "stopped",
                    "last_updated_utc": payload["stopped_at_utc"],
                    "stopped_by_scheduler": True,
                }
            )
            write_json(state_path, state)

    event(event_name, payload)
    return payload


def stop_rollout_processes(case_id: str, *, now: float | None = None) -> dict[str, Any]:
    return _stop_pid_values(
        case_id,
        live_rollout_pids_for_case(case_id, now=now),
        mark_back_half_state=False,
        event_name="scheduler_stopped_rollout_processes",
    )


def stop_case_processes(
    case_id: str,
    *,
    active_back_half: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pid_values = pid_candidates_for_case(case_id)
    active_back_half = active_back_half or {}
    back_half = active_back_half.get(case_id)
    if isinstance(back_half, dict):
        try:
            back_half_pid = int(back_half.get("pid") or 0)
        except Exception:
            back_half_pid = 0
        if back_half_pid > 0 and back_half_pid not in pid_values:
            pid_values.append(back_half_pid)

    return _stop_pid_values(
        case_id,
        pid_values,
        mark_back_half_state=True,
        event_name="scheduler_stopped_case_processes",
    )


def run_health_check(
    queue: list[dict[str, str]],
    *,
    stall_seconds: int,
    min_run_age_before_stall_seconds: int,
    dry_run: bool,
    active_back_half: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    now = time.time()
    active_back_half = active_back_half or {}
    targets = health_check_targets(queue, active_back_half=active_back_half)
    for case_id in targets:
        run_root = latest_run_root(case_id)
        if run_root is None:
            continue

        monitoring_root = case_root(case_id) / "monitoring"
        log_candidates = [
            run_root / "logs" / "official_wrapper.log",
            run_root / "logs" / "official_run.log",
            process_events_path(case_id),
            monitoring_root / "supervisor_status.json",
        ]
        heartbeat_mtime = latest_mtime(log_candidates)
        run_started_at = parse_run_root_started_at(run_root)
        run_age_seconds = int(now - run_started_at) if run_started_at is not None else None
        stale_for_seconds = (
            max(0, int(now - heartbeat_mtime)) if heartbeat_mtime is not None else None
        )
        official_run_log = run_root / "logs" / "official_run.log"
        status = load_json(supervisor_status_path(case_id))
        status_phase = status.get("phase") if isinstance(status, dict) else None
        recovery = load_recovery_state(case_id)
        recovery_phase = recovery.get("phase") if isinstance(recovery, dict) else None
        live_pids = live_rollout_pids_for_case(case_id, now=now)
        back_half = active_back_half.get(case_id) or {}
        back_half_pid = int(back_half.get("pid") or 0) if back_half else 0
        active_state, inactive_back_half_reason = active_back_half_state_details(case_id, now=now)
        rerun_plan = back_half_rerun_plan(case_id)
        executed_meta_mtime = latest_executed_metadata_mtime(case_id)
        executed_meta_age_seconds = (
            max(0, int(now - executed_meta_mtime)) if executed_meta_mtime is not None else None
        )
        artifact_freshness_epoch = back_half_artifact_freshness_epoch(case_id)
        artifact_stale_for_seconds = (
            max(0, int(now - artifact_freshness_epoch))
            if artifact_freshness_epoch is not None
            else None
        )

        reason = ""
        severity = "info"
        should_stop = False
        should_mark_clean_rerun = False

        back_half_log = latest_back_half_launch_log(case_id, str(back_half.get("kind") or "").strip() or None)
        back_half_log_age_seconds = (
            max(0, int(now - safe_stat_mtime(back_half_log)))  # type: ignore[arg-type]
            if back_half_log is not None and safe_stat_mtime(back_half_log) is not None
            else None
        )

        if (
            back_half_pid
            and artifact_stale_for_seconds is not None
            and artifact_stale_for_seconds >= BACK_HALF_ARTIFACT_STALE_SECONDS
            and back_half_log_age_seconds is not None
            and back_half_log_age_seconds >= BACK_HALF_ARTIFACT_STALE_SECONDS
        ):
            # Back-half reproduction work often runs almost entirely inside the container,
            # so the host-side executed artifacts and the scheduler launch log can stay
            # unchanged for hours while the task is still making real progress
            # (`/submission/reproduce.log`, in-container training loops, dataset download,
            # etc.). We still surface the observation, but do not recycle the live worker
            # based on host artifact freshness alone because that produced repeated false
            # positives for long-running reproductions.
            reason = "back_half_artifacts_stale_observed"
            severity = "warning"
            should_stop = False
            should_mark_clean_rerun = False
        elif (
            rerun_plan.get("kind") == "grading_only"
            and rerun_plan.get("reproduction_succeeded")
            and back_half_pid
            and back_half_log is not None
            and back_half_state_has_legacy_retry_budget(active_state)
            and back_half_log_has_trailing_judge_502s(back_half_log, min_count=4)
        ):
            # Judge-side 5xxs are expected to self-heal inside the grading-only rerun
            # worker, which already has long retry budgets. Do not kill a live worker
            # just because the latest responses are 5xxs; only surface the condition.
            reason = "grading_only_retrying_after_judge_api_5xx"
            severity = "warning"
            should_stop = False
            should_mark_clean_rerun = False
        elif (
            rerun_plan.get("kind") == "grading_only"
            and rerun_plan.get("reproduction_succeeded")
            and back_half_pid
            and back_half_log is not None
            and back_half_log_stuck_on_judge_api_502(back_half_log)
        ):
            # Same rule for the non-legacy path: let the grading worker own retries and
            # only re-launch after it actually exits.
            reason = "grading_only_retrying_after_judge_api_5xx"
            severity = "warning"
            should_stop = False
            should_mark_clean_rerun = False
        elif (
            rerun_plan.get("kind") == "reproduction_grading"
            and rerun_plan.get("reproduction_succeeded")
            and back_half_pid
            and back_half_log is not None
            and back_half_log_stuck_on_judge_api_502(back_half_log)
        ):
            # Match the grading-only path: if reproduction has already succeeded and the
            # remaining work is judge-side, let the live worker own retry handling for
            # transient 5xxs instead of recycling it underneath the user.
            reason = "reproduction_grading_retrying_after_judge_api_5xx"
            severity = "warning"
            should_stop = False
            should_mark_clean_rerun = False
        elif (
            back_half_pid
            and back_half_log is not None
            and back_half_log_has_dangling_responses_call(back_half_log, now=now)
        ):
            # Judge calls through the shared gateway can sit in a long retry window
            # after a `Responses call started` line. Recycling the live worker here
            # caused false positives during grading-only recovery, so surface the
            # condition and let the rerun worker's retry budget own it.
            reason = "back_half_dangling_responses_call"
            severity = "warning"
            should_stop = False
            should_mark_clean_rerun = False
        elif (
            back_half_pid
            and rerun_plan.get("kind")
            and back_half.get("kind")
            and str(back_half.get("kind")) != str(rerun_plan.get("kind"))
        ):
            reason = (
                "active_back_half_kind_mismatch:"
                f"expected={rerun_plan.get('kind')},actual={back_half.get('kind')}"
            )
            severity = "warning"
            should_stop = True
            should_mark_clean_rerun = False
        elif recovery_phase == "needs_continuation" and rerun_plan.get("kind") and not live_pids:
            reason = f"rollout_complete_{rerun_plan['kind']}_pending"
            severity = "info"
            should_stop = False
            should_mark_clean_rerun = False
        elif recovery_phase == "needs_continuation":
            reason = ""
            severity = "info"
            should_stop = False
            should_mark_clean_rerun = False
        elif inactive_back_half_reason and rerun_plan.get("kind") and not live_pids:
            reason = f"stale_back_half_state:{inactive_back_half_reason}"
            severity = "warning"
            should_stop = False
            should_mark_clean_rerun = False
        elif back_half_pid:
            reason = f"active_{back_half.get('kind')}_rerun_process"
            severity = "info"
            should_stop = False
            should_mark_clean_rerun = False
        elif not live_pids and rerun_plan.get("kind"):
            reason = f"rollout_complete_waiting_for_{rerun_plan['kind']}_rerun"
            severity = "warning"
            should_stop = False
            should_mark_clean_rerun = False
        elif not live_pids:
            reason = "active_case_without_live_process"
            severity = "error"
            should_mark_clean_rerun = True
        elif (
            run_age_seconds is not None
            and run_age_seconds >= min_run_age_before_stall_seconds
            and stale_for_seconds is not None
            and stale_for_seconds >= stall_seconds
        ):
            reason = "host_heartbeat_stalled"
            severity = "error"
            should_stop = True
            should_mark_clean_rerun = True
        elif (
            run_age_seconds is not None
            and run_age_seconds >= min_run_age_before_stall_seconds
            and wrapper_log.exists()
            and official_run_log.exists()
            and wrapper_log.stat().st_size == official_run_log.stat().st_size
            and stale_for_seconds is not None
            and stale_for_seconds >= max(900, stall_seconds // 2)
        ):
            reason = "wrapper_and_run_logs_frozen_together"
            severity = "warning"

        finding = {
            "case_id": case_id,
            "run_root": str(run_root.relative_to(ROOT)),
            "status_phase": status_phase,
            "recovery_phase": recovery_phase,
            "run_age_seconds": run_age_seconds,
            "stale_for_seconds": stale_for_seconds,
            "heartbeat_paths": [
                str(path.relative_to(ROOT))
                for path in log_candidates
                if path.exists()
            ],
            "live_pids": live_pids,
            "back_half_pid": back_half_pid,
            "back_half_kind": back_half.get("kind") or "",
            "back_half_state_phase": active_state.get("phase") if isinstance(active_state, dict) else "",
            "back_half_state_reason": inactive_back_half_reason,
            "back_half_rerun_needed": rerun_plan.get("kind"),
            "executed_meta_age_seconds": executed_meta_age_seconds,
            "back_half_log_age_seconds": back_half_log_age_seconds,
            "back_half_artifact_stale_for_seconds": artifact_stale_for_seconds,
            "reason": reason,
            "severity": severity,
            "stop_requested": should_stop,
            "clean_rerun_requested": should_mark_clean_rerun,
        }

        if should_stop and not dry_run:
            finding["stop_result"] = stop_case_processes(case_id, active_back_half=active_back_half)

        if should_mark_clean_rerun and not dry_run:
            payload = {
                "case_id": case_id,
                "reason": reason or "health_check_failed",
                "run_root": str(run_root.relative_to(ROOT)),
                "health_check_detected_at_utc": utc_now(),
                "run_age_seconds": run_age_seconds,
                "stale_for_seconds": stale_for_seconds,
            }
            write_json(recovery_state_path(case_id), {"phase": "needs_clean_rerun", **payload})
            status_payload = load_json(supervisor_status_path(case_id))
            if not isinstance(status_payload, dict):
                status_payload = {"case_id": case_id}
            status_payload.update(
                {
                    "phase": "paperbench_rollout_failed",
                    "official_finished": False,
                    "official_failed": True,
                    "official_needs_continuation": False,
                    "official_failure_reason": reason or "health_check_failed",
                    "official_continuation_reason": "",
                    "last_updated_utc": utc_now(),
                }
            )
            write_json(supervisor_status_path(case_id), status_payload)
            event("scheduler_health_check_marked_clean_rerun", payload)

        findings.append(finding)
    return findings


def latest_result_summary(case_id: str) -> dict[str, Any]:
    payload = {
        "has_grade": False,
        "score": None,
        "submission_exists": False,
        "rollout_succeeded": False,
        "reproduction_succeeded": False,
        "judge_success": False,
        "looks_complete": False,
        "monitor_violation_count": 0,
    }
    rerun_plan = back_half_rerun_plan(case_id)
    payload.update(
        {
            "submission_exists": bool(rerun_plan.get("submission_exists")),
            "rollout_succeeded": bool(rerun_plan.get("rollout_succeeded")),
            "reproduction_succeeded": bool(rerun_plan.get("reproduction_succeeded")),
            "judge_success": bool(rerun_plan.get("judge_success")),
            "monitor_violation_count": int(rerun_plan.get("monitor_violation_count") or 0),
        }
    )

    grade_files = sorted(
        (case_root(case_id) / "official_runs").glob("*/runs/**/grade.json"),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    if not grade_files:
        return payload

    grade_path = grade_files[-1]
    data = load_json(grade_path)
    if not isinstance(data, dict):
        return payload
    paperbench_result = data.get("paperbench_result") or {}
    if not isinstance(paperbench_result, dict):
        return payload
    agent = paperbench_result.get("agent_output") or {}
    repro = paperbench_result.get("reproduction_metadata") or {}
    judge = paperbench_result.get("judge_output") or {}
    score = data.get("score")

    judge_success = judge_payload_succeeded(judge)
    reproduction_succeeded = reproduction_metadata_succeeded(repro)
    rollout_succeeded = bool(agent) and not bool(str(agent.get("error_msg") or "").strip())

    payload.update(
        {
            "has_grade": True,
            "score": score,
            "submission_exists": bool(paperbench_result.get("submission_exists")) or payload["submission_exists"],
            "rollout_succeeded": rollout_succeeded,
            "reproduction_succeeded": reproduction_succeeded,
            "judge_success": judge_success,
        }
    )
    payload["looks_complete"] = (
        payload["submission_exists"]
        and payload["rollout_succeeded"]
        and payload["reproduction_succeeded"]
        and payload["judge_success"]
        and score is not None
        and not (isinstance(score, float) and math.isnan(score))
    )
    return payload


def reconcile_supervisor_status(case_id: str, *, dry_run: bool = False) -> dict[str, Any] | None:
    """Clear stale `official_finished=true` flags for invalid historical runs."""

    status_path = supervisor_status_path(case_id)
    status = load_json(status_path)
    if not isinstance(status, dict) or not status:
        return None

    if not bool(status.get("official_finished")):
        return None

    # Do not rewrite status for still-active processes.
    active = False
    if pid_alive(status.get("official_wrapper_pid")):
        active = True
    pid_path = str(status.get("official_pid_path") or "").strip()
    if pid_path:
        pid_file = ROOT / pid_path if not Path(pid_path).is_absolute() else Path(pid_path)
        if pid_file.exists() and pid_alive(pid_file.read_text(encoding="utf-8").strip()):
            active = True
    pid_file = case_root(case_id) / "monitoring" / "official.pid"
    if pid_file.exists() and pid_alive(pid_file.read_text(encoding="utf-8").strip()):
        active = True
    launch_record = load_launch_record(case_id)
    if isinstance(launch_record, dict) and pid_alive(launch_record.get("launcher_pid")):
        active = True
    if active:
        return None

    summary = latest_result_summary(case_id)
    if summary.get("looks_complete"):
        return None

    payload = {
        "case_id": case_id,
        "previous_phase": status.get("phase"),
        "new_phase": "paperbench_rollout_invalidated",
        "dry_run": dry_run,
        "reason": "historical_result_not_complete",
        "summary": summary,
        "reconciled_at_utc": utc_now(),
    }
    if dry_run:
        return payload

    status.update(
        {
            "phase": "paperbench_rollout_invalidated",
            "official_finished": False,
            "official_failed": True,
            "official_needs_continuation": False,
            "official_failure_reason": "historical_result_not_complete",
            "official_continuation_reason": "",
            "last_updated_utc": utc_now(),
        }
    )
    write_json(status_path, status)
    event("scheduler_reconciled_supervisor_status", payload)
    return payload


def already_launched_for(case_id: str, paper_id: str) -> bool:
    del paper_id
    recovery = load_recovery_state(case_id)
    if recovery.get("phase") in {"needs_clean_rerun", "needs_continuation"}:
        return False
    return bool(latest_result_summary(case_id).get("looks_complete"))


def cleanup_archive_root(case_id: str, reason: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", reason.strip() or "unknown").strip("._-")
    slug = (slug or "unknown")[:80]
    return case_root(case_id) / "monitoring" / "cleanup_archive" / f"{stamp()}_{slug}"


def clean_failed_outputs_for_rerun(case_id: str, dry_run: bool) -> dict[str, Any] | None:
    recovery = load_recovery_state(case_id)
    if recovery.get("phase") != "needs_clean_rerun":
        return None
    if back_half_rerun_plan(case_id).get("kind"):
        return None

    reason = str(recovery.get("reason") or "needs_clean_rerun")
    archive_root = cleanup_archive_root(case_id, reason)
    monitoring_root = case_root(case_id) / "monitoring"
    scheduler_logs = sorted(monitoring_root.glob("official_scheduler_*.log"))
    sources: list[tuple[Path, Path]] = [
        (case_root(case_id) / "official_runs", Path("official_runs")),
        (monitoring_root / "official.pid", Path("monitoring") / "official.pid"),
        (recovery_state_path(case_id), Path("monitoring") / "official_recovery_state.json"),
        (continuation_state_path(case_id), Path("monitoring") / "official_continuation_state.json"),
        (launch_record_path(case_id), Path("scheduler_state") / f"{case_id}.json"),
    ]
    sources.extend(
        (path, Path("monitoring") / path.name)
        for path in scheduler_logs
    )

    moved: list[str] = []
    missing: list[str] = []
    for src, rel_dst in sources:
        if not src.exists():
            missing.append(str(src.relative_to(ROOT)))
            continue
        moved.append(str(src.relative_to(ROOT)))
        if dry_run:
            continue
        dst = archive_root / rel_dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise RuntimeError(f"cleanup archive destination already exists: {dst}")
        shutil.move(str(src), str(dst))

    payload: dict[str, Any] = {
        "case_id": case_id,
        "reason": reason,
        "archive_root": str(archive_root.relative_to(ROOT)),
        "moved": moved,
        "missing": missing,
        "dry_run": dry_run,
        "cleaned_at_utc": utc_now(),
    }

    if not dry_run:
        write_json(archive_root / "cleanup_summary.json", payload)
        event("scheduler_cleaned_failed_case", payload)
    return payload


def archive_continuation_state(case_id: str, dry_run: bool) -> dict[str, Any] | None:
    """Archive a continuation run for manual cleanup.

    This must not run inline before launching a continuation attempt because the
    continuation state points at the previous run root and checkpoint files. If
    those are moved away first, the relaunched wrapper cannot recover the prior
    submission state.
    """
    recovery = load_recovery_state(case_id)
    if recovery.get("phase") != "needs_continuation":
        return None

    continuation = load_continuation_state(case_id)
    run_root_raw = str(continuation.get("run_root") or recovery.get("run_root") or "").strip()
    if not run_root_raw:
        return None

    run_root = ROOT / run_root_raw if not Path(run_root_raw).is_absolute() else Path(run_root_raw)
    if not run_root.exists():
        return None

    archive_root = cleanup_archive_root(case_id, str(recovery.get("reason") or "needs_continuation"))
    monitoring_root = case_root(case_id) / "monitoring"
    scheduler_logs = sorted(monitoring_root.glob("official_scheduler_*.log"))
    sources: list[tuple[Path, Path]] = [
        (run_root, Path("official_runs") / run_root.name),
        (monitoring_root / "official.pid", Path("monitoring") / "official.pid"),
        (recovery_state_path(case_id), Path("monitoring") / "official_recovery_state.json"),
        (continuation_state_path(case_id), Path("monitoring") / "official_continuation_state.json"),
        (launch_record_path(case_id), Path("scheduler_state") / f"{case_id}.json"),
    ]
    sources.extend((path, Path("monitoring") / path.name) for path in scheduler_logs)

    moved: list[str] = []
    missing: list[str] = []
    for src, rel_dst in sources:
        if not src.exists():
            missing.append(str(src.relative_to(ROOT)))
            continue
        moved.append(str(src.relative_to(ROOT)))
        if dry_run:
            continue
        dst = archive_root / rel_dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise RuntimeError(f"continuation archive destination already exists: {dst}")
        shutil.move(str(src), str(dst))

    payload: dict[str, Any] = {
        "case_id": case_id,
        "reason": str(recovery.get("reason") or "needs_continuation"),
        "archive_root": str(archive_root.relative_to(ROOT)),
        "moved": moved,
        "missing": missing,
        "dry_run": dry_run,
        "cleaned_at_utc": utc_now(),
    }

    if not dry_run:
        write_json(archive_root / "cleanup_summary.json", payload)
        event("scheduler_archived_continuation_case", payload)
    return payload


def active_cases(
    queue: list[dict[str, str]],
    *,
    active_back_half: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    active: dict[str, list[str]] = {}
    active_back_half = active_back_half or {}
    now = time.time()
    for item in queue:
        case_id = item["case_id"]
        reasons: list[str] = []
        rerun_plan = back_half_rerun_plan(case_id)
        wrapper_yieldable = wrapper_can_yield_to_back_half(case_id, rerun_plan)

        active_state, _ = active_back_half_state_details(case_id, now=now)
        back_half = active_back_half.get(case_id)
        back_half_launch_record = launch_record_is_back_half_rerun(case_id)
        supervisor_counts_as_live_rollout = supervisor_pids_should_count_as_live_rollout(case_id, now=now)
        if isinstance(back_half, dict) and pid_alive(back_half.get("pid")):
            reasons.append(f"{back_half.get('kind')}_rerun_pid from active process scan")

        status = load_json(case_root(case_id) / "monitoring" / "supervisor_status.json")
        if isinstance(status, dict):
            if (
                not wrapper_yieldable
                and supervisor_counts_as_live_rollout
                and pid_alive(status.get("official_wrapper_pid"))
            ):
                reasons.append("official_wrapper_pid from supervisor_status.json")
            pid_path = str(status.get("official_pid_path") or "").strip()
            if pid_path:
                pid_file = ROOT / pid_path if not Path(pid_path).is_absolute() else Path(pid_path)
                if (
                    not wrapper_yieldable
                    and supervisor_counts_as_live_rollout
                    and pid_file.exists()
                    and pid_alive(pid_file.read_text(encoding="utf-8").strip())
                ):
                    reasons.append("official_pid_path from supervisor_status.json")

        pid_file = case_root(case_id) / "monitoring" / "official.pid"
        if (
            not wrapper_yieldable
            and supervisor_counts_as_live_rollout
            and pid_file.exists()
            and pid_alive(pid_file.read_text(encoding="utf-8").strip())
        ):
            reasons.append("official.pid")

        launch_record = load_launch_record(case_id)
        if (
            not wrapper_yieldable
            and isinstance(launch_record, dict)
            and pid_alive(launch_record.get("launcher_pid"))
            and (not back_half_launch_record or active_state is not None)
        ):
            reasons.append("launcher_pid from scheduler launch record")

        if reasons:
            active[case_id] = sorted(set(reasons))
            continue

        if isinstance(status, dict) and bool(status.get("official_finished")):
            continue
    return active


def assigned_gpu_ids(
    queue: list[dict[str, str]],
    *,
    active_back_half: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    assigned: dict[str, str] = {}
    active_back_half = active_back_half or {}
    now = time.time()
    for item in queue:
        case_id = item["case_id"]
        launch_record = load_launch_record(case_id)
        gpu_id = str(launch_record.get("gpu_id") or "").strip() if isinstance(launch_record, dict) else ""
        if not gpu_id:
            back_half = active_back_half.get(case_id)
            if isinstance(back_half, dict):
                gpu_id = str(back_half.get("gpu_id") or "").strip()
        if not gpu_id:
            continue

        status = load_json(case_root(case_id) / "monitoring" / "supervisor_status.json")
        pid_file = case_root(case_id) / "monitoring" / "official.pid"
        active = False
        gpu_required = True
        back_half = active_back_half.get(case_id)
        active_state, _ = active_back_half_state_details(case_id, now=now)
        back_half_launch_record = launch_record_is_back_half_rerun(case_id)
        supervisor_counts_as_live_rollout = supervisor_pids_should_count_as_live_rollout(case_id, now=now)
        if isinstance(back_half, dict) and pid_alive(back_half.get("pid")):
            active = True
            gpu_required = str(back_half.get("kind") or "").strip() == "reproduction_grading"
        if isinstance(status, dict):
            if supervisor_counts_as_live_rollout and pid_alive(status.get("official_wrapper_pid")):
                active = True
                rerun_plan = back_half_rerun_plan(case_id)
                if (
                    str(rerun_plan.get("kind") or "").strip() == "grading_only"
                    and bool(rerun_plan.get("reproduction_succeeded"))
                ):
                    gpu_required = False
        if (
            supervisor_counts_as_live_rollout
            and pid_file.exists()
            and pid_alive(pid_file.read_text(encoding="utf-8").strip())
        ):
            active = True
            rerun_plan = back_half_rerun_plan(case_id)
            if (
                str(rerun_plan.get("kind") or "").strip() == "grading_only"
                and bool(rerun_plan.get("reproduction_succeeded"))
            ):
                gpu_required = False
        if (
            isinstance(launch_record, dict)
            and pid_alive(launch_record.get("launcher_pid"))
            and (not back_half_launch_record or active_state is not None)
        ):
            active = True
        if isinstance(status, dict) and bool(status.get("official_finished")) and not active:
            active = False
        if active and gpu_required:
            assigned[case_id] = gpu_id
    return assigned


def gpu_slot_usage(active_assignments: dict[str, str]) -> dict[str, int]:
    usage: dict[str, int] = {gpu_id: 0 for gpu_id in DEFAULT_GPU_IDS}
    for gpu_id in active_assignments.values():
        usage[gpu_id] = usage.get(gpu_id, 0) + 1
    return usage


def require_allowed_gpu_id(gpu_id: str, context: str) -> str:
    if gpu_id not in ALLOWED_GPU_IDS:
        raise RuntimeError(
            f"{context} requested GPU {gpu_id!r}; PaperBench is restricted to GPUs 4-7"
        )
    return gpu_id


def has_free_gpu_slot(gpu_usage: dict[str, int], max_cases_per_gpu: int) -> bool:
    return any(gpu_usage.get(gpu_id, 0) < max_cases_per_gpu for gpu_id in DEFAULT_GPU_IDS)


def choose_gpu_id(
    case_id: str,
    gpu_usage: dict[str, int],
    active_assignments: dict[str, str],
    max_cases_per_gpu: int,
) -> str:
    if case_id in active_assignments:
        return require_allowed_gpu_id(active_assignments[case_id], f"{case_id} active assignment")
    continuation = load_continuation_state(case_id)
    continuation_gpu = str(continuation.get("gpu_id") or "").strip()
    if (
        continuation_gpu
        and continuation_gpu in ALLOWED_GPU_IDS
        and gpu_usage.get(continuation_gpu, 0) < max_cases_per_gpu
    ):
        return continuation_gpu
    for gpu_id in DEFAULT_GPU_IDS:
        if gpu_usage.get(gpu_id, 0) < max_cases_per_gpu:
            return gpu_id
    raise RuntimeError(
        f"no free GPU slots available for {case_id}; gpu_usage={gpu_usage}, max_cases_per_gpu={max_cases_per_gpu}"
    )


def resolve_run_root(case_id: str, paper_id: str) -> str:
    return f"sota/cases/{case_id}/official_runs/{stamp()}_{paper_id}"


def launch_back_half_rerun(
    item: dict[str, str],
    *,
    dry_run: bool,
    kind: str,
    gpu_id: str = "",
) -> dict[str, Any]:
    case_id = item["case_id"]
    paper_id = item["paper_id"]
    latest_run = latest_run_root(case_id)
    run_root = str(latest_run.relative_to(ROOT)) if latest_run is not None else ""
    script_name = (
        "rerun_reproduction_and_grading.py" if kind == "reproduction_grading" else "rerun_grading_only.py"
    )
    module_name = (
        "sota.scripts.rerun_reproduction_and_grading"
        if kind == "reproduction_grading"
        else "sota.scripts.rerun_grading_only"
    )
    launch_log = (
        case_root(case_id)
        / "monitoring"
        / f"official_scheduler_{case_id}_{kind}_{stamp()}.log"
    )
    record: dict[str, Any] = {
        "case_id": case_id,
        "paper_id": paper_id,
        "mode": f"{kind}_rerun",
        "script": f"sota/scripts/{script_name}",
        "run_root": run_root,
        "launch_log": str(launch_log.relative_to(ROOT)),
        "gpu_id": gpu_id,
    }
    if kind == "reproduction_grading":
        require_allowed_gpu_id(gpu_id, f"{case_id} reproduction_grading rerun")
    if dry_run:
        record["launcher_pid"] = 0
        return record

    if kind == "reproduction_grading":
        inner_tokens = [
            ".venv/bin/python",
            "-m",
            module_name,
            "--case",
            case_id,
        ]
        if gpu_id:
            inner_tokens.extend(["--gpu-id", gpu_id])
    else:
        inner_tokens = [
            ".venv/bin/python",
            "-m",
            module_name,
            "--case",
            case_id,
        ]
    inner_cmd = " ".join(shlex.quote(token) for token in inner_tokens)
    cmd = (
        "set -euo pipefail && "
        "cd /share/project/yuyang/workspace/Paperbench && "
        "if [[ -f /share/project/yuyang/workspace/setvpn.sh ]]; then "
        "  source /share/project/yuyang/workspace/setvpn.sh >/dev/null 2>&1; "
        "fi && "
        "set -a && "
        "[[ -f .env ]] && source .env; "
        "[[ -f paperbench/solvers/agent.env ]] && source paperbench/solvers/agent.env; "
        "[[ -f sota/.env ]] && source sota/.env; "
        "set +a && "
        "export "
        "JUDGE_BASE_URL=\"${JUDGE_BASE_URL:-https://cn2.su8.codes/v1}\" "
        "PAPERBENCH_GRADE_RERUN_MAX_ATTEMPTS=\"${PAPERBENCH_GRADE_RERUN_MAX_ATTEMPTS:-100}\" "
        "PAPERBENCH_GRADE_RERUN_RETRY_SLEEP_SECONDS=\"${PAPERBENCH_GRADE_RERUN_RETRY_SLEEP_SECONDS:-20}\" "
        "PAPERBENCH_REPRO_RERUN_MAX_ATTEMPTS=\"${PAPERBENCH_REPRO_RERUN_MAX_ATTEMPTS:-100}\" "
        "PAPERBENCH_REPRO_RERUN_RETRY_SLEEP_SECONDS=\"${PAPERBENCH_REPRO_RERUN_RETRY_SLEEP_SECONDS:-20}\" "
        "PAPERBENCH_SIMPLE_JUDGE_LEAF_CONCURRENCY=\"${PAPERBENCH_SIMPLE_JUDGE_LEAF_CONCURRENCY:-1}\" && "
        f"exec sg docker -c {shlex.quote(inner_cmd)}"
    )

    launch_log.parent.mkdir(parents=True, exist_ok=True)
    with launch_log.open("ab") as handle:
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=str(ROOT),
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    record["launcher_pid"] = proc.pid
    record["launched_at_utc"] = utc_now()
    write_json(
        back_half_rerun_state_path(case_id),
        {
            "case_id": case_id,
            "phase": "launching",
            "kind": kind,
            "pid": proc.pid,
            "gpu_id": gpu_id,
            "run_root": run_root,
            "last_updated_utc": utc_now(),
        },
    )
    write_json(launch_record_path(case_id), record)
    event("official_case_back_half_rerun_launched", record)
    return record


def launch_case(item: dict[str, str], dry_run: bool, gpu_id: str) -> dict[str, Any]:
    case_id = item["case_id"]
    expected_paper_id = item["paper_id"]
    meta = metadata(case_id)
    actual_paper_id = str(meta.get("paper_id") or "").strip()
    if actual_paper_id != expected_paper_id:
        raise RuntimeError(
            f"{case_id} metadata mismatch: expected {expected_paper_id!r}, got {actual_paper_id!r}"
        )

    wrapper = ROOT / "sota" / "scripts" / "run_official_sota.sh"
    if not wrapper.exists():
        raise RuntimeError(f"missing official wrapper: {wrapper}")
    chosen_prompt_mode = prompt_mode(meta)
    chosen_skills_dir = skills_dir(meta, case_id)
    skills_path = Path(chosen_skills_dir)
    if not skills_path.is_absolute():
        skills_path = ROOT / chosen_skills_dir
    if not skills_path.exists():
        raise RuntimeError(f"missing skills directory for {case_id}: {skills_path}")
    require_allowed_gpu_id(gpu_id, f"{case_id} official rollout")

    run_root = resolve_run_root(case_id, expected_paper_id)
    launch_log = (
        case_root(case_id)
        / "monitoring"
        / f"official_scheduler_{expected_paper_id}_{stamp()}.log"
    )
    record = {
        "case_id": case_id,
        "paper_id": expected_paper_id,
        "wrapper": str(wrapper.relative_to(ROOT)),
        "prompt_mode": chosen_prompt_mode,
        "skills_dir": chosen_skills_dir,
        "run_root": run_root,
        "launch_log": str(launch_log.relative_to(ROOT)),
        "gpu_id": gpu_id,
    }

    if dry_run:
        record["launcher_pid"] = 0
        return record

    env = os.environ.copy()
    env["CASE_ID"] = case_id
    env["CASE_SPLIT"] = case_id
    env["PROMPT_MODE"] = chosen_prompt_mode
    env["SKILLS_DIR"] = chosen_skills_dir
    env["RUN_ROOT"] = run_root
    env["OFFICIAL_RUN_MODE"] = "sota_direct"
    env["CASE_GPU_ID"] = gpu_id
    refresh_continuation_state_checkpoint(case_id, dry_run=dry_run)
    continuation = load_continuation_state(case_id)
    if continuation.get("case_id") == case_id and continuation_state_path(case_id).exists():
        env["PAPERBENCH_CONTINUATION_STATE"] = str(continuation_state_path(case_id))
        record["continuation_state"] = str(continuation_state_path(case_id).relative_to(ROOT))

    launch_log.parent.mkdir(parents=True, exist_ok=True)
    with launch_log.open("ab") as handle:
        proc = subprocess.Popen(
            ["bash", str(wrapper.relative_to(ROOT))],
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    record["launcher_pid"] = proc.pid
    record["launched_at_utc"] = utc_now()
    write_json(launch_record_path(case_id), record)
    event("official_case_launched", record)
    return record


def scheduler_once(
    max_active_cases: int,
    launches_per_tick: int,
    dry_run: bool,
    *,
    max_cases_per_gpu: int,
    health_stall_seconds: int,
    health_min_run_age_before_stall_seconds: int,
) -> dict[str, Any]:
    queue = parse_queue()
    active_back_half = active_back_half_processes()
    reconciled_statuses = []
    for item in queue:
        payload = reconcile_supervisor_status(item["case_id"], dry_run=dry_run)
        if payload:
            reconciled_statuses.append(payload)
    health_findings = run_health_check(
        queue,
        stall_seconds=health_stall_seconds,
        min_run_age_before_stall_seconds=health_min_run_age_before_stall_seconds,
        dry_run=dry_run,
        active_back_half=active_back_half,
    )
    active_back_half = active_back_half_processes()
    active = active_cases(queue, active_back_half=active_back_half)
    reclaimable_active_cases = {
        str(finding.get("case_id") or "").strip()
        for finding in health_findings
        if isinstance(finding, dict) and finding.get("stop_requested")
    }
    effective_active = {
        case_id: reasons
        for case_id, reasons in active.items()
        if case_id not in reclaimable_active_cases
    }
    active_gpu_assignments = assigned_gpu_ids(queue, active_back_half=active_back_half)
    gpu_usage = gpu_slot_usage(active_gpu_assignments)
    total_active_case_count = len(set(effective_active))
    gpu_bound_active_case_count = len(active_gpu_assignments)
    reproduction_grading_active_cases = sorted(
        case_id
        for case_id, payload in active_back_half.items()
        if str(payload.get("kind") or "").strip() == "reproduction_grading"
    )
    grading_only_active_cases = sorted(
        case_id
        for case_id, payload in active_back_half.items()
        if str(payload.get("kind") or "").strip() == "grading_only"
    )
    official_wrapper_only_active_cases = sorted(
        case_id
        for case_id in active
        if case_id not in active_back_half
    )
    deferred_cases = load_deferred_cases()
    status: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "queue_length": len(queue),
        "max_active_cases": max_active_cases,
        "max_total_active_cases": max_active_cases,
        "max_gpu_bound_cases": max_active_cases,
        "launches_per_tick": launches_per_tick,
        "max_cases_per_gpu": max_cases_per_gpu,
        "dry_run": dry_run,
        "active_case_count": len(set(effective_active)),
        "total_active_case_count": total_active_case_count,
        "gpu_bound_active_case_count": gpu_bound_active_case_count,
        "reproduction_grading_active_case_count": len(reproduction_grading_active_cases),
        "grading_only_active_case_count": len(grading_only_active_cases),
        "official_wrapper_only_active_case_count": len(official_wrapper_only_active_cases),
        "active_cases": {case_id: reasons for case_id, reasons in sorted(effective_active.items())},
        "reproduction_grading_active_cases": reproduction_grading_active_cases,
        "grading_only_active_cases": grading_only_active_cases,
        "official_wrapper_only_active_cases": official_wrapper_only_active_cases,
        "gpu_usage": gpu_usage,
        "launched": [],
        "cleaned_for_rerun": [],
        "reconciled_statuses": reconciled_statuses,
        "health_findings": health_findings,
        "skipped_already_launched": [],
        "deferred_cases": sorted(deferred_cases),
        "waiting": False,
    }

    # `max_active_cases` is the ceiling for total active official PaperBench work
    # (rollout wrappers plus back-half reruns). GPU-bound work additionally requires
    # a free GPU slot, but grading-only reruns still consume judge/container/network
    # capacity and must count toward the same overall cap.
    launch_budget = launches_per_tick
    queued_items: list[tuple[dict[str, str], dict[str, Any]]] = []
    for item in queue:
        queued_items.append((item, back_half_rerun_plan(item["case_id"])))
    queued_items.sort(key=lambda pair: launch_priority(pair[1].get("kind")))
    grading_only_phase_open = bool(grading_only_active_cases)
    for item, plan in queued_items:
        case_id = item["case_id"]
        if case_id in effective_active or case_id in deferred_cases:
            continue
        if str(plan.get("kind") or "").strip() != "grading_only":
            continue
        finished_state = back_half_finished_state(case_id, "grading_only")
        if not finished_state or back_half_finished_state_retryable(case_id, "grading_only"):
            grading_only_phase_open = True
            break
    status["grading_only_phase_open"] = grading_only_phase_open

    for item, rerun_plan in queued_items:
        case_id = item["case_id"]
        paper_id = item["paper_id"]
        if case_id in effective_active:
            continue
        if case_id in deferred_cases:
            continue
        if rerun_plan.get("kind"):
            cleanup = None
        else:
            cleanup = clean_failed_outputs_for_rerun(case_id, dry_run=dry_run)
        if cleanup:
            status["cleaned_for_rerun"].append(cleanup)
        if launch_budget <= 0:
            continue
        if rerun_plan.get("kind"):
            rerun_kind = str(rerun_plan["kind"])
            if rerun_kind == "reproduction_grading" and grading_only_phase_open:
                status.setdefault("blocked_back_half_launches", []).append(
                    {
                        "case_id": case_id,
                        "kind": rerun_kind,
                        "reason": "waiting_for_grading_only_phase_to_finish",
                        "grading_only_active_cases": grading_only_active_cases,
                    }
                )
                continue
            if total_active_case_count >= max_active_cases:
                status.setdefault("blocked_back_half_launches", []).append(
                    {
                        "case_id": case_id,
                        "kind": rerun_kind,
                        "reason": "no_free_total_case_slot",
                        "total_active_case_count": total_active_case_count,
                        "max_total_active_cases": max_active_cases,
                    }
                )
                continue
            if rerun_kind == "reproduction_grading" and gpu_bound_active_case_count >= max_active_cases:
                status.setdefault("blocked_back_half_launches", []).append(
                    {
                        "case_id": case_id,
                        "kind": rerun_kind,
                        "reason": "no_free_gpu_bound_case_slot",
                        "gpu_bound_active_case_count": gpu_bound_active_case_count,
                        "max_gpu_bound_cases": max_active_cases,
                    }
                )
                continue
            finished_state = back_half_finished_state(case_id, rerun_kind)
            if finished_state:
                retryable_finished_state = back_half_finished_state_retryable(case_id, rerun_kind)
                if retryable_finished_state:
                    status.setdefault("retryable_finished_back_half", []).append(
                        {
                            "case_id": case_id,
                            "kind": rerun_kind,
                            "reason": "previous_back_half_rerun_finished_but_retryable",
                            "last_updated_utc": retryable_finished_state.get("last_updated_utc"),
                            "score": retryable_finished_state.get("score"),
                            "judge_success": retryable_finished_state.get("judge_success"),
                            "reproduction_succeeded": retryable_finished_state.get("reproduction_succeeded"),
                            "repro_exit_code": retryable_finished_state.get("repro_exit_code"),
                            "transient_network_failure": retryable_finished_state.get("transient_network_failure"),
                        }
                    )
                else:
                    status.setdefault("blocked_back_half_launches", []).append(
                        {
                            "case_id": case_id,
                            "kind": rerun_kind,
                            "reason": "previous_back_half_rerun_finished",
                            "last_updated_utc": finished_state.get("last_updated_utc"),
                            "score": finished_state.get("score"),
                            "judge_success": finished_state.get("judge_success"),
                            "reproduction_succeeded": finished_state.get("reproduction_succeeded"),
                            "repro_exit_code": finished_state.get("repro_exit_code"),
                            "transient_network_failure": finished_state.get("transient_network_failure"),
                        }
                    )
                    continue
            live_rollout_pids = live_rollout_pids_for_case(case_id)
            if live_rollout_pids and wrapper_can_yield_to_back_half(case_id, rerun_plan):
                if dry_run:
                    status.setdefault("stopped_wrappers_for_back_half", []).append(
                        {
                            "case_id": case_id,
                            "kind": rerun_kind,
                            "dry_run": True,
                            "pid_candidates": live_rollout_pids,
                        }
                    )
                    live_rollout_pids = []
                else:
                    stop_result = stop_rollout_processes(case_id)
                    status.setdefault("stopped_wrappers_for_back_half", []).append(
                        {
                            "case_id": case_id,
                            "kind": rerun_kind,
                            "dry_run": False,
                            "stop_result": stop_result,
                        }
                    )
                    live_rollout_pids = live_rollout_pids_for_case(case_id)
            if live_rollout_pids:
                status.setdefault("blocked_back_half_launches", []).append(
                    {
                        "case_id": case_id,
                        "kind": rerun_kind,
                        "live_rollout_pids": live_rollout_pids,
                    }
                )
                continue
            rerun_gpu_id = ""
            if rerun_kind == "reproduction_grading":
                preferred_gpu = ""
                status_payload = load_json(supervisor_status_path(case_id))
                if isinstance(status_payload, dict):
                    preferred_gpu = str(status_payload.get("official_gpu_id") or "").strip()
                if gpu_bound_active_case_count >= max_active_cases or not has_free_gpu_slot(
                    gpu_usage, max_cases_per_gpu
                ):
                    status.setdefault("blocked_back_half_launches", []).append(
                        {
                            "case_id": case_id,
                            "kind": rerun_kind,
                            "reason": "no_free_gpu_slot_for_reproduction_grading",
                            "gpu_bound_active_case_count": gpu_bound_active_case_count,
                            "max_gpu_bound_cases": max_active_cases,
                            "gpu_usage": dict(gpu_usage),
                        }
                    )
                    continue
                if (
                    preferred_gpu
                    and preferred_gpu in ALLOWED_GPU_IDS
                    and gpu_usage.get(preferred_gpu, 0) < max_cases_per_gpu
                ):
                    rerun_gpu_id = preferred_gpu
                else:
                    rerun_gpu_id = choose_gpu_id(case_id, gpu_usage, active_gpu_assignments, max_cases_per_gpu)
                active_gpu_assignments[case_id] = rerun_gpu_id
                gpu_usage[rerun_gpu_id] = gpu_usage.get(rerun_gpu_id, 0) + 1
                gpu_bound_active_case_count += 1
            status["launched"].append(
                launch_back_half_rerun(item, dry_run=dry_run, kind=rerun_kind, gpu_id=rerun_gpu_id)
            )
            active[case_id] = [f"{rerun_kind}_rerun_launched_this_tick"]
            total_active_case_count += 1
            if len(status["launched"]) >= launch_budget:
                break
            continue
        if already_launched_for(case_id, paper_id):
            status["skipped_already_launched"].append(case_id)
            continue
        if total_active_case_count >= max_active_cases:
            status.setdefault("blocked_rollout_launches", []).append(
                {
                    "case_id": case_id,
                    "reason": "no_free_total_case_slot_for_official_rollout",
                    "total_active_case_count": total_active_case_count,
                    "max_total_active_cases": max_active_cases,
                }
            )
            continue
        if gpu_bound_active_case_count >= max_active_cases or not has_free_gpu_slot(
            gpu_usage, max_cases_per_gpu
        ):
            status.setdefault("blocked_rollout_launches", []).append(
                {
                    "case_id": case_id,
                    "reason": "no_free_gpu_bound_case_slot_for_official_rollout",
                    "gpu_bound_active_case_count": gpu_bound_active_case_count,
                    "max_gpu_bound_cases": max_active_cases,
                    "gpu_usage": dict(gpu_usage),
                }
            )
            continue
        gpu_id = choose_gpu_id(case_id, gpu_usage, active_gpu_assignments, max_cases_per_gpu)
        active_gpu_assignments[case_id] = gpu_id
        gpu_usage[gpu_id] = gpu_usage.get(gpu_id, 0) + 1
        gpu_bound_active_case_count += 1
        status["launched"].append(launch_case(item, dry_run=dry_run, gpu_id=gpu_id))
        total_active_case_count += 1
        active[case_id] = ["launched_this_tick"]
        if len(status["launched"]) >= launch_budget:
            break

    if not status["launched"] and not status["waiting"]:
        blocked_reasons: list[str] = []
        if status.get("blocked_back_half_launches"):
            blocked_reasons.append("gpu slots unavailable for reproduction recovery")
        if status.get("blocked_rollout_launches"):
            blocked_reasons.append("gpu slots unavailable for rollout launch")
        status["waiting"] = bool(blocked_reasons)
        status["reason"] = "; ".join(blocked_reasons) or "queue exhausted or all cases already launched"
    elif status.get("blocked_back_half_launches") or status.get("blocked_rollout_launches"):
        status["reason"] = "some launches blocked on GPU slot availability"
    write_json(STATE_ROOT / "status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one scheduler tick and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without launching.")
    parser.add_argument("--max-active-cases", type=int, default=DEFAULT_MAX_ACTIVE_CASES)
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--launches-per-tick", type=int, default=DEFAULT_LAUNCHES_PER_TICK)
    parser.add_argument("--max-cases-per-gpu", type=int, default=DEFAULT_MAX_CASES_PER_GPU)
    parser.add_argument("--health-stall-seconds", type=int, default=DEFAULT_HEALTH_STALL_SECONDS)
    parser.add_argument(
        "--health-min-run-age-before-stall-seconds",
        type=int,
        default=DEFAULT_MIN_RUN_AGE_BEFORE_STALL_SECONDS,
    )
    args = parser.parse_args()

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.once and not args.dry_run:
        (STATE_ROOT / "scheduler.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    event(
        "scheduler_started",
        {
            "pid": os.getpid(),
            "once": args.once,
            "dry_run": args.dry_run,
            "interval_seconds": args.interval_seconds,
            "max_active_cases": args.max_active_cases,
            "launches_per_tick": args.launches_per_tick,
            "max_cases_per_gpu": args.max_cases_per_gpu,
            "health_stall_seconds": args.health_stall_seconds,
            "health_min_run_age_before_stall_seconds": args.health_min_run_age_before_stall_seconds,
        },
    )

    while True:
        try:
            with SchedulerTickLock(SCHEDULER_TICK_LOCK):
                status = scheduler_once(
                    max_active_cases=args.max_active_cases,
                    launches_per_tick=args.launches_per_tick,
                    dry_run=args.dry_run,
                    max_cases_per_gpu=args.max_cases_per_gpu,
                    health_stall_seconds=args.health_stall_seconds,
                    health_min_run_age_before_stall_seconds=args.health_min_run_age_before_stall_seconds,
                )
            print(json.dumps(status, indent=2), flush=True)
        except Exception as exc:
            payload = {"timestamp_utc": utc_now(), "error": repr(exc)}
            write_json(STATE_ROOT / "last_error.json", payload)
            event("scheduler_error", payload)
            print(json.dumps(payload, indent=2), flush=True)
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
