from __future__ import annotations

import asyncio
import io
import json
import os
import re
import shlex
import tarfile
import time
from pathlib import Path
from typing import Any

import blobfile as bf
import chz
import structlog.stdlib
from dotenv import load_dotenv
from typing_extensions import override

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.constants import AGENT_DIR, AGENT_DIR_CONFIG, CODEX_HOME_DIR, LOGS_DIR, SUBMISSION_DIR
from paperbench.infra.alcatraz import extract_file_from_computer
from paperbench.nano.structs import AgentOutput
from paperbench.nano.task import PBTask
from paperbench.solvers.base import BasePBSolver
from paperbench.solvers.upload import upload_heavy_logs, upload_status
from paperbench.solvers.utils import check_for_existing_run
from paperbench.utils import build_canonical_sub_path, find_dotenv, get_root, get_timestamp

load_dotenv(find_dotenv())

logger = structlog.stdlib.get_logger(component=__name__)

LOOP_COMPLETION_PROMISE = "PAPERBENCH_COMPLETE"
PAPERBENCH_MODULE_ROOT = get_root()
REPO_ROOT = PAPERBENCH_MODULE_ROOT.parent
OFFICIAL_INSTRUCTIONS_PATH = PAPERBENCH_MODULE_ROOT / "instructions" / "instructions.txt"
RETRYABLE_FAILURE_TOKENS = [
    "selected model is at capacity",
    "rate limit",
    "too many requests",
    "429",
    "temporarily unavailable",
    "try a different model",
    "timeout",
    "stream disconnected before completion",
    "openai api error (500)",
    "error code 520",
    "520: web server is returning an unknown error",
    "web server is returning an unknown error",
    "internal_server_error",
    "internal_error",
    "server_error",
    "stream error",
    "error sending request for url",
    "transport error",
    "connection reset by peer",
    "connection refused",
    "connection aborted",
    "unexpected eof",
    " eof",
]

OFFICIAL_RUN_ROOT_DIR_NAMES = ("official_runs", "official_vanilla_runs")
SUBMISSION_FINALIZATION_FAILURE_EXIT_CODE = 65


CODEX_CONFIG_TEMPLATE = """model_provider = "su8"
model = "{model}"
network_access = "enabled"
disable_response_storage = true
model_verbosity = "high"
model_reasoning_effort = "xhigh"
service_tier = "default"

[model_providers.su8]
name = "su8"
base_url = "{base_url}"
wire_api = "responses"
requires_openai_auth = true

[projects."/home"]
trust_level = "trusted"

[projects."/home/submission"]
trust_level = "trusted"
"""


OFFICIAL_PAPERBENCH_INSTRUCTIONS = OFFICIAL_INSTRUCTIONS_PATH.read_text(encoding="utf-8").strip()
OFFICIAL_CODE_ONLY_INSTRUCTIONS_PATH = (
    PAPERBENCH_MODULE_ROOT / "instructions" / "code_only_instructions.txt"
)
OFFICIAL_CODE_ONLY_PAPERBENCH_INSTRUCTIONS = (
    OFFICIAL_CODE_ONLY_INSTRUCTIONS_PATH.read_text(encoding="utf-8").strip()
)


PROMPT_MODE_NOTES: dict[str, str] = {
    "official_skill_recovery_strict": (
        "CASE-SPECIFIC NOTE\n---\n"
        "Use the distilled skills as supporting prior work for this paper. They come from "
        "related and prerequisite papers and can help recover implementation details, training "
        "setup, evaluation protocol, baseline structure, and debugging tactics. The target is "
        "the official full PaperBench reproduction, not a reduced proxy.\n"
    ),
    "lca_on_the_line_skill_recovery": (
        "CASE-SPECIFIC NOTE\n---\n"
        "For this paper, a compact ImageNet-style proxy can at most be a clearly labeled "
        "fallback or sanity check. It is not a valid substitute for the full main-body "
        "reproduction over the paper's actual setting.\n"
    ),
    "sample_specific_masks_skill_recovery": (
        "CASE-SPECIFIC NOTE\n---\n"
        "If the paper specifies real datasets, training, masking, and baseline comparison, do "
        "that real pipeline. Do not replace it with a miniature skill demo.\n"
    ),
}


def _load_api_key(env_var: str) -> str:
    value = os.getenv(env_var) or os.getenv("OPENAI_API_KEY")
    if value:
        return value

    auth_path = Path.home() / ".codex" / "auth.json"
    if auth_path.exists():
        try:
            auth = json.loads(auth_path.read_text())
            value = auth.get("OPENAI_API_KEY")
            if isinstance(value, str) and value:
                return value
        except json.JSONDecodeError:
            pass

    raise RuntimeError(
        f"No API key found. Set {env_var} or OPENAI_API_KEY before running CodexAgentSolver."
    )


def _tar_directory_bytes(source: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(mode="w:gz", fileobj=buf) as tar:
        tar.add(source, arcname=source.name)
    return buf.getvalue()


def _tar_multiple_paths_bytes(sources: list[tuple[Path, str]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(mode="w:gz", fileobj=buf) as tar:
        for src, arcname in sources:
            if src.exists():
                tar.add(src, arcname=arcname)
    return buf.getvalue()


def _codex_home_archive_snippet(log_dir: str) -> str:
    return "\n".join(
        [
            "python3 - <<'PY'",
            "from pathlib import Path",
            "import tarfile",
            "home = Path('/home')",
            "src = home / 'codex_home'",
            f"out = Path('{log_dir}') / 'codex_home.tar.gz'",
            "if src.exists():",
            "    with tarfile.open(out, 'w:gz') as tar:",
            "        tar.add(src, arcname='codex_home')",
            "PY",
        ]
    )


def _normalize_archive_member_name(name: str) -> str:
    normalized = name.lstrip("/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _submission_payload_file_count(tar_path: Path) -> int:
    try:
        with tarfile.open(tar_path, "r:gz") as archive:
            count = 0
            for member in archive.getmembers():
                normalized = _normalize_archive_member_name(member.name)
                if not normalized.startswith("submission/"):
                    continue
                rel = normalized[len("submission/") :]
                if not rel or rel == ".git" or rel.startswith(".git/"):
                    continue
                if member.isfile() or member.islnk() or member.issym():
                    count += 1
            return count
    except Exception:
        return -1


def _run_root_for_submission_tar(tar_path: Path) -> Path | None:
    for parent in tar_path.parents:
        try:
            if parent.parent.name in OFFICIAL_RUN_ROOT_DIR_NAMES:
                return parent
        except Exception:
            continue
    return None


def _repo_relative_str(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)




@chz.chz
class CodexAgentSolver(BasePBSolver):
    """Runs the Codex CLI as the PaperBench rollout agent inside the task container."""

    codex_model: str = chz.field(default="gpt-5.5")
    codex_base_url: str = chz.field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://cn2.su8.codes/v1"))
    codex_api_key_env: str = chz.field(default="OPENAI_API_KEY")
    skills_dir: str = chz.field(default="sota/skills/sequential-neural-score-estimation/skill")
    time_limit: int | None = chz.field(default=6 * 3600)
    timeout_kill_after: int = chz.field(default=30)
    prompt_mode: str = chz.field(default="standard")
    continuation_enabled: bool = chz.field(default=False)
    loop_enabled: bool = chz.field(default=False)
    loop_max_iterations: int = chz.field(default=0)
    loop_completion_promise: str = chz.field(default=LOOP_COMPLETION_PROMISE)
    loop_consecutive_failure_limit: int = chz.field(default=6)
    single_run_retryable_failure_limit: int = chz.field(default=8)
    single_run_retry_backoff_seconds: int = chz.field(default=30)

    def _continuation_manifest(self) -> dict[str, object]:
        if not self.continuation_enabled:
            return {}
        path = Path(os.getenv("PAPERBENCH_CONTINUATION_STATE", "")).expanduser()
        if not path or str(path) == "." or not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}

        resolved = dict(payload)
        raw_checkpoint = str(resolved.get("previous_submission_checkpoint") or "").strip()
        current_checkpoint = None
        if raw_checkpoint:
            current_checkpoint = Path(raw_checkpoint)
            if not current_checkpoint.is_absolute():
                current_checkpoint = REPO_ROOT / current_checkpoint
        current_score = (
            _submission_payload_file_count(current_checkpoint)
            if current_checkpoint is not None and current_checkpoint.exists()
            else -1
        )

        if current_score <= 0:
            best_checkpoint = current_checkpoint if current_checkpoint and current_checkpoint.exists() else None
            best_score = current_score
            for official_runs_root in [
                path.parent.parent / name for name in OFFICIAL_RUN_ROOT_DIR_NAMES
            ]:
                if not official_runs_root.exists():
                    continue
                candidates = sorted(
                    official_runs_root.glob("**/submission.tar.gz"),
                    key=lambda candidate: (candidate.stat().st_mtime, str(candidate)),
                )
                for candidate in candidates:
                    score = _submission_payload_file_count(candidate)
                    if score > best_score:
                        best_checkpoint = candidate
                        best_score = score
            if best_checkpoint is not None and best_score > current_score:
                resolved["previous_submission_checkpoint"] = _repo_relative_str(best_checkpoint)
                best_run_root = _run_root_for_submission_tar(best_checkpoint)
                if best_run_root is not None:
                    resolved["run_root"] = _repo_relative_str(best_run_root)

        return resolved

    @staticmethod
    def _is_retryable_iteration_failure(iteration_payload: dict[str, Any]) -> bool:
        joined = "\n".join(
            [
                str(iteration_payload.get("last_message_excerpt") or ""),
                str(iteration_payload.get("iter_log_excerpt") or ""),
                str(iteration_payload.get("last_error_excerpt") or ""),
            ]
        ).lower()
        return any(token in joined for token in RETRYABLE_FAILURE_TOKENS)

    @override
    def shortname(self) -> str:
        return "codexagent"

    @staticmethod
    def _extract_promise_text(message: str) -> str:
        match = re.search(r"<promise>(.*?)</promise>", message or "", flags=re.DOTALL)
        if not match:
            return ""
        return " ".join(match.group(1).split())

    def _loop_contract(self, *, code_only: bool = False) -> str:
        reproduce_requirement = (
            "- `reproduce.sh` exists and is the canonical path that regenerates the work you "
            "expect the grader to credit."
            if not code_only
            else "- The codebase is ready for code-development grading, and you are not required "
            "to create `reproduce.sh` unless it is useful for the paper itself."
        )
        return (
            "ITERATIVE LOOP CONTRACT\n"
            "---\n"
            "You are running inside a Ralph-style persistent loop over the same PaperBench "
            "workspace. Each iteration starts in a fresh Codex session with the full prompt "
            "resent, but the files in the workspace, any existing git history, and the partial "
            "submission remain available across iterations. Continue from the persisted project "
            "state on disk rather than from prior chat memory.\n\n"
            f"Only when the submission is genuinely ready to hand off may you include exactly "
            f"`<promise>{self.loop_completion_promise}</promise>` in your final response for an "
            "iteration. You may emit that promise only if the statement is fully true. Do not "
            "emit a false promise to escape the loop.\n\n"
            "Before emitting the completion promise, all of the following should be true:\n"
            f"- `{SUBMISSION_DIR}` contains the intended repository state.\n"
            f"{reproduce_requirement}\n"
            "- The important in-scope method, training or evaluation pipeline, and major paper "
            "artifacts have been implemented as far as the time budget allows.\n"
            "- `README.md` explains what was reproduced, what remains missing, and how the "
            "repository maps onto the paper.\n"
            "- You are not relying on unstated manual steps for anything you expect to be graded.\n\n"
            "A compact, toy, reduced, proxy, or fallback submission is not eligible for the "
            "completion promise unless it is clearly labeled as such and it is genuinely the best "
            "faithful reproduction still possible under real blockers.\n\n"
            "If these conditions are not all true, do not emit the completion promise. End the "
            "current iteration normally and continue making progress when the loop resumes.\n"
        )

    def _prompt_instructions(self, code_only: bool) -> str:
        return (
            OFFICIAL_CODE_ONLY_PAPERBENCH_INSTRUCTIONS
            if code_only
            else OFFICIAL_PAPERBENCH_INSTRUCTIONS
        )

    def _additional_paperbench_note(self, code_only: bool) -> str:
        if code_only:
            return (
                "ADDITIONAL PAPERBENCH NOTE\n---\n"
                "Use the provided distilled skills when they help, but keep the final "
                "repository centered on the target paper itself. Align the method, datasets, "
                "preprocessing, training, baselines, ablations, tables, figures, and addendum "
                "requirements as faithfully as possible. This run is code-development only, so "
                "focus on a clean and complete implementation without inventing unnecessary GPU "
                "dependencies or a reproduce.sh path unless the paper itself genuinely needs "
                "them.\n\n"
            )
        return (
            "ADDITIONAL PAPERBENCH NOTE\n---\n"
            "Use the provided distilled skills when they help, but keep the final repository "
            "centered on the target paper itself. Align the method, datasets, preprocessing, "
            "training, baselines, ablations, tables, figures, and addendum requirements as "
            "faithfully as possible. Do not default to smoke, quick, toy, reduced, proxy, or "
            "CPU-only fallbacks when the paper specifies a concrete full setup. When GPU is "
            "needed for the paper's real training or evaluation pipeline, actually use the "
            "provided GPU, verify that your stack sees it, and wire `reproduce.sh` to run the "
            "real GPU-backed path rather than silently degrading to CPU.\n\n"
        )

    @staticmethod
    def _build_submission_finalization_status(
        *,
        raw_exit_code: int,
        reproduce_sh_ok: bool,
        reproduce_sh_message: str,
        loop_state: dict[str, Any],
        completion_promise: str,
        solver: str,
        code_only: bool = False,
    ) -> dict[str, Any]:
        effective_exit_code = int(raw_exit_code)
        reproduce_sh_required = not code_only
        effective_reproduce_sh_ok = bool(reproduce_sh_ok or not reproduce_sh_required)
        if effective_exit_code == 0 and reproduce_sh_required and not reproduce_sh_ok:
            effective_exit_code = SUBMISSION_FINALIZATION_FAILURE_EXIT_CODE
        elif effective_exit_code == 124 and effective_reproduce_sh_ok:
            effective_exit_code = 0
        raw_agent_completed = bool(loop_state.get("completed"))
        return {
            "solver": solver,
            "code_only": bool(code_only),
            "raw_agent_exit_code": int(raw_exit_code),
            "raw_agent_completed": raw_agent_completed,
            "effective_exit_code": effective_exit_code,
            "salvaged_timeout_with_reproduce_sh": bool(
                raw_exit_code == 124 and effective_reproduce_sh_ok
            ),
            "submission_reproduce_sh_required": reproduce_sh_required,
            "submission_reproduce_sh_ok": effective_reproduce_sh_ok,
            "submission_reproduce_sh_message": (
                "" if not reproduce_sh_required else reproduce_sh_message
            ),
            "iteration_count": int(loop_state.get("iteration_count") or 0),
            "completed": raw_agent_completed and effective_exit_code == 0,
            "stop_reason": str(loop_state.get("stop_reason") or ""),
            "completion_promise": completion_promise,
            "last_iteration": loop_state.get("last_iteration") or {},
        }

    @staticmethod
    def _submission_finalization_error(status: dict[str, Any]) -> str | None:
        if not bool(status.get("submission_reproduce_sh_required", True)):
            return None
        if not bool(status.get("submission_reproduce_sh_ok")):
            return str(
                status.get("submission_reproduce_sh_message")
                or "submission/reproduce.sh not found"
            )
        effective_exit_code = int(status.get("effective_exit_code") or 0)
        if effective_exit_code != 0:
            agent_name = "Pi" if str(status.get("solver") or "").startswith("piagent") else "Codex"
            return f"{agent_name} command exited with {effective_exit_code}"
        return None

    @staticmethod
    def _finalized_effective_exit_code(
        finalization_status: dict[str, Any],
        fallback_exit_code: int,
    ) -> int:
        effective_exit_code = finalization_status.get("effective_exit_code")
        if effective_exit_code is None:
            return int(fallback_exit_code)
        return int(effective_exit_code)

    @staticmethod
    def _codex_exit_status_text(status: dict[str, Any]) -> str:
        return (
            f"codex_exit={int(status.get('effective_exit_code') or 0)}\n"
            f"raw_agent_exit={int(status.get('raw_agent_exit_code') or 0)}\n"
            f"raw_agent_completed={1 if status.get('raw_agent_completed') else 0}\n"
            f"submission_reproduce_sh_required={1 if status.get('submission_reproduce_sh_required', True) else 0}\n"
            f"submission_reproduce_sh_ok={1 if status.get('submission_reproduce_sh_ok') else 0}\n"
            f"submission_reproduce_sh_message={status.get('submission_reproduce_sh_message') or ''}\n"
            f"completion_promise={status.get('completion_promise') or ''}\n"
            f"completed={1 if status.get('completed') else 0}\n"
            f"iteration_count={int(status.get('iteration_count') or 0)}\n"
            f"stop_reason={status.get('stop_reason') or ''}\n"
        )

    async def _check_remote_submission_reproduce_sh(
        self, computer: ComputerInterface, *, required: bool = True
    ) -> tuple[bool, str]:
        if not required:
            return True, ""
        script = "\n".join(
            [
                "set +e",
                f"cd {shlex.quote(SUBMISSION_DIR)} 2>/dev/null",
                "if [ $? -ne 0 ]; then echo 'submission directory missing'; exit 1; fi",
                "if [ ! -f reproduce.sh ]; then echo 'submission/reproduce.sh not found'; exit 1; fi",
                "if [ ! -s reproduce.sh ]; then echo 'submission/reproduce.sh is empty'; exit 1; fi",
                "echo ok",
            ]
        )
        result = await computer.send_shell_command("bash -lc " + shlex.quote(script))
        message = result.unicode_output_best_effort.strip().splitlines()
        last_message = message[-1] if message else ""
        if result.exit_code == 0:
            return True, ""
        return False, last_message or "submission/reproduce.sh not found"

    def _single_run_contract(self) -> str:
        return (
            "EXECUTION MODE\n"
            "---\n"
            "This run is a single Codex execution, not a multi-iteration loop. Work within this "
            "one session to build the best complete PaperBench submission you can, leave the "
            "repository in a final submitted state in `/home/submission`, and do not rely on a "
            "follow-up Codex iteration to finish missing work.\n"
        )

    async def _load_remote_loop_resume_state(self, computer: ComputerInterface) -> dict[str, Any]:
        cmd = "bash -lc " + shlex.quote(
            "\n".join(
                [
                    "python3 - <<'PY'",
                    "import json",
                    "from pathlib import Path",
                    f"loop_path = Path('{LOGS_DIR}/codex_loop_state.json')",
                    "loop_payload = {}",
                    "if loop_path.exists():",
                    "    try:",
                    "        loop_payload = json.loads(loop_path.read_text(encoding='utf-8'))",
                    "    except Exception:",
                    "        loop_payload = {}",
                    "payload = {",
                    "    'iteration_count': int(loop_payload.get('iteration_count') or 0),",
                    "}",
                    "print(json.dumps(payload))",
                    "PY",
                ]
            )
        )
        result = await computer.send_shell_command(cmd)
        try:
            payload = json.loads(result.unicode_output_best_effort.strip() or "{}")
        except Exception:
            payload = {}
        return payload if isinstance(payload, dict) else {}

    async def _write_remote_json(
        self, computer: ComputerInterface, remote_path: str, payload: dict[str, Any]
    ) -> None:
        await computer.upload((json.dumps(payload, indent=2) + "\n").encode("utf-8"), remote_path)

    async def _run_codex_iteration(
        self,
        computer: ComputerInterface,
        prompt_path: str,
        iteration: int,
        time_limit_seconds: int | None,
    ) -> dict[str, Any]:
        loop_dir = f"{LOGS_DIR}/loop"
        iter_slug = f"iter_{iteration:04d}"
        iter_log_path = f"{loop_dir}/{iter_slug}.agent.log"
        last_message_path = f"{loop_dir}/{iter_slug}.last_message.txt"
        summary_path = f"{loop_dir}/{iter_slug}.summary.json"
        codex_exec = (
            "codex exec "
            "--skip-git-repo-check "
            "--dangerously-bypass-approvals-and-sandbox "
            "--model "
            f"{shlex.quote(self.codex_model)} "
            "--sandbox danger-full-access "
            "--cd /home/submission "
            "--add-dir /home "
            "--json "
            "--output-last-message "
            f"{shlex.quote(last_message_path)} "
            f"< {shlex.quote(prompt_path)} "
            f"> {shlex.quote(iter_log_path)} 2>&1"
        )

        if time_limit_seconds:
            codex_exec = (
                f"timeout --kill-after={int(self.timeout_kill_after)}s "
                f"{int(time_limit_seconds)}s {codex_exec}"
            )

        script = "\n".join(
            [
                "set -euo pipefail",
                f"export CODEX_HOME={CODEX_HOME_DIR}",
                "export HOME=/home",
                "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/conda/bin",
                f"mkdir -p {shlex.quote(loop_dir)}",
                f"rm -f {shlex.quote(last_message_path)}",
                f"cd {shlex.quote(SUBMISSION_DIR)}",
                "git init >/dev/null 2>&1 || true",
                "git config user.email paperbench-codex@example.invalid >/dev/null 2>&1 || true",
                "git config user.name 'PaperBench Codex' >/dev/null 2>&1 || true",
                "set +e",
                codex_exec,
                "codex_exit=$?",
                "set -e",
                f"cat {shlex.quote(iter_log_path)} >> {shlex.quote(LOGS_DIR)}/agent.log 2>/dev/null || true",
                "python3 - "
                f"{int(iteration)} "
                f"{shlex.quote(self.loop_completion_promise)} "
                f"{shlex.quote(last_message_path)} "
                f"{shlex.quote(iter_log_path)} "
                f"{shlex.quote(summary_path)} "
                "\"$codex_exit\" <<'PY'",
                "import json",
                "import re",
                "import sys",
                "from pathlib import Path",
                "iteration = int(sys.argv[1])",
                "completion_promise = sys.argv[2]",
                "last_message_path = Path(sys.argv[3])",
                "iter_log_path = Path(sys.argv[4])",
                "summary_path = Path(sys.argv[5])",
                "exit_code = int(sys.argv[6])",
                "last_message = last_message_path.read_text(encoding='utf-8', errors='replace') if last_message_path.exists() else ''",
                "iter_log = iter_log_path.read_text(encoding='utf-8', errors='replace') if iter_log_path.exists() else ''",
                "session_id = ''",
                "for pattern in [r'\"session_meta\",\"payload\":\\{\"id\":\"([^\"]+)\"', r'\"session_id\":\"([^\"]+)\"', r'\"thread_id\":\"([^\"]+)\"', r'\"thread.started\",\"thread_id\":\"([^\"]+)\"']:",
                "    match = re.search(pattern, iter_log)",
                "    if match:",
                "        session_id = match.group(1)",
                "        break",
                "promise_text = ''",
                "match = re.search(r'<promise>(.*?)</promise>', last_message, flags=re.DOTALL)",
                "if match:",
                "    promise_text = ' '.join(match.group(1).split())",
                "payload = {",
                "    'iteration': iteration,",
                "    'exit_code': exit_code,",
                "    'session_id': session_id,",
                "    'completion_promise': completion_promise,",
                "    'promise_text': promise_text,",
                "    'completed': bool(completion_promise) and promise_text == completion_promise,",
                "    'last_message_path': str(last_message_path),",
                "    'iter_log_path': str(iter_log_path),",
                "    'summary_path': str(summary_path),",
                "    'last_message_excerpt': last_message[-4000:],",
                "    'iter_log_excerpt': iter_log[-4000:],",
                "}",
                "summary_path.write_text(json.dumps(payload, indent=2) + '\\n', encoding='utf-8')",
                "print(json.dumps(payload))",
                "PY",
            ]
        )

        result = await computer.send_shell_command("bash -lc " + shlex.quote(script))
        text = result.unicode_output_best_effort.strip()
        try:
            payload = json.loads(text.splitlines()[-1]) if text else {}
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse Codex iteration summary for iteration {iteration}: {text[:2000]}"
            ) from exc
        if isinstance(payload, dict):
            await self._write_remote_json(
                computer,
                f"{LOGS_DIR}/codex_session_state.json",
                {
                    "iteration": iteration,
                    "session_id": str(payload.get("session_id") or ""),
                    "exit_code": int(payload.get("exit_code") or 0),
                    "updated_at_epoch": int(time.time()),
                },
            )
        return payload if isinstance(payload, dict) else {}

    async def _finalize_submission(
        self,
        computer: ComputerInterface,
        final_exit_code: int,
        loop_state: dict[str, Any],
        *,
        code_only: bool = False,
    ) -> dict[str, Any]:
        script = "\n".join(
            [
                "set +e",
                f"export CODEX_HOME={CODEX_HOME_DIR}",
                "export HOME=/home",
                "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/conda/bin",
                f"mkdir -p {shlex.quote(SUBMISSION_DIR)} {shlex.quote(LOGS_DIR)}",
                f"cd {shlex.quote(SUBMISSION_DIR)}",
                "git init >/dev/null 2>&1 || true",
                "git config user.email paperbench-codex@example.invalid >/dev/null 2>&1 || true",
                "git config user.name 'PaperBench Codex' >/dev/null 2>&1 || true",
                "rm -rf .paperbench_cache .pytest_cache .mypy_cache .ruff_cache .tox .nox .paperbench_venv .venv-paperbench .venv venv env node_modules",
                "find . -type d \\( -name __pycache__ -o -name .paperbench_venv -o -name .venv-paperbench -o -name .venv -o -name venv -o -name env -o -name node_modules \\) -prune -exec rm -rf {} +",
                "if [ -f .gitignore ]; then python3 - <<'PY'\nfrom pathlib import Path\np = Path('.gitignore')\nblocked = {'results/', 'results', '/results/', '/results', 'reproduce.log', 'reproduce.log.creation_time'}\nlines = p.read_text().splitlines()\nkept = [line for line in lines if line.strip() not in blocked]\np.write_text('\\n'.join(kept).rstrip() + ('\\n' if kept else ''))\nPY\nfi",
                "git add -A",
                "git diff --cached --quiet || git commit -m 'final paperbench submission'",
                _codex_home_archive_snippet(LOGS_DIR),
                "exit 0",
            ]
        )
        await computer.send_shell_command("bash -lc " + shlex.quote(script))
        reproduce_sh_ok, reproduce_sh_message = await self._check_remote_submission_reproduce_sh(
            computer, required=not code_only
        )
        finalization_status = self._build_submission_finalization_status(
            raw_exit_code=final_exit_code,
            reproduce_sh_ok=reproduce_sh_ok,
            reproduce_sh_message=reproduce_sh_message,
            loop_state=loop_state,
            completion_promise=self.loop_completion_promise,
            solver=self.shortname(),
            code_only=code_only,
        )
        await self._write_remote_json(
            computer,
            f"{LOGS_DIR}/codex_loop_state.json",
            {
                "iteration_count": int(loop_state.get("iteration_count") or 0),
                "completed": bool(finalization_status.get("completed")),
                "raw_agent_completed": bool(finalization_status.get("raw_agent_completed")),
                "stop_reason": str(loop_state.get("stop_reason") or ""),
                "completion_promise": self.loop_completion_promise,
                "last_iteration": loop_state.get("last_iteration") or {},
            },
        )
        await self._write_remote_json(
            computer,
            f"{LOGS_DIR}/submission_finalization.json",
            finalization_status,
        )
        await computer.upload(
            self._codex_exit_status_text(finalization_status).encode("utf-8"),
            f"{LOGS_DIR}/codex_exit_status.txt",
        )
        return finalization_status

    async def _finalize_after_supervisor_timeout(
        self,
        computer: ComputerInterface,
        *,
        final_iteration_payload: dict[str, Any],
        timeout_error_msg: str,
        code_only: bool = False,
    ) -> tuple[int, str | None, dict[str, Any] | None]:
        payload = dict(final_iteration_payload)
        payload["completed"] = False
        payload["stop_reason"] = payload.get("stop_reason") or "supervisor_timeout"
        finalization_status = await self._finalize_submission(
            computer, 124, payload, code_only=code_only
        )
        result_exit_code = self._finalized_effective_exit_code(finalization_status, 124)
        if result_exit_code == 0:
            await computer.upload(
                (
                    "Codex rollout supervisor timed out, but finalization found a valid "
                    "submission/reproduce.sh and salvaged the submission.\n\n"
                    + json.dumps(
                        {
                            "final_iteration": payload,
                            "submission_finalization": finalization_status,
                        },
                        indent=2,
                    )
                ).encode("utf-8", errors="replace"),
                f"{LOGS_DIR}/codex_solver_error.log",
            )
            return result_exit_code, None, finalization_status

        finalization_error = self._submission_finalization_error(finalization_status)
        error_msg = finalization_error or timeout_error_msg
        await computer.upload(
            (
                f"{error_msg}\n\n"
                + json.dumps(
                    {
                        "final_iteration": payload,
                        "submission_finalization": finalization_status,
                    },
                    indent=2,
                )
            ).encode("utf-8", errors="replace"),
            f"{LOGS_DIR}/codex_solver_error.log",
        )
        return result_exit_code, error_msg, finalization_status

    async def _save_submission_snapshot(
        self,
        computer: ComputerInterface,
        task: PBTask,
        *,
        timestamp: str,
    ) -> str:
        submission_path = build_canonical_sub_path(task.run_dir, timestamp)
        script = "\n".join(
            [
                "set -euo pipefail",
                f"mkdir -p /tmp/paperbench_intermediate/{shlex.quote(timestamp)}",
                f"rm -rf /tmp/paperbench_intermediate/{shlex.quote(timestamp)}/submission",
                f"cp -rp {shlex.quote(SUBMISSION_DIR)} /tmp/paperbench_intermediate/{shlex.quote(timestamp)}/submission",
                (
                    "find "
                    f"/tmp/paperbench_intermediate/{shlex.quote(timestamp)}"
                    " -type f -not -name 'agent.log' -not -name 'inspect.log' -size +10M -printf '%P\\n' > /tmp/exclude.txt"
                ),
                (
                    "tar -czf "
                    f"/tmp/{shlex.quote(timestamp)}_submission.tar.gz "
                    "-X /tmp/exclude.txt "
                    f"-C /tmp/paperbench_intermediate/{shlex.quote(timestamp)} ."
                ),
            ]
        )
        await computer.check_shell_command("bash -lc " + shlex.quote(script))
        archive_bytes = await computer.download(f"/tmp/{timestamp}_submission.tar.gz")
        parent = os.path.dirname(submission_path)
        if parent and not bf.exists(parent):
            bf.makedirs(parent)
        bf.write_bytes(submission_path, archive_bytes)
        await computer.send_shell_command(
            "bash -lc "
            + shlex.quote(
                f"rm -rf /tmp/paperbench_intermediate/{timestamp} /tmp/{timestamp}_submission.tar.gz /tmp/exclude.txt"
            )
        )
        return submission_path

    async def _sync_rollout_artifacts_to_host(self, computer: ComputerInterface, task: PBTask) -> dict[str, Any]:
        """Best-effort sync of critical rollout artifacts from the container to task.run_dir."""
        artifacts = {
            "agent_log_exists": False,
            "codex_exit_status_exists": False,
            "codex_exit_code": None,
        }
        async def _copy_if_present(remote_path: str, local_path: str) -> bool:
            return await self._copy_remote_file_to_host_if_present(
                computer,
                remote_path=remote_path,
                local_path=local_path,
                task=task,
            )

        agent_log_path = bf.join(task.run_dir, "agent.log")
        if await _copy_if_present(f"{LOGS_DIR}/agent.log", agent_log_path):
            artifacts["agent_log_exists"] = True

        codex_exit_host_path = bf.join(task.run_dir, "logs", "codex_exit_status.txt")
        if await _copy_if_present(f"{LOGS_DIR}/codex_exit_status.txt", codex_exit_host_path):
            artifacts["codex_exit_status_exists"] = True
            try:
                with bf.BlobFile(codex_exit_host_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("codex_exit="):
                            artifacts["codex_exit_code"] = int(line.split("=", 1)[1])
                            break
            except Exception:
                logger.exception(
                    "Failed to parse synced codex exit status",
                    run_group_id=task.run_group_id,
                    run_id=task.run_id,
                    runs_dir=task.runs_dir,
                )

        # These files are useful for outer-wrapper diagnosis and continuation.
        for remote_rel, host_rel in [
            ("codex_loop_state.json", bf.join(task.run_dir, "logs", "codex_loop_state.json")),
            (
                "submission_finalization.json",
                bf.join(task.run_dir, "logs", "submission_finalization.json"),
            ),
            ("codex_solver_error.log", bf.join(task.run_dir, "logs", "codex_solver_error.log")),
        ]:
            try:
                await _copy_if_present(f"{LOGS_DIR}/{remote_rel}", host_rel)
            except Exception:
                logger.exception(
                    "Best-effort rollout artifact sync failed",
                    artifact=remote_rel,
                    run_group_id=task.run_group_id,
                    run_id=task.run_id,
                    runs_dir=task.runs_dir,
                )

        return artifacts

    async def _copy_remote_file_to_host_if_present(
        self,
        computer: ComputerInterface,
        *,
        remote_path: str,
        local_path: str,
        task: PBTask,
    ) -> bool:
        result = await computer.send_shell_command(f"test -f {shlex.quote(remote_path)}")
        if result.exit_code != 0:
            return False
        await extract_file_from_computer(
            computer=computer,
            path_on_computer=Path(remote_path),
            extract_to=local_path,
            run_group_id=task.run_group_id,
            runs_dir=task.runs_dir,
            run_id=task.run_id,
        )
        return bf.exists(local_path)

    async def _copy_remote_directory_archive_to_host_if_present(
        self,
        computer: ComputerInterface,
        *,
        remote_dir: str,
        local_archive_path: str,
    ) -> bool:
        result = await computer.send_shell_command(f"test -d {shlex.quote(remote_dir)}")
        if result.exit_code != 0:
            return False

        remote_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", remote_dir.strip("/")) or "root"
        remote_archive = f"/tmp/paperbench_{remote_slug}.tar.gz"
        parent = str(Path(remote_dir).parent)
        name = Path(remote_dir).name

        await computer.check_shell_command(
            "bash -lc "
            + shlex.quote(
                f"set -euo pipefail; "
                f"rm -f {remote_archive}; "
                f"tar -czf {remote_archive} -C {shlex.quote(parent)} {shlex.quote(name)}"
            )
        )
        archive_bytes = await computer.download(remote_archive)
        local_parent = os.path.dirname(local_archive_path)
        if local_parent and not bf.exists(local_parent):
            bf.makedirs(local_parent)
        bf.write_bytes(local_archive_path, archive_bytes)
        await computer.send_shell_command(f"rm -f {shlex.quote(remote_archive)}")
        return True

    async def _record_intermediate_grade(
        self,
        computer: ComputerInterface,
        task: PBTask,
        *,
        iteration: int,
        agent_output: AgentOutput,
    ) -> dict[str, Any]:
        timestamp = f"{get_timestamp()}_step_{iteration:04d}"
        submission_path = await self._save_submission_snapshot(
            computer,
            task,
            timestamp=timestamp,
        )
        grade_output_path = bf.join(
            task.run_dir, "intermediate_grades", f"iteration_{iteration:04d}_grade.json"
        )
        grade = await task.grade_explicit_checkpoint(
            submission_path,
            grade_output_path=grade_output_path,
            agent_output=agent_output,
            require_rollout_success=False,
            run_monitor=False,
        )
        summary = {
            "iteration": iteration,
            "timestamp": timestamp,
            "submission_path": submission_path,
            "grade_path": grade_output_path,
            "score": grade.score,
            "judge_success": bool(
                grade.paperbench_result.judge_output
                and grade.paperbench_result.judge_output.success
            ),
            "reproduction_succeeded": bool(
                grade.paperbench_result.reproduction_metadata
                and grade.paperbench_result.reproduction_metadata.reproduction_succeeded()
            ),
        }
        index_path = bf.join(task.run_dir, "intermediate_grades", "index.json")
        existing: dict[str, Any] = {"evaluations": []}
        if bf.exists(index_path):
            with bf.BlobFile(index_path, "r") as f:
                try:
                    existing = json.loads(f.read())
                except Exception:
                    existing = {"evaluations": []}
        evaluations = existing.get("evaluations")
        if not isinstance(evaluations, list):
            evaluations = []
        evaluations = [item for item in evaluations if item.get("iteration") != iteration] + [summary]
        existing["evaluations"] = sorted(evaluations, key=lambda item: int(item["iteration"]))
        bf.write_bytes(index_path, json.dumps(existing, indent=2).encode("utf-8"))
        return summary

    async def _setup_codex_home(
        self,
        computer: ComputerInterface,
        *,
        include_skill_dirs: bool,
    ) -> None:
        mkdir_parts = [AGENT_DIR, LOGS_DIR, CODEX_HOME_DIR]
        if include_skill_dirs:
            mkdir_parts.extend([f"{CODEX_HOME_DIR}/skills", "/home/extracted_skills"])
        await computer.check_shell_command(f"mkdir -p {' '.join(mkdir_parts)}")

        api_key = _load_api_key(self.codex_api_key_env)
        auth_json = json.dumps({"OPENAI_API_KEY": api_key}, indent=2)
        config_toml = CODEX_CONFIG_TEMPLATE.format(
            model=self.codex_model,
            base_url=self.codex_base_url,
        )
        await computer.upload(auth_json.encode("utf-8"), f"{CODEX_HOME_DIR}/auth.json")
        await computer.upload(config_toml.encode("utf-8"), f"{CODEX_HOME_DIR}/config.toml")

    async def _install_distilled_skills(self, computer: ComputerInterface) -> None:
        skills_path = Path(self.skills_dir).resolve()
        if not skills_path.exists():
            raise FileNotFoundError(f"Codex skills directory does not exist: {skills_path}")
        await computer.upload(_tar_directory_bytes(skills_path), f"{AGENT_DIR}/extracted_skills.tar.gz")
        await computer.check_shell_command(
            "bash -lc "
            + shlex.quote(
                f"set -euo pipefail; "
                f"rm -rf /home/extracted_skills/* {CODEX_HOME_DIR}/skills/*; "
                f"tar -xzf {AGENT_DIR}/extracted_skills.tar.gz -C /home; "
                f"cp -R /home/skill/. /home/extracted_skills/; "
                f"cp -R /home/skill/. {CODEX_HOME_DIR}/skills/; "
                f"find {CODEX_HOME_DIR}/skills -name SKILL.md | sort > {LOGS_DIR}/installed_skills.txt"
            )
        )

    async def _restore_continuation_state(self, computer: ComputerInterface) -> None:
        continuation = self._continuation_manifest()
        checkpoint = str(continuation.get("previous_submission_checkpoint") or "").strip()
        continuation_run_root = str(continuation.get("run_root") or "").strip()
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.exists():
                await computer.upload(checkpoint_path.read_bytes(), f"{AGENT_DIR}/continuation_submission.tar.gz")
                await computer.check_shell_command(
                    "bash -lc "
                    + shlex.quote(
                        f"set -euo pipefail; "
                        f"mkdir -p /tmp/paperbench_continuation_extract; "
                        f"tar -xzf {AGENT_DIR}/continuation_submission.tar.gz -C /tmp/paperbench_continuation_extract; "
                        f"rm -rf {SUBMISSION_DIR}; "
                        f"cp -R /tmp/paperbench_continuation_extract/submission {SUBMISSION_DIR}; "
                        f"if [ -d /tmp/paperbench_continuation_extract/logs ]; then "
                        f"  mkdir -p {LOGS_DIR}; "
                        f"  cp -R /tmp/paperbench_continuation_extract/logs/. {LOGS_DIR}/; "
                        f"fi; "
                        f"if [ -f /tmp/paperbench_continuation_extract/logs/codex_home.tar.gz ]; then "
                        f"  rm -rf {CODEX_HOME_DIR}; "
                        f"  mkdir -p /home; "
                        f"  tar -xzf /tmp/paperbench_continuation_extract/logs/codex_home.tar.gz -C /home; "
                        f"fi; "
                        f"rm -rf /tmp/paperbench_continuation_extract"
                    )
                )

        if continuation_run_root:
            run_root_path = Path(continuation_run_root)
            if not run_root_path.is_absolute():
                run_root_path = REPO_ROOT / run_root_path
            latest_codex_home = sorted(
                [
                    *run_root_path.glob("runs/**/submissions/*/codex_home.tar.gz"),
                    *run_root_path.glob("runs/**/logs/codex_trajectory/codex_home.tar.gz"),
                    *run_root_path.glob("runs/**/logs/codex_trajectory/codex_home_live.tar.gz"),
                ],
                key=lambda candidate: (candidate.stat().st_mtime, str(candidate)),
            )
            if latest_codex_home:
                codex_home_tar = latest_codex_home[-1]
                await computer.upload(codex_home_tar.read_bytes(), f"{AGENT_DIR}/continuation_codex_home.tar.gz")
                await computer.check_shell_command(
                    "bash -lc "
                    + shlex.quote(
                        f"set -euo pipefail; "
                        f"rm -rf {CODEX_HOME_DIR}; "
                        f"mkdir -p {CODEX_HOME_DIR}; "
                        f"tar -xzf {AGENT_DIR}/continuation_codex_home.tar.gz -C /home"
                    )
                )

    async def _setup_computer(self, computer: ComputerInterface, task: PBTask) -> None:
        del task
        await self._setup_codex_home(computer, include_skill_dirs=True)
        await self._restore_continuation_state(computer)
        # A restored codex_home may contain the previous run's config.toml with
        # stale bridge ports or model settings. Re-write the current runtime
        # config after restore, then install the current skill set.
        await self._setup_codex_home(computer, include_skill_dirs=True)
        await self._install_distilled_skills(computer)


    def _prompt(self, task: PBTask) -> str:
        continuation = self._continuation_manifest()
        prefix = ""
        if continuation:
            pieces: list[str] = []
            reason = str(continuation.get("reason") or "").strip()
            if reason:
                pieces.append(f"Continuation reason: {reason}")
            last_run_root = str(continuation.get("run_root") or "").strip()
            if last_run_root:
                pieces.append(f"Previous official run root: {last_run_root}")
            previous_submission = continuation.get("previous_submission_checkpoint")
            if previous_submission:
                pieces.append(
                    "A previous submission checkpoint has already been restored into `/home/submission`. Continue improving that repository rather than restarting from scratch."
                )
                pieces.append("If you need prior context, inspect these container-local continuation logs if they exist:")
                for item in [
                    "/home/logs/agent.log",
                    "/home/logs/codex_loop_state.json",
                    "/home/logs/codex_solver_error.log",
                    "/home/logs/codex_exit_status.txt",
                    "/home/logs/loop",
                ]:
                    pieces.append(f"- {item}")
            if pieces:
                prefix = "CONTINUATION CONTEXT\n---\n" + "\n".join(pieces) + "\n\n"
        prompt_mode_note = PROMPT_MODE_NOTES.get(self.prompt_mode, "")
        if prompt_mode_note:
            prefix += prompt_mode_note + "\n"
        prefix += self._additional_paperbench_note(task.judge.code_only)
        mode_contract = (
            self._loop_contract(code_only=task.judge.code_only)
            if self.loop_enabled
            else self._single_run_contract()
        )

        return (
            f"/goal Reproduce the assigned PaperBench paper `{task.paper_id}` from the "
            f"materials in /home/paper and create a complete submission in "
            f"{SUBMISSION_DIR}.\n\n"
            f"{mode_contract}\n"
            f"{prefix}"
            f"{self._prompt_instructions(task.judge.code_only)}"
        )

    @staticmethod
    def _extract_session_id(agent_log_text: str) -> str:
        patterns = [
            r'"session_meta","payload":\{"id":"([^"]+)"',
            r'"session_id":"([^"]+)"',
            r'"thread_id":"([^"]+)"',
            r'"thread.started","thread_id":"([^"]+)"',
        ]
        for pattern in patterns:
            match = re.search(pattern, agent_log_text)
            if match:
                return match.group(1)
        return ""

    async def _run_agent(self, computer: ComputerInterface, task: PBTask) -> AgentOutput:
        agent_output = await check_for_existing_run(task)
        if agent_output:
            return agent_output

        start_time = time.time()
        await upload_status(int(start_time), task.run_dir, "running")

        prompt_path = f"{AGENT_DIR}/codex_prompt.txt"
        await computer.upload(self._prompt(task).encode("utf-8"), prompt_path)

        timed_out = False
        result_exit_code: int | None = None
        error_msg: str | None = None
        rollout_artifacts: dict[str, Any] = {
            "agent_log_exists": False,
            "codex_exit_status_exists": False,
            "codex_exit_code": None,
        }
        if not self.loop_enabled:
            final_iteration_payload: dict[str, Any] = {
                "completed": False,
                "iteration_count": 0,
                "stop_reason": "",
            }
            deadline = time.time() + float(self.time_limit) if self.time_limit is not None else None
            attempt = 1
            try:
                while True:
                    remaining: int | None
                    if deadline is not None:
                        remaining = int(deadline - time.time())
                        if remaining <= 0:
                            timed_out = True
                            result_exit_code = 124
                            final_iteration_payload = {
                                "completed": False,
                                "iteration_count": max(0, attempt - 1),
                                "stop_reason": "single_run_time_limit_reached",
                            }
                            break
                    else:
                        remaining = None

                    iteration_payload = await asyncio.wait_for(
                        self._run_codex_iteration(
                            computer=computer,
                            prompt_path=prompt_path,
                            iteration=attempt,
                            time_limit_seconds=remaining,
                        ),
                        timeout=(
                            remaining + min(180, max(30, remaining // 10))
                            if remaining is not None
                            else None
                        ),
                    ) if remaining is not None else await self._run_codex_iteration(
                        computer=computer,
                        prompt_path=prompt_path,
                        iteration=attempt,
                        time_limit_seconds=None,
                    )

                    iteration_exit_code = int(iteration_payload.get("exit_code") or 0)
                    submission_reproduce_sh_ok = False
                    submission_reproduce_sh_message = ""
                    if iteration_exit_code == 0:
                        (
                            submission_reproduce_sh_ok,
                            submission_reproduce_sh_message,
                        ) = await self._check_remote_submission_reproduce_sh(
                            computer, required=not task.judge.code_only
                        )
                        iteration_payload["submission_reproduce_sh_ok_after_iteration"] = (
                            submission_reproduce_sh_ok
                        )
                        iteration_payload["submission_reproduce_sh_message_after_iteration"] = (
                            submission_reproduce_sh_message
                        )
                    retryable_failure = self._is_retryable_iteration_failure(
                        iteration_payload
                    ) and (iteration_exit_code != 0 or not submission_reproduce_sh_ok)
                    timed_out = iteration_exit_code == 124
                    if iteration_exit_code == 0:
                        stop_reason = (
                            "single_run_finished"
                            if submission_reproduce_sh_ok
                            else "single_run_missing_reproduce_sh"
                        )
                    else:
                        stop_reason = (
                            "single_run_timeout"
                            if iteration_exit_code == 124
                            else "single_run_error"
                        )
                    final_iteration_payload = {
                        "completed": iteration_exit_code == 0
                        and submission_reproduce_sh_ok,
                        "iteration_count": attempt,
                        "stop_reason": stop_reason,
                        "last_iteration": iteration_payload,
                    }

                    if iteration_exit_code == 0 and submission_reproduce_sh_ok:
                        result_exit_code = 0
                        break

                    if retryable_failure:
                        remaining_after_failure = (
                            int(deadline - time.time()) if deadline is not None else None
                        )
                        enough_time_for_retry = (
                            remaining_after_failure is None
                            or remaining_after_failure
                            > max(60, self.single_run_retry_backoff_seconds + 30)
                        )
                        if (
                            attempt < self.single_run_retryable_failure_limit
                            and enough_time_for_retry
                        ):
                            final_iteration_payload["stop_reason"] = (
                                f"single_run_retryable_failure_retrying_{attempt}"
                            )
                            await self._write_remote_json(
                                computer,
                                f"{LOGS_DIR}/codex_loop_state.json",
                                {
                                    "iteration_count": attempt,
                                    "completed": False,
                                    "stop_reason": final_iteration_payload["stop_reason"],
                                    "completion_promise": self.loop_completion_promise,
                                    "last_iteration": iteration_payload,
                                },
                            )
                            if self.single_run_retry_backoff_seconds > 0:
                                await asyncio.sleep(self.single_run_retry_backoff_seconds)
                            attempt += 1
                            continue

                        final_iteration_payload["stop_reason"] = (
                            "single_run_retryable_failure_limit_reached"
                            if attempt >= self.single_run_retryable_failure_limit
                            else "single_run_retryable_failure_out_of_time"
                        )

                    result_exit_code = iteration_exit_code
                    break

                if result_exit_code is None:
                    result_exit_code = 0
                finalization_status = await self._finalize_submission(
                    computer,
                    result_exit_code,
                    final_iteration_payload,
                    code_only=task.judge.code_only,
                )
                result_exit_code = self._finalized_effective_exit_code(
                    finalization_status,
                    result_exit_code,
                )
                if result_exit_code == 0:
                    timed_out = False
                    error_msg = None
                finalization_error = self._submission_finalization_error(finalization_status)
                if result_exit_code != 0:
                    error_msg = (
                        f"Codex rollout timed out after {self.time_limit} seconds"
                        if timed_out
                        else finalization_error or f"Codex command exited with {result_exit_code}"
                    )
                    await computer.upload(
                        (
                            f"{error_msg}\n\n"
                            + json.dumps(
                                {
                                    "final_iteration": final_iteration_payload,
                                    "submission_finalization": finalization_status,
                                },
                                indent=2,
                            )
                        ).encode("utf-8", errors="replace"),
                        f"{LOGS_DIR}/codex_solver_error.log",
                    )
            except asyncio.TimeoutError:
                timed_out = True
                error_msg = f"Codex rollout supervisor timed out after {self.time_limit} seconds"
                try:
                    await asyncio.wait_for(
                        computer.send_shell_command(
                            "pkill -TERM -f 'codex exec' || true; "
                            "sleep 5; "
                            "pkill -KILL -f 'codex exec' || true"
                        ),
                        timeout=30,
                    )
                except Exception:
                    logger.exception(
                        "Best-effort Codex process cleanup failed after supervisor timeout",
                        run_group_id=task.run_group_id,
                        run_id=task.run_id,
                        runs_dir=task.runs_dir,
                    )
                await computer.upload(
                    f"Codex rollout timed out after {self.time_limit} seconds.\n".encode("utf-8"),
                    f"{LOGS_DIR}/codex_solver_error.log",
                )
                result_exit_code = 124
                try:
                    (
                        result_exit_code,
                        error_msg,
                        _,
                    ) = await self._finalize_after_supervisor_timeout(
                        computer,
                        final_iteration_payload=final_iteration_payload,
                        timeout_error_msg=error_msg,
                        code_only=task.judge.code_only,
                    )
                    if result_exit_code == 0:
                        timed_out = False
                except Exception:
                    logger.exception(
                        "Best-effort submission finalization failed after supervisor timeout",
                        run_group_id=task.run_group_id,
                        run_id=task.run_id,
                        runs_dir=task.runs_dir,
                    )

            status = "timeout" if timed_out else ("done" if result_exit_code == 0 else "failed")
            end_time = time.time()
            await upload_heavy_logs(
                computer=computer,
                agent_start_time=int(start_time),
                agent_dir_config=AGENT_DIR_CONFIG,
                run_dir=task.run_dir,
                run_group_id=task.run_group_id,
                runs_dir=task.runs_dir,
                run_id=task.run_id,
                runtime=end_time - start_time,
            )
            rollout_artifacts = await self._sync_rollout_artifacts_to_host(computer, task)
            if result_exit_code == 0 and not rollout_artifacts["agent_log_exists"]:
                error_msg = (
                    "Codex rollout finished without syncing /home/logs/agent.log to host run_dir"
                )
                status = "failed"
            await upload_status(int(start_time), task.run_dir, status, int(end_time))

            return AgentOutput(
                run_id=task.run_id,
                time_start=start_time,
                time_end=end_time,
                error_msg=error_msg,
                runtime_in_seconds=end_time - start_time,
                status_exists=bf.exists(bf.join(task.run_dir, "status.json")),
                agent_log_exists=bool(rollout_artifacts["agent_log_exists"]),
                codex_exit_status_exists=bool(rollout_artifacts["codex_exit_status_exists"]),
                codex_exit_code=rollout_artifacts["codex_exit_code"],
            )

        loop_state = await self._load_remote_loop_resume_state(computer)
        iteration_count = int(loop_state.get("iteration_count") or 0)
        start_iteration = max(1, iteration_count + 1)
        failure_streak = 0
        final_iteration_payload: dict[str, Any] = {
            "completed": False,
            "iteration_count": iteration_count,
            "stop_reason": "",
        }
        intermediate_grade_recorded = False
        try:
            deadline = time.time() + float(self.time_limit) if self.time_limit else None
            iteration = start_iteration
            while True:
                if self.loop_max_iterations > 0 and iteration > self.loop_max_iterations:
                    result_exit_code = 0
                    final_iteration_payload.update(
                        {
                            "completed": False,
                            "iteration_count": iteration - 1,
                            "stop_reason": f"max_iterations_reached_{self.loop_max_iterations}",
                        }
                    )
                    break

                if deadline is not None:
                    remaining = int(deadline - time.time())
                    if remaining <= 0:
                        timed_out = True
                        result_exit_code = 124
                        final_iteration_payload.update(
                            {
                                "completed": False,
                                "iteration_count": iteration - 1,
                                "stop_reason": "loop_time_limit_reached",
                            }
                        )
                        break
                    per_iter_limit = remaining
                else:
                    per_iter_limit = None

                await self._write_remote_json(
                    computer,
                    f"{LOGS_DIR}/codex_loop_state.json",
                    {
                        "iteration_count": iteration - 1,
                        "next_iteration": iteration,
                        "completion_promise": self.loop_completion_promise,
                        "time_limit_seconds": self.time_limit,
                    },
                )

                iteration_payload = await asyncio.wait_for(
                    self._run_codex_iteration(
                        computer=computer,
                        prompt_path=prompt_path,
                        iteration=iteration,
                        time_limit_seconds=per_iter_limit,
                    ),
                    timeout=per_iter_limit + min(180, max(30, per_iter_limit // 10)),
                ) if deadline is not None else await self._run_codex_iteration(
                    computer=computer,
                    prompt_path=prompt_path,
                    iteration=iteration,
                    time_limit_seconds=None,
                )

                iteration_exit = int(iteration_payload.get("exit_code") or 0)
                completed = bool(iteration_payload.get("completed"))
                final_iteration_payload = {
                    "completed": completed,
                    "iteration_count": iteration,
                    "stop_reason": "completion_promise" if completed else "",
                    "last_iteration": iteration_payload,
                }

                await self._write_remote_json(
                    computer,
                    f"{LOGS_DIR}/codex_loop_state.json",
                    {
                        "iteration_count": iteration,
                        "last_iteration_exit_code": iteration_exit,
                        "completion_promise": self.loop_completion_promise,
                        "completed": completed,
                        "last_message_path": iteration_payload.get("last_message_path", ""),
                        "iter_log_path": iteration_payload.get("iter_log_path", ""),
                    },
                )

                if iteration == 1 and not intermediate_grade_recorded:
                    live_agent_output = AgentOutput(
                        run_id=task.run_id,
                        time_start=start_time,
                        time_end=time.time(),
                        error_msg=None,
                        runtime_in_seconds=time.time() - start_time,
                        status_exists=bf.exists(bf.join(task.run_dir, "status.json")),
                    )
                    try:
                        await self._record_intermediate_grade(
                            computer=computer,
                            task=task,
                            iteration=iteration,
                            agent_output=live_agent_output,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to record intermediate grade after first iteration",
                            run_group_id=task.run_group_id,
                            run_id=task.run_id,
                            runs_dir=task.runs_dir,
                        )
                    intermediate_grade_recorded = True

                if completed:
                    result_exit_code = 0
                    break

                if iteration_exit == 124:
                    timed_out = True
                    result_exit_code = 124
                    final_iteration_payload["stop_reason"] = "iteration_timeout"
                    break

                if iteration_exit != 0:
                    if self._is_retryable_iteration_failure(iteration_payload):
                        failure_streak = 0
                    else:
                        failure_streak += 1
                    if failure_streak >= self.loop_consecutive_failure_limit:
                        result_exit_code = iteration_exit
                        final_iteration_payload["stop_reason"] = (
                            f"consecutive_failure_limit_{self.loop_consecutive_failure_limit}"
                        )
                        break
                else:
                    failure_streak = 0

                iteration += 1

            if result_exit_code is None:
                result_exit_code = 0

            finalization_status = await self._finalize_submission(
                computer,
                result_exit_code,
                final_iteration_payload,
                code_only=task.judge.code_only,
            )
            result_exit_code = self._finalized_effective_exit_code(
                finalization_status,
                result_exit_code,
            )
            if result_exit_code == 0:
                timed_out = False
                error_msg = None
            finalization_error = self._submission_finalization_error(finalization_status)

            if result_exit_code != 0:
                error_msg = (
                    f"Codex rollout timed out after {self.time_limit} seconds"
                    if timed_out or result_exit_code == 124
                    else finalization_error or f"Codex command exited with {result_exit_code}"
                )
                await computer.upload(
                    (
                        f"{error_msg}\n\n"
                        + json.dumps(
                            {
                                "final_iteration": final_iteration_payload,
                                "submission_finalization": finalization_status,
                            },
                            indent=2,
                        )
                    ).encode("utf-8", errors="replace"),
                    f"{LOGS_DIR}/codex_solver_error.log",
                )
        except asyncio.TimeoutError:
            timed_out = True
            error_msg = f"Codex rollout supervisor timed out after {self.time_limit} seconds"
            try:
                await asyncio.wait_for(
                    computer.send_shell_command(
                        "pkill -TERM -f 'codex exec' || true; "
                        "sleep 5; "
                        "pkill -KILL -f 'codex exec' || true"
                    ),
                    timeout=30,
                )
            except Exception:
                logger.exception(
                    "Best-effort Codex process cleanup failed after supervisor timeout",
                    run_group_id=task.run_group_id,
                    run_id=task.run_id,
                    runs_dir=task.runs_dir,
                )
            await computer.upload(
                f"Codex rollout timed out after {self.time_limit} seconds.\n".encode("utf-8"),
                f"{LOGS_DIR}/codex_solver_error.log",
            )
            result_exit_code = 124
            try:
                (
                    result_exit_code,
                    error_msg,
                    _,
                ) = await self._finalize_after_supervisor_timeout(
                    computer,
                    final_iteration_payload=final_iteration_payload,
                    timeout_error_msg=error_msg,
                    code_only=task.judge.code_only,
                )
                if result_exit_code == 0:
                    timed_out = False
            except Exception:
                logger.exception(
                    "Best-effort submission finalization failed after supervisor timeout",
                    run_group_id=task.run_group_id,
                    run_id=task.run_id,
                    runs_dir=task.runs_dir,
                )

        status = "timeout" if timed_out else ("done" if result_exit_code == 0 else "failed")
        end_time = time.time()
        await upload_heavy_logs(
            computer=computer,
            agent_start_time=int(start_time),
            agent_dir_config=AGENT_DIR_CONFIG,
            run_dir=task.run_dir,
            run_group_id=task.run_group_id,
            runs_dir=task.runs_dir,
            run_id=task.run_id,
            runtime=end_time - start_time,
        )
        rollout_artifacts = await self._sync_rollout_artifacts_to_host(computer, task)
        if result_exit_code == 0 and not rollout_artifacts["agent_log_exists"]:
            error_msg = "Codex rollout finished without syncing /home/logs/agent.log to host run_dir"
            status = "failed"
        await upload_status(int(start_time), task.run_dir, status, int(end_time))

        return AgentOutput(
            run_id=task.run_id,
            time_start=start_time,
            time_end=end_time,
            error_msg=error_msg,
            runtime_in_seconds=end_time - start_time,
            status_exists=bf.exists(bf.join(task.run_dir, "status.json")),
            agent_log_exists=bool(rollout_artifacts["agent_log_exists"]),
            codex_exit_status_exists=bool(rollout_artifacts["codex_exit_status_exists"]),
            codex_exit_code=rollout_artifacts["codex_exit_code"],
        )
