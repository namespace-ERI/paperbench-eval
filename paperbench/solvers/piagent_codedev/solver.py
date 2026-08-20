from __future__ import annotations

import json
import shlex
from typing import Any

import blobfile as bf
import chz
from typing_extensions import override

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.constants import AGENT_DIR, LOGS_DIR, PI_HOME_DIR, SUBMISSION_DIR
from paperbench.nano.task import PBTask
from paperbench.solvers.piagent.solver import (
    PI_SESSIONS_DIR,
    _pi_home_archive_snippet,
    _pi_home_redact_auth_snippet,
)
from paperbench.solvers.piagent_vanilla.solver import PiVanillaAgentSolver


SGLANG_QWEN_PROVIDER = "sglang-qwen"
SGLANG_QWEN_HOST = "arex.autoresearch.eval.baai.ac.cn"
SGLANG_QWEN_HOST_IP = "10.1.1.47"


def _sglang_qwen_models_json(*, model_id: str, base_url: str) -> str:
    payload = {
        "providers": {
            SGLANG_QWEN_PROVIDER: {
                "baseUrl": base_url.rstrip("/"),
                "api": "openai-completions",
                "apiKey": "inspectai",
                "authHeader": True,
                "models": [
                    {
                        "id": model_id,
                        "name": f"{model_id} (SGLang)",
                        "reasoning": True,
                        "input": ["text", "image"],
                        "contextWindow": 262_144,
                        "maxTokens": 32_768,
                        "compat": {
                            "supportsDeveloperRole": False,
                            "maxTokensField": "max_tokens",
                            "thinkingFormat": "qwen-chat-template",
                        },
                    }
                ],
            }
        }
    }
    return json.dumps(payload, indent=2) + "\n"


@chz.chz
class PiVanillaCodeDevAgentSolver(PiVanillaAgentSolver):
    """Pure PI with the exact official Code-Dev prompt and submission contract."""

    @override
    def shortname(self) -> str:
        return "piagent-codedev"

    @override
    def _prompt(self, task: PBTask) -> str:
        if not task.judge.code_only:
            raise ValueError("PiVanillaCodeDevAgentSolver requires paperbench.judge.code_only=true")
        if not task.prompt or not isinstance(task.prompt[0].get("content"), str):
            raise ValueError("PaperBench did not provide the official Code-Dev instructions")
        return str(task.prompt[0]["content"]).strip()

    @override
    async def _setup_pi_home(self, computer: ComputerInterface) -> None:
        await super()._setup_pi_home(computer)
        if self.pi_provider != SGLANG_QWEN_PROVIDER:
            raise ValueError(
                f"PiVanillaCodeDevAgentSolver expects provider {SGLANG_QWEN_PROVIDER!r}; "
                f"got {self.pi_provider!r}"
            )

        hosts_script = "\n".join(
            [
                "set -euo pipefail",
                f"if ! getent hosts {shlex.quote(SGLANG_QWEN_HOST)} >/dev/null 2>&1; then",
                f"  echo {shlex.quote(SGLANG_QWEN_HOST_IP + ' ' + SGLANG_QWEN_HOST)} | sudo tee -a /etc/hosts >/dev/null",
                "fi",
                f"getent hosts {shlex.quote(SGLANG_QWEN_HOST)} >/dev/null",
            ]
        )
        await computer.check_shell_command("bash -lc " + shlex.quote(hosts_script))
        await computer.upload(
            _sglang_qwen_models_json(
                model_id=self.pi_model,
                base_url=self.pi_upstream_base_url,
            ).encode("utf-8"),
            f"{PI_HOME_DIR}/models.json",
        )

    @override
    async def _check_remote_submission_reproduce_sh(
        self, computer: ComputerInterface, *, required: bool = True
    ) -> tuple[bool, str]:
        # The inherited rollout code calls this hook with required=False in Code-Dev
        # mode.  Code-Dev does not require reproduce.sh, but it still needs a real,
        # non-empty source submission for the official judge to inspect.
        script = "\n".join(
            [
                "set +e",
                f"cd {shlex.quote(SUBMISSION_DIR)} 2>/dev/null",
                "if [ $? -ne 0 ]; then echo 'submission directory missing'; exit 1; fi",
                "candidate=$(find . -path './.git' -prune -o -type f -print -quit)",
                "if [ -z \"${candidate:-}\" ]; then echo 'code-dev submission contains no files'; exit 1; fi",
                "echo 'code-dev submission contains files'",
            ]
        )
        result = await computer.send_shell_command("bash -lc " + shlex.quote(script))
        lines = result.unicode_output_best_effort.strip().splitlines()
        message = lines[-1] if lines else ""
        return result.exit_code == 0, message

    @override
    async def _finalize_submission(
        self,
        computer: ComputerInterface,
        final_exit_code: int,
        loop_state: dict[str, Any],
        *,
        code_only: bool = False,
    ) -> dict[str, Any]:
        if not code_only:
            raise ValueError("PiVanillaCodeDevAgentSolver requires code_only=True")

        # Keep Pi's normal finalization behavior isolated to this Code-Dev subclass,
        # while passing the current Codex/PaperBench code_only keyword through to the
        # status builder.  The legacy Pi solver's method predates that keyword.
        script = "\n".join(
            [
                "set +e",
                f"export PI_CODING_AGENT_DIR={PI_HOME_DIR}",
                f"export PI_CODING_AGENT_SESSION_DIR={PI_SESSIONS_DIR}",
                "export HOME=/home",
                "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/conda/bin",
                f"mkdir -p {shlex.quote(SUBMISSION_DIR)} {shlex.quote(LOGS_DIR)}",
                f"cd {shlex.quote(SUBMISSION_DIR)}",
                "git init >/dev/null 2>&1 || true",
                "git config user.email paperbench-pi@example.invalid >/dev/null 2>&1 || true",
                "git config user.name 'PaperBench Pi' >/dev/null 2>&1 || true",
                "rm -rf .paperbench_cache .pytest_cache .mypy_cache .ruff_cache .tox .nox .paperbench_venv .venv-paperbench .venv venv env node_modules",
                "find . -type d \\( -name __pycache__ -o -name .paperbench_venv -o -name .venv-paperbench -o -name .venv -o -name venv -o -name env -o -name node_modules \\) -prune -exec rm -rf {} +",
                "if [ -f .gitignore ]; then python3 - <<'PY'\nfrom pathlib import Path\np = Path('.gitignore')\nblocked = {'results/', 'results', '/results/', '/results', 'reproduce.log', 'reproduce.log.creation_time'}\nlines = p.read_text().splitlines()\nkept = [line for line in lines if line.strip() not in blocked]\np.write_text('\\n'.join(kept).rstrip() + ('\\n' if kept else ''))\nPY\nfi",
                "git add -A",
                "git diff --cached --quiet || git commit -m 'final paperbench submission'",
                _pi_home_archive_snippet(LOGS_DIR),
                _pi_home_redact_auth_snippet(),
                "exit 0",
            ]
        )
        await computer.send_shell_command("bash -lc " + shlex.quote(script))
        submission_ok, submission_message = await self._check_remote_submission_reproduce_sh(
            computer, required=False
        )
        status = self._build_submission_finalization_status(
            raw_exit_code=final_exit_code,
            reproduce_sh_ok=submission_ok,
            reproduce_sh_message=submission_message,
            loop_state=loop_state,
            completion_promise=self.loop_completion_promise,
            solver=self.shortname(),
            code_only=True,
        )
        status.update(
            {
                "submission_requirement": "nonempty_source_submission",
                "reproduce_sh_required": False,
            }
        )
        await self._write_remote_json(
            computer,
            f"{LOGS_DIR}/submission_finalization.json",
            status,
        )
        return status

    @override
    async def _sync_rollout_artifacts_to_host(
        self, computer: ComputerInterface, task: PBTask
    ) -> dict[str, object]:
        artifacts = await super()._sync_rollout_artifacts_to_host(computer, task)
        manifest_path = bf.join(task.run_dir, "logs", "pi_trajectory", "manifest.json")
        if bf.exists(manifest_path):
            try:
                with bf.BlobFile(manifest_path, "r") as handle:
                    manifest = json.loads(handle.read())
                if isinstance(manifest, dict):
                    manifest["prompt_source"] = "paperbench/instructions/code_only_instructions.txt"
                    manifest["code_only"] = True
                    bf.write_bytes(
                        manifest_path,
                        (json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
                    )
            except Exception:
                pass
        return artifacts
