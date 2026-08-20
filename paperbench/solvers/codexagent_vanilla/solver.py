from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

import blobfile as bf
import chz
from typing_extensions import override

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.constants import AGENT_DIR, CODEX_HOME_DIR, LOGS_DIR
from paperbench.nano.task import PBTask
from paperbench.solvers.codexagent.solver import (
    CODEX_CONFIG_TEMPLATE,
    CodexAgentSolver,
    _load_api_key,
)


@chz.chz
class CodexVanillaAgentSolver(CodexAgentSolver):
    """Codex CLI solver that uses the raw official PaperBench prompt and no skills."""

    codex_endpoint_mode: str = chz.field(default="responses")
    codex_chat_bridge_port: int = chz.field(default=18911)
    codex_context_window: int | None = chz.field(default=None)
    codex_auto_compact_token_limit: int | None = chz.field(default=None)
    codex_auto_compact_token_limit_scope: str = chz.field(default="total")

    @override
    def shortname(self) -> str:
        return "codexagent-vanilla"

    async def _write_vanilla_codex_config(
        self,
        computer: ComputerInterface,
        *,
        codex_base_url: str,
        requires_openai_auth: bool,
    ) -> None:
        config_lines = [
            CODEX_CONFIG_TEMPLATE.replace(
                'disable_response_storage = true',
                'disable_response_storage = false',
                1,
            )
            .format(
                model=self.codex_model,
                base_url=codex_base_url,
            )
            .replace(
                'requires_openai_auth = true',
                f"requires_openai_auth = {'true' if requires_openai_auth else 'false'}",
                1,
            )
            .rstrip(),
        ]
        if self.codex_context_window is not None:
            config_lines.append(f'model_context_window = {int(self.codex_context_window)}')
        if self.codex_auto_compact_token_limit is not None:
            config_lines.append(
                f'model_auto_compact_token_limit = {int(self.codex_auto_compact_token_limit)}'
            )
        if self.codex_auto_compact_token_limit_scope:
            config_lines.append(
                f'model_auto_compact_token_limit_scope = "{self.codex_auto_compact_token_limit_scope}"'
            )
        config_toml = "\n".join(config_lines) + "\n"
        auth_json = json.dumps({"OPENAI_API_KEY": _load_api_key(self.codex_api_key_env)}, indent=2)
        await computer.upload(auth_json.encode("utf-8"), f"{CODEX_HOME_DIR}/auth.json")
        await computer.upload(config_toml.encode("utf-8"), f"{CODEX_HOME_DIR}/config.toml")

    @override
    async def _setup_computer(self, computer: ComputerInterface, task: PBTask) -> None:
        del task
        await self._setup_codex_home(computer, include_skill_dirs=False)
        codex_base_url = self.codex_base_url
        requires_openai_auth = True
        if self.codex_endpoint_mode == "chat_bridge":
            await self._start_chat_bridge(computer, upstream_base_url=self.codex_base_url)
            codex_base_url = f"http://127.0.0.1:{int(self.codex_chat_bridge_port)}"
            requires_openai_auth = False
        elif self.codex_endpoint_mode != "responses":
            raise ValueError(
                f"Unsupported codex_endpoint_mode={self.codex_endpoint_mode!r}; expected 'responses' or 'chat_bridge'."
            )
        await self._write_vanilla_codex_config(
            computer,
            codex_base_url=codex_base_url,
            requires_openai_auth=requires_openai_auth,
        )
        await self._restore_continuation_state(computer)
        # Restored codex_home archives preserve trajectory/session state, but
        # their config.toml can point at a previous run's dead bridge port.
        await self._write_vanilla_codex_config(
            computer,
            codex_base_url=codex_base_url,
            requires_openai_auth=requires_openai_auth,
        )

    @override
    def _prompt(self, task: PBTask) -> str:
        return self._prompt_instructions(task.judge.code_only)

    async def _start_chat_bridge(
        self,
        computer: ComputerInterface,
        *,
        upstream_base_url: str,
    ) -> None:
        upstream_api_key = _load_api_key(self.codex_api_key_env)
        local_bridge_path = Path(__file__).with_name("chat_responses_bridge.py")
        remote_bridge_path = f"{AGENT_DIR}/chat_responses_bridge.py"
        remote_config_path = f"{AGENT_DIR}/chat_responses_bridge_config.json"
        await computer.upload(local_bridge_path.read_bytes(), remote_bridge_path)
        await computer.upload(
            (
                json.dumps(
                    {
                        "listen_host": "127.0.0.1",
                        "listen_port": int(self.codex_chat_bridge_port),
                        "upstream_base_url": upstream_base_url,
                        "alternate_base_urls": [
                            item.strip()
                            for item in os.getenv("PAPERBENCH_RESPONSES_ALT_BASE_URLS", "")
                            .replace("\n", ",")
                            .split(",")
                            if item.strip()
                        ],
                        "upstream_api_key": upstream_api_key,
                        "timeout_seconds": max(1800, int(self.time_limit or 1800)),
                        "max_retries": 8,
                        "retry_initial_sleep_seconds": 5,
                        "retry_max_sleep_seconds": 120,
                    },
                    indent=2,
                )
                + "\n"
            ).encode("utf-8"),
            remote_config_path,
        )
        startup_script = "\n".join(
            [
                "set -euo pipefail",
                f"mkdir -p {shlex.quote(AGENT_DIR)} {shlex.quote(LOGS_DIR)}",
                f"pid_file={shlex.quote(LOGS_DIR)}/chat_responses_bridge.pid",
                "if [ -f \"$pid_file\" ]; then",
                "  old_pid=$(cat \"$pid_file\" 2>/dev/null || true)",
                "  if [ -n \"${old_pid:-}\" ] && kill -0 \"$old_pid\" 2>/dev/null; then",
                "    kill \"$old_pid\" >/dev/null 2>&1 || true",
                "    sleep 1",
                "    if kill -0 \"$old_pid\" 2>/dev/null; then",
                "      kill -9 \"$old_pid\" >/dev/null 2>&1 || true",
                "    fi",
                "  fi",
                "fi",
                f"rm -f {shlex.quote(LOGS_DIR)}/chat_responses_bridge.log \"$pid_file\"",
                (
                    f"nohup python3 {shlex.quote(remote_bridge_path)} "
                    f"--config {shlex.quote(remote_config_path)} "
                    f"> {shlex.quote(LOGS_DIR)}/chat_responses_bridge.log 2>&1 & echo $! > \"$pid_file\""
                ),
                "python3 - <<'PY'",
                "import sys",
                "import time",
                "import urllib.request",
                f"url = 'http://127.0.0.1:{int(self.codex_chat_bridge_port)}/healthz'",
                "deadline = time.time() + 20.0",
                "last_error = None",
                "while time.time() < deadline:",
                "    try:",
                "        with urllib.request.urlopen(url, timeout=1.0) as response:",
                "            body = response.read().decode('utf-8', errors='replace').strip()",
                "            if response.status == 200 and body == 'ok':",
                "                sys.exit(0)",
                "    except Exception as exc:",
                "        last_error = exc",
                "    time.sleep(0.25)",
                "raise SystemExit(f'chat bridge failed readiness check: {last_error}')",
                "PY",
            ]
        )
        await computer.check_shell_command("bash -lc " + shlex.quote(startup_script))

    @override
    def _prompt(self, task: PBTask) -> str:
        del task
        return OFFICIAL_PAPERBENCH_INSTRUCTIONS.strip()

    @override
    async def _sync_rollout_artifacts_to_host(self, computer: ComputerInterface, task: PBTask) -> dict[str, object]:
        artifacts = await super()._sync_rollout_artifacts_to_host(computer, task)
        trajectory_root = bf.join(task.run_dir, "logs", "codex_trajectory")
        preserved: dict[str, str] = {}
        trajectory_root_path = Path(trajectory_root)

        run_dir_path = Path(task.run_dir)
        if len(run_dir_path.parents) >= 3:
            run_root_path = run_dir_path.parents[2]
            host_bridge_sources = {
                "host_chat_responses_bridge.log": run_root_path / "logs" / "host_chat_responses_bridge.log",
                "host_chat_responses_bridge_config.json": run_root_path / "logs" / "host_chat_responses_bridge_config.json",
            }
            for artifact_name, source_path in host_bridge_sources.items():
                if source_path.exists():
                    trajectory_root_path.mkdir(parents=True, exist_ok=True)
                    target_path = trajectory_root_path / artifact_name
                    target_path.write_bytes(source_path.read_bytes())
                    preserved[artifact_name] = str(target_path)

        prompt_host = bf.join(trajectory_root, "codex_prompt.txt")
        if await self._copy_remote_file_to_host_if_present(
            computer,
            remote_path=f"{AGENT_DIR}/codex_prompt.txt",
            local_path=prompt_host,
            task=task,
        ):
            preserved["prompt"] = prompt_host

        loop_archive_host = bf.join(trajectory_root, "loop.tar.gz")
        if await self._copy_remote_directory_archive_to_host_if_present(
            computer,
            remote_dir=f"{LOGS_DIR}/loop",
            local_archive_path=loop_archive_host,
        ):
            preserved["loop_archive"] = loop_archive_host

        codex_home_archive_host = bf.join(trajectory_root, "codex_home.tar.gz")
        if await self._copy_remote_file_to_host_if_present(
            computer,
            remote_path=f"{LOGS_DIR}/codex_home.tar.gz",
            local_path=codex_home_archive_host,
            task=task,
        ):
            preserved["codex_home_archive"] = codex_home_archive_host

        codex_home_live_archive_host = bf.join(trajectory_root, "codex_home_live.tar.gz")
        if await self._copy_remote_directory_archive_to_host_if_present(
            computer,
            remote_dir=CODEX_HOME_DIR,
            local_archive_path=codex_home_live_archive_host,
        ):
            preserved["codex_home_live_archive"] = codex_home_live_archive_host

        for remote_name in [
            "codex_session_state.json",
            "codex_loop_state.json",
            "codex_exit_status.txt",
            "codex_solver_error.log",
        ]:
            host_path = bf.join(trajectory_root, remote_name)
            if await self._copy_remote_file_to_host_if_present(
                computer,
                remote_path=f"{LOGS_DIR}/{remote_name}",
                local_path=host_path,
                task=task,
            ):
                preserved[remote_name] = host_path

        bridge_log_host = bf.join(trajectory_root, "chat_responses_bridge.log")
        if await self._copy_remote_file_to_host_if_present(
            computer,
            remote_path=f"{LOGS_DIR}/chat_responses_bridge.log",
            local_path=bridge_log_host,
            task=task,
        ):
            preserved["chat_responses_bridge.log"] = bridge_log_host

        manifest_path = bf.join(trajectory_root, "manifest.json")
        manifest = {
            "solver": self.shortname(),
            "full_trajectory_enabled": True,
            "prompt_source": "paperbench/instructions/instructions.txt",
            "response_storage_disabled": False,
            "aggregated_agent_log": bf.join(task.run_dir, "agent.log"),
            "preserved_artifacts": preserved,
        }
        if not bf.exists(trajectory_root):
            bf.makedirs(trajectory_root)
        bf.write_bytes(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
        artifacts["full_trajectory_manifest_exists"] = True
        artifacts["full_trajectory_manifest_path"] = manifest_path
        return artifacts
