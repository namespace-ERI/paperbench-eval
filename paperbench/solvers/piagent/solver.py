from __future__ import annotations

import json
import os
import shlex
import time
import shutil
from pathlib import Path
from typing import Any
import urllib.request

import blobfile as bf
import chz
from typing_extensions import override

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.constants import AGENT_DIR, LOGS_DIR, PI_HOME_DIR, SUBMISSION_DIR
from paperbench.nano.task import PBTask
from paperbench.solvers.codexagent.solver import (
    CodexAgentSolver,
    OFFICIAL_RUN_ROOT_DIR_NAMES,
    _load_api_key,
    _repo_relative_str,
    _run_root_for_submission_tar,
    _submission_payload_file_count,
    _tar_directory_bytes,
)
from paperbench.utils import get_root


PAPERBENCH_MODULE_ROOT = get_root()
REPO_ROOT = PAPERBENCH_MODULE_ROOT.parent
PI_PROXY_DIR = f"{AGENT_DIR}/cliproxyapi-su8"
PI_PROXY_CONFIG_PATH = f"{PI_PROXY_DIR}/config.yaml"
PI_PROXY_BIN_PATH = f"{PI_PROXY_DIR}/cli-proxy-api"
PI_SESSIONS_DIR = f"{PI_HOME_DIR}/sessions"
PI_SKILLS_DIR = f"{PI_HOME_DIR}/skills"
# The relay key is normally injected by the surrounding environment. Fall back
# to the historical default when the override is absent so the code still works
# in older/local setups.
PI_LOCAL_API_KEY = os.getenv("PAPERBENCH_PI_LOCAL_API_KEY", "sk-local-su8-paperbench")
PI_NODE_HOME = "/home/agent/node-v22.19.0-linux-x64"
PI_NODE_TARBALL_NAME = "node-v22.19.0-linux-x64.tar.xz"
PI_NODE_TARBALL_URL = "https://nodejs.org/dist/v22.19.0/node-v22.19.0-linux-x64.tar.xz"
PI_NODE_TARBALL_CACHE = Path.home() / ".cache" / "paperbench" / "pi-agent" / PI_NODE_TARBALL_NAME
KIMI_CODING_PROVIDER = "kimi-coding"
KIMI_CODING_DEFAULT_BASE_URL = "https://api.kimi.com/coding/v1"
SGLANG_QWEN_PROVIDER = "sglang-qwen"
SGLANG_QWEN_DEFAULT_BASE_URL = "http://arex.autoresearch.eval.baai.ac.cn/"
SGLANG_QWEN_DEFAULT_MODEL_ID = "qwen35_35b_a3b"
SGLANG_QWEN_DEFAULT_MODEL_NAME = "Qwen3.5-35B-A3B (SGLang)"
SGLANG_QWEN_DEFAULT_CONTEXT_WINDOW = 262_144
SGLANG_QWEN_DEFAULT_MAX_TOKENS = 32_768
SGLANG_QWEN_DEFAULT_THINKING_FORMAT = "qwen-chat-template"


PI_MODELS: list[dict[str, Any]] = [
    {
        "id": "gpt-5.5-su8",
        "name": "GPT-5.5 (CLIProxyAPI su8)",
        "reasoning": True,
        "input": ["text", "image"],
        "contextWindow": 272000,
        "maxTokens": 128000,
        "thinkingLevelMap": {
            "off": "none",
            "minimal": None,
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh",
            "max": None,
        },
        "compat": {
            "supportsStrictMode": True,
            "supportsOpenAIGrammarTools": True,
            "supportsToolSearch": True,
        },
    },
    {
        "id": "gpt-5.6-sol-su8",
        "name": "GPT-5.6 Sol (CLIProxyAPI su8)",
        "reasoning": True,
        "input": ["text", "image"],
        "contextWindow": 272000,
        "maxTokens": 128000,
        "thinkingLevelMap": {
            "off": "none",
            "minimal": None,
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh",
            "max": "max",
        },
        "compat": {
            "supportsStrictMode": True,
            "supportsOpenAIGrammarTools": True,
            "supportsToolSearch": True,
        },
    },
    {
        "id": "gpt-5.6-terra-su8",
        "name": "GPT-5.6 Terra (CLIProxyAPI su8)",
        "reasoning": True,
        "input": ["text", "image"],
        "contextWindow": 272000,
        "maxTokens": 128000,
        "thinkingLevelMap": {
            "off": "none",
            "minimal": None,
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh",
            "max": "max",
        },
        "compat": {
            "supportsStrictMode": True,
            "supportsOpenAIGrammarTools": True,
            "supportsToolSearch": True,
        },
    },
    {
        "id": "gpt-5.6-luna-su8",
        "name": "GPT-5.6 Luna (CLIProxyAPI su8)",
        "reasoning": True,
        "input": ["text", "image"],
        "contextWindow": 272000,
        "maxTokens": 128000,
        "thinkingLevelMap": {
            "off": "none",
            "minimal": None,
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh",
            "max": "max",
        },
        "compat": {
            "supportsStrictMode": True,
            "supportsOpenAIGrammarTools": True,
            "supportsToolSearch": True,
        },
    },
]


def _find_cliproxyapi_binary() -> Path:
    candidates = [
        Path(os.getenv("PAPERBENCH_CLIPROXYAPI_BIN", "")),
        Path.home() / ".local" / "bin" / "cli-proxy-api",
        Path.home() / ".local" / "share" / "cliproxyapi" / "7.2.72" / "cli-proxy-api",
        Path.home() / ".local" / "share" / "cliproxyapi" / "7.2.72" / "CLIProxyAPI",
    ]
    for candidate in candidates:
        if str(candidate) and candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "CLIProxyAPI binary not found. Install it with the SU8tips instructions or set "
        "PAPERBENCH_CLIPROXYAPI_BIN."
    )


def _pi_home_archive_snippet(log_dir: str) -> str:
    return "\n".join(
        [
            "python3 - <<'PY'",
            "from pathlib import Path",
            "import json",
            "import shutil",
            "import tarfile",
            "import tempfile",
            f"src = Path('{PI_HOME_DIR}')",
            f"out = Path('{log_dir}') / 'pi_home.tar.gz'",
            "if src.exists():",
            "    with tempfile.TemporaryDirectory() as tmp:",
            "        dst = Path(tmp) / 'pi_agent'",
            "        shutil.copytree(src, dst)",
            "        for auth_path in dst.rglob('auth.json'):",
            "            try:",
            "                payload = json.loads(auth_path.read_text(encoding='utf-8'))",
            "                if isinstance(payload, dict):",
            "                    for value in payload.values():",
            "                        if isinstance(value, dict) and 'key' in value:",
            "                            value['key'] = '<redacted>'",
            "                auth_path.write_text(json.dumps(payload, indent=2) + '\\n', encoding='utf-8')",
            "            except Exception:",
            "                auth_path.write_text('<redacted>\\n', encoding='utf-8')",
            "        with tarfile.open(out, 'w:gz') as tar:",
            "            tar.add(dst, arcname='pi_agent')",
            "PY",
        ]
    )


def _pi_home_redact_auth_snippet() -> str:
    return "\n".join(
        [
            "python3 - <<'PY'",
            "from pathlib import Path",
            "import json",
            f"src = Path('{PI_HOME_DIR}')",
            "if src.exists():",
            "    for auth_path in src.rglob('auth.json'):",
            "        try:",
            "            payload = json.loads(auth_path.read_text(encoding='utf-8'))",
            "            if isinstance(payload, dict):",
            "                for value in payload.values():",
            "                    if isinstance(value, dict) and 'key' in value:",
            "                        value['key'] = '<redacted>'",
            "            auth_path.write_text(json.dumps(payload, indent=2) + '\\n', encoding='utf-8')",
            "        except Exception:",
            "            auth_path.write_text('<redacted>\\n', encoding='utf-8')",
            "PY",
        ]
    )


def _ensure_pi_node_tarball() -> Path:
    cache_path = PI_NODE_TARBALL_CACHE
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.parent / f"{cache_path.name}.tmp"
    with urllib.request.urlopen(PI_NODE_TARBALL_URL, timeout=120) as response, temp_path.open("wb") as out:
        shutil.copyfileobj(response, out)
    temp_path.replace(cache_path)
    return cache_path


def _pi_proxy_config_yaml(*, upstream_api_key: str, upstream_base_url: str, port: int) -> str:
    zero_width_space = "\u200b"
    proxy_url = os.getenv("PB_PROXY_URL", "http://172.17.0.1:7895")
    return f"""host: "127.0.0.1"
port: {int(port)}
tls:
  enable: false
  cert: ""
  key: ""
remote-management:
  allow-remote: false
  secret-key: ""
  disable-control-panel: true
auth-dir: "{PI_PROXY_DIR}/auth"
api-keys:
  - "{PI_LOCAL_API_KEY}"
debug: true
commercial-mode: false
disable-image-generation: "chat"
request-log: true
logging-to-file: true
logs-max-total-size-mb: 100
usage-statistics-enabled: false
request-retry: 3
disable-cooling: true
proxy-url: "{proxy_url}"
routing:
  strategy: "round-robin"
openai-compatibility:
  - name: "su8"
    base-url: "{upstream_base_url}"
    headers:
      User-Agent: "codex_exec/0.142.3 (Ubuntu 24.4.0; x86_64) vscode/1.100.0 (codex_exec; 0.142.3)"
      originator: "codex_exec"
      x-codex-beta-features: "remote_compaction_v2"
      X-Api-Key: "{upstream_api_key}"
    api-key-entries:
      - api-key: "{upstream_api_key}"
    models:
      - name: "gpt-5.5"
        alias: "gpt-5.5-su8"
        display-name: "GPT-5.5 SU8"
        force-mapping: true
        is-compat: true
      - name: "gpt-5.6-sol"
        alias: "gpt-5.6-sol-su8"
        display-name: "GPT-5.6 Sol SU8"
        force-mapping: true
        is-compat: true
      - name: "gpt-5.6-terra"
        alias: "gpt-5.6-terra-su8"
        display-name: "GPT-5.6 Terra SU8"
        force-mapping: true
        is-compat: true
      - name: "gpt-5.6-luna"
        alias: "gpt-5.6-luna-su8"
        display-name: "GPT-5.6 Luna SU8"
        force-mapping: true
        is-compat: true
payload:
  override:
    - models:
        - name: "gpt-*"
          protocol: "openai-response"
      params:
        instructions: "{zero_width_space}"
"""


def _pi_models_json(*, port: int) -> str:
    payload = {
        "providers": {
            "su8": {
                "baseUrl": f"http://127.0.0.1:{int(port)}/v1",
                "api": "openai-responses",
                "authHeader": True,
                "headers": {"User-Agent": "pi-coding-agent"},
                "models": PI_MODELS,
            }
        }
    }
    return json.dumps(payload, indent=2) + "\n"


def _kimi_coding_models_json(*, model_id: str, base_url: str) -> str:
    context_window = 1_048_576 if model_id == "k3" else 262_144
    payload = {
        "providers": {
            KIMI_CODING_PROVIDER: {
                "baseUrl": base_url,
                "api": "openai-completions",
                "authHeader": True,
                "models": [
                    {
                        "id": model_id,
                        "name": f"Kimi Code {model_id}",
                        "reasoning": True,
                        "input": ["text", "image"],
                        "contextWindow": context_window,
                        "maxTokens": 131_072,
                        "thinkingLevelMap": {
                            "off": None,
                            "minimal": None,
                            "low": "low",
                            "medium": "high",
                            "high": "high",
                            "xhigh": "max",
                            "max": "max",
                        },
                        "compat": {
                            "supportsDeveloperRole": False,
                            "supportsReasoningEffort": True,
                            "deferredToolsMode": "kimi",
                        },
                    }
                ],
            }
        }
    }
    return json.dumps(payload, indent=2) + "\n"


def _sglang_qwen_models_json(*, model_id: str, base_url: str, api_key: str) -> str:
    payload = {
        "providers": {
            SGLANG_QWEN_PROVIDER: {
                "baseUrl": base_url,
                "api": "openai-completions",
                "apiKey": api_key,
                "authHeader": True,
                "models": [
                    {
                        "id": model_id,
                        "name": SGLANG_QWEN_DEFAULT_MODEL_NAME,
                        "reasoning": True,
                        "input": ["text", "image"],
                        "contextWindow": SGLANG_QWEN_DEFAULT_CONTEXT_WINDOW,
                        "maxTokens": SGLANG_QWEN_DEFAULT_MAX_TOKENS,
                        "compat": {
                            "supportsDeveloperRole": False,
                            "maxTokensField": "max_tokens",
                            "thinkingFormat": SGLANG_QWEN_DEFAULT_THINKING_FORMAT,
                        },
                    }
                ],
            }
        }
    }
    return json.dumps(payload, indent=2) + "\n"


@chz.chz
class PiAgentSolver(CodexAgentSolver):
    """Runs the Pi CLI as the skill-enabled PaperBench rollout agent inside the task container."""

    pi_model: str = chz.field(default="gpt-5.5-su8")
    pi_provider: str = chz.field(default="su8")
    pi_thinking: str = chz.field(default="xhigh")
    pi_api_key_env: str = chz.field(default="OPENAI_API_KEY")
    pi_upstream_base_url: str = chz.field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://www.su8.codes/v1")
    )
    pi_kimi_base_url: str = chz.field(
        default_factory=lambda: os.getenv("PI_KIMI_BASE_URL", KIMI_CODING_DEFAULT_BASE_URL)
    )
    pi_kimi_custom_models_enabled: bool = chz.field(
        default_factory=lambda: os.getenv("PAPERBENCH_PI_KIMI_CUSTOM_MODELS", "").lower()
        in {"1", "true", "yes", "on"}
    )
    pi_relay_port: int = chz.field(default=8318)
    pi_relay_enabled: bool = chz.field(default=True)
    pi_cli_version: str = chz.field(default="0.84.0")
    pi_install_timeout_seconds: int = chz.field(default=600)

    @override
    def shortname(self) -> str:
        return "piagent"

    def _pi_use_skills(self) -> bool:
        return True

    def _pi_skill_args(self) -> list[str]:
        if not self._pi_use_skills():
            return ["--no-skills"]
        skills_path = Path(self.skills_dir).resolve()
        if not skills_path.exists():
            raise FileNotFoundError(f"Pi skills directory does not exist: {skills_path}")
        return ["--no-skills", "--skill", PI_SKILLS_DIR]

    def _pi_use_relay(self) -> bool:
        return bool(self.pi_relay_enabled) and self.pi_provider == "su8"

    def _pi_auth_provider(self) -> str:
        return self.pi_provider

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

    async def _setup_pi_home(self, computer: ComputerInterface) -> None:
        mkdir_parts = [
            AGENT_DIR,
            LOGS_DIR,
            PI_HOME_DIR,
            PI_SESSIONS_DIR,
        ]
        if self._pi_use_relay():
            mkdir_parts.extend([PI_PROXY_DIR, f"{PI_PROXY_DIR}/auth", f"{PI_PROXY_DIR}/logs"])
        await computer.check_shell_command(f"mkdir -p {' '.join(mkdir_parts)}")
        await computer.check_shell_command(f"rm -rf {shlex.quote(PI_SESSIONS_DIR)}/*")

        if self._pi_use_relay():
            proxy_binary = _find_cliproxyapi_binary()
            await computer.upload(proxy_binary.read_bytes(), PI_PROXY_BIN_PATH)
            await computer.check_shell_command(f"chmod 755 {shlex.quote(PI_PROXY_BIN_PATH)}")

            upstream_api_key = _load_api_key(self.pi_api_key_env)
            await computer.upload(
                _pi_proxy_config_yaml(
                    upstream_api_key=upstream_api_key,
                    upstream_base_url=self.pi_upstream_base_url,
                    port=self.pi_relay_port,
                ).encode("utf-8"),
                PI_PROXY_CONFIG_PATH,
            )
            await computer.upload(
                _pi_models_json(port=self.pi_relay_port).encode("utf-8"),
                f"{PI_HOME_DIR}/models.json",
            )
            auth_payload = {"su8": {"type": "api_key", "key": PI_LOCAL_API_KEY}}
        else:
            api_key = os.getenv(self.pi_api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"No API key found. Set {self.pi_api_key_env} before running Pi provider "
                    f"{self.pi_provider!r} without the SU8 relay."
                )
            if self._pi_auth_provider() == SGLANG_QWEN_PROVIDER:
                await computer.upload(
                    _sglang_qwen_models_json(
                        model_id=self.pi_model,
                        base_url=self.pi_upstream_base_url,
                        api_key=api_key,
                    ).encode("utf-8"),
                    f"{PI_HOME_DIR}/models.json",
                )
            elif self._pi_auth_provider() == KIMI_CODING_PROVIDER and self.pi_kimi_custom_models_enabled:
                await computer.upload(
                    _kimi_coding_models_json(
                        model_id=self.pi_model,
                        base_url=self.pi_kimi_base_url,
                    ).encode("utf-8"),
                    f"{PI_HOME_DIR}/models.json",
                )
            else:
                await computer.send_shell_command(f"rm -f {shlex.quote(PI_HOME_DIR)}/models.json")
            auth_payload = {self._pi_auth_provider(): {"type": "api_key", "key": api_key}}

        await computer.upload(
            (json.dumps(auth_payload, indent=2) + "\n").encode("utf-8"),
            f"{PI_HOME_DIR}/auth.json",
        )
        pi_node_tarball = _ensure_pi_node_tarball()
        await computer.upload(pi_node_tarball.read_bytes(), f"{AGENT_DIR}/{PI_NODE_TARBALL_NAME}")
        await computer.upload(
            (
                json.dumps(
                    {
                        "lastChangelogVersion": self.pi_cli_version,
                        "defaultProvider": self.pi_provider,
                        "defaultModel": self.pi_model,
                        "defaultThinkingLevel": self.pi_thinking,
                        "defaultProjectTrust": "always",
                        "enableInstallTelemetry": False,
                        "theme": "dark",
                    },
                    indent=2,
                )
                + "\n"
            ).encode("utf-8"),
            f"{PI_HOME_DIR}/settings.json",
        )

        install_script = "\n".join(
            [
                "set -euo pipefail",
                "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/conda/bin",
                "export HTTP_PROXY=${PB_PROXY_URL:-http://172.17.0.1:7895}",
                "export HTTPS_PROXY=${PB_PROXY_URL:-http://172.17.0.1:7895}",
                "export ALL_PROXY=${PB_PROXY_URL:-http://172.17.0.1:7895}",
                "export http_proxy=${PB_PROXY_URL:-http://172.17.0.1:7895}",
                "export https_proxy=${PB_PROXY_URL:-http://172.17.0.1:7895}",
                "export all_proxy=${PB_PROXY_URL:-http://172.17.0.1:7895}",
                "export npm_config_proxy=${PB_PROXY_URL:-http://172.17.0.1:7895}",
                "export npm_config_https_proxy=${PB_PROXY_URL:-http://172.17.0.1:7895}",
                "export npm_config_registry=https://registry.npmjs.org/",
                f"if [ ! -x {shlex.quote(PI_NODE_HOME + '/bin/node')} ] || [ \"$({shlex.quote(PI_NODE_HOME + '/bin/node')} --version 2>/dev/null || true)\" != v22.19.0 ]; then",
                f"  python3 - <<'PY'\nfrom pathlib import Path\nimport tarfile\narchive = Path({(AGENT_DIR + '/' + PI_NODE_TARBALL_NAME)!r})\nwith tarfile.open(archive, 'r:xz') as tar:\n    tar.extractall('/home/agent')\nPY",
                "fi",
                f"export PATH={shlex.quote(PI_NODE_HOME)}/bin:$PATH",
                "if ! command -v pi >/dev/null 2>&1 || [ \"$(pi --version 2>/dev/null || true)\" != "
                f"{shlex.quote(self.pi_cli_version)} ]; then",
                f"  timeout {int(self.pi_install_timeout_seconds)}s npm install -g @earendil-works/pi-coding-agent@{shlex.quote(self.pi_cli_version)}",
                "fi",
                "pi --version > /home/logs/pi_version.txt",
            ]
        )
        await computer.check_shell_command("bash -lc " + shlex.quote(install_script))

        if self._pi_use_relay():
            startup_script = "\n".join(
                [
                    "set -euo pipefail",
                    f"mkdir -p {shlex.quote(PI_PROXY_DIR)}/auth {shlex.quote(PI_PROXY_DIR)}/logs {shlex.quote(LOGS_DIR)}",
                    f"pid_file={shlex.quote(LOGS_DIR)}/pi_cliproxyapi.pid",
                    "if [ -f \"$pid_file\" ]; then",
                    "  old_pid=$(cat \"$pid_file\" 2>/dev/null || true)",
                    "  if [ -n \"${old_pid:-}\" ] && kill -0 \"$old_pid\" 2>/dev/null; then",
                    "    kill \"$old_pid\" >/dev/null 2>&1 || true",
                    "    sleep 1",
                    "    if kill -0 \"$old_pid\" 2>/dev/null; then kill -9 \"$old_pid\" >/dev/null 2>&1 || true; fi",
                    "  fi",
                    "fi",
                    f"rm -f {shlex.quote(LOGS_DIR)}/pi_cliproxyapi.log \"$pid_file\"",
                    "export NO_PROXY=127.0.0.1,localhost,::1",
                    "export no_proxy=127.0.0.1,localhost,::1",
                    (
                        f"nohup {shlex.quote(PI_PROXY_BIN_PATH)} --config {shlex.quote(PI_PROXY_CONFIG_PATH)} "
                        f"> {shlex.quote(LOGS_DIR)}/pi_cliproxyapi.log 2>&1 & echo $! > \"$pid_file\""
                    ),
                    "python3 - <<'PY'",
                    "import sys",
                    "import time",
                    "import urllib.request",
                    f"url = 'http://127.0.0.1:{int(self.pi_relay_port)}/healthz'",
                    "deadline = time.time() + 90.0",
                    "last_error = None",
                    "while time.time() < deadline:",
                    "    try:",
                    "        with urllib.request.urlopen(url, timeout=1.0) as response:",
                    "            response.read()",
                    "            if response.status == 200:",
                    "                sys.exit(0)",
                    "    except Exception as exc:",
                    "        last_error = exc",
                    "    time.sleep(0.25)",
                    "raise SystemExit(f'pi relay failed readiness check: {last_error}')",
                    "PY",
                ]
            )
            await computer.check_shell_command("bash -lc " + shlex.quote(startup_script))

            # The relay may rewrite or preserve its effective API key at startup.
            # Refresh the client auth file from the active relay config so the PI CLI
            # always authenticates with the same key the relay is actually using.
            sync_auth_script = "\n".join(
                [
                    "python3 - <<'PY'",
                    "from pathlib import Path",
                    "import json",
                    f"config_path = Path({PI_PROXY_CONFIG_PATH!r})",
                    f"auth_path = Path({PI_HOME_DIR!r}) / 'auth.json'",
                    "def first_api_key(text: str) -> str | None:",
                    "    in_api_keys = False",
                    "    api_keys_indent = 0",
                    "    for raw_line in text.splitlines():",
                    "        stripped = raw_line.strip()",
                    "        if not in_api_keys:",
                    "            if stripped in {'api-keys:', 'api_keys:'}:",
                    "                in_api_keys = True",
                    "                api_keys_indent = len(raw_line) - len(raw_line.lstrip())",
                    "            continue",
                    "        if not stripped or stripped.startswith('#'):",
                    "            continue",
                    "        current_indent = len(raw_line) - len(raw_line.lstrip())",
                    "        if current_indent <= api_keys_indent:",
                    "            break",
                    "        if stripped.startswith('- '):",
                    "            value = stripped[2:].strip()",
                    "            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', \"'\"}:",
                    "                value = value[1:-1]",
                    "            return value or None",
                    "    return None",
                    "try:",
                    "    relay_key = first_api_key(config_path.read_text())",
                    "except Exception:",
                    "    relay_key = None",
                    f"if not isinstance(relay_key, str) or not relay_key.strip(): relay_key = {PI_LOCAL_API_KEY!r}",
                    "auth_path.write_text(",
                    "    json.dumps({'su8': {'type': 'api_key', 'key': relay_key}}, indent=2) + '\\n'",
                    ")",
                    "PY",
                ]
            )
            await computer.check_shell_command("bash -lc " + shlex.quote(sync_auth_script))

    async def _install_pi_distilled_skills(self, computer: ComputerInterface) -> None:
        if not self._pi_use_skills():
            return

        skills_path = Path(self.skills_dir).resolve()
        if not skills_path.exists():
            raise FileNotFoundError(f"Pi skills directory does not exist: {skills_path}")

        await computer.upload(_tar_directory_bytes(skills_path), f"{AGENT_DIR}/pi_skills.tar.gz")
        await computer.check_shell_command(
            "bash -lc "
            + shlex.quote(
                f"set -euo pipefail; "
                f"rm -rf /home/skill {shlex.quote(PI_SKILLS_DIR)}; "
                f"mkdir -p {shlex.quote(PI_SKILLS_DIR)}; "
                f"tar -xzf {shlex.quote(AGENT_DIR)}/pi_skills.tar.gz -C /home; "
                f"cp -R /home/skill/. {shlex.quote(PI_SKILLS_DIR)}/; "
                f"find {shlex.quote(PI_SKILLS_DIR)} -name SKILL.md | sort > {shlex.quote(LOGS_DIR)}/installed_pi_skills.txt"
            )
        )

    @override
    async def _setup_computer(self, computer: ComputerInterface, task: PBTask) -> None:
        del task
        await self._setup_pi_home(computer)
        await self._restore_continuation_state(computer)
        await self._setup_pi_home(computer)
        await self._install_pi_distilled_skills(computer)

    @override
    def _prompt(self, task: PBTask) -> str:
        prefix = ""
        continuation = self._continuation_manifest()
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

        if self._pi_use_skills():
            prefix += self._additional_paperbench_note(task.judge.code_only)

        mode_contract = (
            self._loop_contract(code_only=task.judge.code_only)
            if self.loop_enabled
            else self._single_run_contract()
        )

        return (
            f"/goal Reproduce the assigned PaperBench paper `{task.paper_id}` from the materials in /home/paper and create a complete submission in {SUBMISSION_DIR}.\n\n"
            f"{mode_contract}\n"
            f"{prefix}"
            f"{self._prompt_instructions(task.judge.code_only)}"
        )

    async def _run_codex_iteration(
        self,
        computer: ComputerInterface,
        prompt_path: str,
        iteration: int,
        time_limit_seconds: int | None,
    ) -> dict[str, Any]:
        loop_dir = f"{LOGS_DIR}/loop"
        iter_slug = f"iter_{iteration:04d}"
        iter_log_path = f"{loop_dir}/{iter_slug}.pi.jsonl"
        last_message_path = f"{loop_dir}/{iter_slug}.last_message.txt"
        summary_path = f"{loop_dir}/{iter_slug}.summary.json"
        cli_args = " ".join(shlex.quote(item) for item in self._pi_skill_args())
        if self._pi_use_relay():
            proxy_env = (
                "HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= "
                "NO_PROXY=127.0.0.1,localhost,::1 no_proxy=127.0.0.1,localhost,::1 "
            )
        else:
            proxy_env = (
                "NO_PROXY=${NO_PROXY:-127.0.0.1,localhost,::1} "
                "no_proxy=${no_proxy:-127.0.0.1,localhost,::1} "
            )
        pi_exec = (
            "prompt=$(cat "
            f"{shlex.quote(prompt_path)}"
            "); "
            "PI_TELEMETRY=0 "
            f"PI_CODING_AGENT_DIR={shlex.quote(PI_HOME_DIR)} "
            f"PI_CODING_AGENT_SESSION_DIR={shlex.quote(PI_SESSIONS_DIR)} "
            f"{proxy_env}"
            "pi "
            f"--provider {shlex.quote(self.pi_provider)} "
            f"--model {shlex.quote(self.pi_model)} "
            f"--thinking {shlex.quote(self.pi_thinking)} "
            "--mode json "
            f"--session-dir {shlex.quote(PI_SESSIONS_DIR)} "
            "--approve "
            "--no-extensions "
            "--no-context-files "
            "--no-prompt-templates "
            "--no-themes "
            f"{cli_args} "
            "-p "
            "\"$prompt\" "
            f"> {shlex.quote(iter_log_path)} 2>&1"
        )

        if time_limit_seconds:
            pi_exec = (
                f"timeout --kill-after={int(self.timeout_kill_after)}s "
                f"{int(time_limit_seconds)}s bash -lc {shlex.quote(pi_exec)}"
            )
        else:
            pi_exec = f"bash -lc {shlex.quote(pi_exec)}"

        script = "\n".join(
            [
                "set -euo pipefail",
                f"export PI_CODING_AGENT_DIR={PI_HOME_DIR}",
                f"export PI_CODING_AGENT_SESSION_DIR={PI_SESSIONS_DIR}",
                "export HOME=/home",
                f"export PATH={PI_NODE_HOME}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/conda/bin",
                "export NO_PROXY=127.0.0.1,localhost,::1",
                "export no_proxy=127.0.0.1,localhost,::1",
                f"mkdir -p {shlex.quote(loop_dir)} {shlex.quote(PI_SESSIONS_DIR)}",
                f"rm -f {shlex.quote(last_message_path)}",
                f"cd {shlex.quote(SUBMISSION_DIR)}",
                "git init >/dev/null 2>&1 || true",
                "git config user.email paperbench-pi@example.invalid >/dev/null 2>&1 || true",
                "git config user.name 'PaperBench Pi' >/dev/null 2>&1 || true",
                "set +e",
                pi_exec,
                "pi_exit=$?",
                "set -e",
                f"cat {shlex.quote(iter_log_path)} >> {shlex.quote(LOGS_DIR)}/agent.log 2>/dev/null || true",
                "python3 - "
                f"{int(iteration)} "
                f"{shlex.quote(self.loop_completion_promise)} "
                f"{shlex.quote(last_message_path)} "
                f"{shlex.quote(iter_log_path)} "
                f"{shlex.quote(summary_path)} "
                "\"$pi_exit\" <<'PY'",
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
                "raw_log = iter_log_path.read_text(encoding='utf-8', errors='replace') if iter_log_path.exists() else ''",
                "session_id = ''",
                "last_message = ''",
                "last_error = ''",
                "json_events = 0",
                "def extract_text(message):",
                "    if not isinstance(message, dict):",
                "        return ''",
                "    content = message.get('content')",
                "    pieces = []",
                "    if isinstance(content, str):",
                "        pieces.append(content)",
                "    elif isinstance(content, list):",
                "        for item in content:",
                "            if isinstance(item, dict) and item.get('type') in {'text', 'output_text'} and isinstance(item.get('text'), str):",
                "                pieces.append(item['text'])",
                "    return '\\n'.join(piece for piece in pieces if piece)",
                "for line in raw_log.splitlines():",
                "    try:",
                "        event = json.loads(line)",
                "    except Exception:",
                "        continue",
                "    if not isinstance(event, dict):",
                "        continue",
                "    json_events += 1",
                "    if event.get('type') == 'session' and isinstance(event.get('id'), str):",
                "        session_id = event['id']",
                "    message = event.get('message')",
                "    if isinstance(message, dict) and message.get('role') == 'assistant':",
                "        text = extract_text(message)",
                "        if text:",
                "            last_message = text",
                "        if isinstance(message.get('errorMessage'), str):",
                "            last_error = message['errorMessage']",
                "    if event.get('type') == 'agent_end':",
                "        messages = event.get('messages')",
                "        if isinstance(messages, list):",
                "            for candidate in reversed(messages):",
                "                if isinstance(candidate, dict) and candidate.get('role') == 'assistant':",
                "                    text = extract_text(candidate)",
                "                    if text:",
                "                        last_message = text",
                "                        break",
                "if not last_error:",
                "    for line in reversed(raw_log.splitlines()):",
                "        if line and not line.lstrip().startswith('{'):",
                "            last_error = line[-1000:]",
                "            break",
                "last_message_path.write_text(last_message, encoding='utf-8')",
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
                "    'json_event_count': json_events,",
                "    'last_message_excerpt': last_message[-4000:],",
                "    'iter_log_excerpt': raw_log[-4000:],",
                "    'last_error_excerpt': last_error[-4000:],",
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
                f"Failed to parse Pi iteration summary for iteration {iteration}: {text[:2000]}"
            ) from exc
        if isinstance(payload, dict):
            await self._write_remote_json(
                computer,
                f"{LOGS_DIR}/pi_session_state.json",
                {
                    "iteration": iteration,
                    "session_id": str(payload.get("session_id") or ""),
                    "exit_code": int(payload.get("exit_code") or 0),
                    "updated_at_epoch": int(time.time()),
                },
            )
            await self._write_remote_json(
                computer,
                f"{LOGS_DIR}/codex_session_state.json",
                {
                    "iteration": iteration,
                    "session_id": str(payload.get("session_id") or ""),
                    "exit_code": int(payload.get("exit_code") or 0),
                    "solver": self.shortname(),
                    "updated_at_epoch": int(time.time()),
                },
            )
        return payload if isinstance(payload, dict) else {}

    async def _finalize_submission(
        self, computer: ComputerInterface, final_exit_code: int, loop_state: dict[str, Any]
    ) -> dict[str, Any]:
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
        reproduce_sh_ok, reproduce_sh_message = await self._check_remote_submission_reproduce_sh(
            computer
        )
        finalization_status = self._build_submission_finalization_status(
            raw_exit_code=final_exit_code,
            reproduce_sh_ok=reproduce_sh_ok,
            reproduce_sh_message=reproduce_sh_message,
            loop_state=loop_state,
            completion_promise=self.loop_completion_promise,
            solver=self.shortname(),
        )
        state = {
            "iteration_count": int(finalization_status.get("iteration_count") or 0),
            "completed": bool(finalization_status.get("completed")),
            "raw_agent_completed": bool(finalization_status.get("raw_agent_completed")),
            "stop_reason": str(finalization_status.get("stop_reason") or ""),
            "completion_promise": self.loop_completion_promise,
            "last_iteration": finalization_status.get("last_iteration") or {},
            "solver": self.shortname(),
        }
        await self._write_remote_json(computer, f"{LOGS_DIR}/pi_loop_state.json", state)
        await self._write_remote_json(computer, f"{LOGS_DIR}/codex_loop_state.json", state)
        await self._write_remote_json(
            computer,
            f"{LOGS_DIR}/submission_finalization.json",
            finalization_status,
        )
        exit_status = (
            f"pi_exit={int(finalization_status.get('effective_exit_code') or 0)}\n"
            f"codex_exit={int(finalization_status.get('effective_exit_code') or 0)}\n"
            f"raw_agent_exit={int(finalization_status.get('raw_agent_exit_code') or 0)}\n"
            f"raw_agent_completed={1 if finalization_status.get('raw_agent_completed') else 0}\n"
            f"submission_reproduce_sh_ok={1 if finalization_status.get('submission_reproduce_sh_ok') else 0}\n"
            f"submission_reproduce_sh_message={finalization_status.get('submission_reproduce_sh_message') or ''}\n"
            f"completion_promise={self.loop_completion_promise}\n"
            f"completed={1 if finalization_status.get('completed') else 0}\n"
            f"iteration_count={int(finalization_status.get('iteration_count') or 0)}\n"
            f"stop_reason={finalization_status.get('stop_reason') or ''}\n"
        ).encode("utf-8")
        await computer.upload(exit_status, f"{LOGS_DIR}/pi_exit_status.txt")
        await computer.upload(exit_status, f"{LOGS_DIR}/codex_exit_status.txt")
        return finalization_status

    @override
    async def _sync_rollout_artifacts_to_host(self, computer: ComputerInterface, task: PBTask) -> dict[str, object]:
        artifacts = await super()._sync_rollout_artifacts_to_host(computer, task)
        trajectory_root = bf.join(task.run_dir, "logs", "pi_trajectory")
        preserved: dict[str, str] = {}

        prompt_host = bf.join(trajectory_root, "pi_prompt.txt")
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

        pi_home_archive_host = bf.join(trajectory_root, "pi_home.tar.gz")
        if await self._copy_remote_file_to_host_if_present(
            computer,
            remote_path=f"{LOGS_DIR}/pi_home.tar.gz",
            local_path=pi_home_archive_host,
            task=task,
        ):
            preserved["pi_home_archive"] = pi_home_archive_host

        pi_home_live_archive_host = bf.join(trajectory_root, "pi_home_live.tar.gz")
        if await self._copy_remote_file_to_host_if_present(
            computer,
            remote_path=f"{LOGS_DIR}/pi_home.tar.gz",
            local_path=pi_home_live_archive_host,
            task=task,
        ):
            preserved["pi_home_live_archive"] = pi_home_live_archive_host

        session_archive_host = bf.join(trajectory_root, "pi_sessions.tar.gz")
        if await self._copy_remote_directory_archive_to_host_if_present(
            computer,
            remote_dir=PI_SESSIONS_DIR,
            local_archive_path=session_archive_host,
        ):
            preserved["pi_sessions_archive"] = session_archive_host

        proxy_archive_host = bf.join(trajectory_root, "pi_cliproxyapi.tar.gz")
        if await self._copy_remote_directory_archive_to_host_if_present(
            computer,
            remote_dir=PI_PROXY_DIR,
            local_archive_path=proxy_archive_host,
        ):
            preserved["pi_cliproxyapi_archive"] = proxy_archive_host

        for remote_name in [
            "pi_session_state.json",
            "pi_loop_state.json",
            "pi_exit_status.txt",
            "pi_cliproxyapi.log",
            "pi_version.txt",
            "installed_pi_skills.txt",
            "codex_session_state.json",
            "codex_loop_state.json",
            "codex_exit_status.txt",
            "submission_finalization.json",
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

        manifest_path = bf.join(trajectory_root, "manifest.json")
        manifest = {
            "solver": self.shortname(),
            "full_trajectory_enabled": True,
            "prompt_source": (
                "paperbench/instructions/code_only_instructions.txt"
                if task.judge.code_only
                else "paperbench/instructions/instructions.txt"
            ),
            "pi_model": self.pi_model,
            "pi_provider": self.pi_provider,
            "pi_thinking": self.pi_thinking,
            "relay": {
                "enabled": self._pi_use_relay(),
                "tool": "CLIProxyAPI" if self._pi_use_relay() else None,
                "port": self.pi_relay_port if self._pi_use_relay() else None,
                "payload_override": (
                    "instructions U+200B for openai-response protocol"
                    if self._pi_use_relay()
                    else None
                ),
            },
            "aggregated_agent_log": bf.join(task.run_dir, "agent.log"),
            "preserved_artifacts": preserved,
        }
        if not bf.exists(trajectory_root):
            bf.makedirs(trajectory_root)
        bf.write_bytes(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
        artifacts["full_trajectory_manifest_exists"] = True
        artifacts["full_trajectory_manifest_path"] = manifest_path

        pi_exit_host_path = bf.join(task.run_dir, "logs", "pi_exit_status.txt")
        if bf.exists(pi_exit_host_path):
            artifacts["pi_exit_status_exists"] = True
            try:
                with bf.BlobFile(pi_exit_host_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("pi_exit="):
                            artifacts["codex_exit_code"] = int(line.split("=", 1)[1])
                            break
            except Exception:
                pass
        return artifacts
