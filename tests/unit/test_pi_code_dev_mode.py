from __future__ import annotations

import json
from types import SimpleNamespace

from paperbench.solvers.codexagent.solver import (
    OFFICIAL_CODE_ONLY_PAPERBENCH_INSTRUCTIONS,
    OFFICIAL_PAPERBENCH_INSTRUCTIONS,
)
from paperbench.solvers.piagent.solver import _sglang_qwen_models_json
from paperbench.solvers.piagent_vanilla.solver import PiVanillaAgentSolver


def test_pi_vanilla_prompt_switches_to_code_only_instructions() -> None:
    solver = PiVanillaAgentSolver()
    code_only_task = SimpleNamespace(
        paper_id="demo-paper",
        judge=SimpleNamespace(code_only=True),
    )
    vanilla_task = SimpleNamespace(
        paper_id="demo-paper",
        judge=SimpleNamespace(code_only=False),
    )

    assert solver._prompt(code_only_task) == OFFICIAL_CODE_ONLY_PAPERBENCH_INSTRUCTIONS
    assert solver._prompt(vanilla_task) == OFFICIAL_PAPERBENCH_INSTRUCTIONS


def test_submission_finalization_status_treats_code_only_as_no_reproduce_requirement() -> None:
    solver = PiVanillaAgentSolver()
    status = solver._build_submission_finalization_status(
        raw_exit_code=0,
        reproduce_sh_ok=False,
        reproduce_sh_message="submission/reproduce.sh not found",
        loop_state={
            "completed": True,
            "iteration_count": 3,
            "last_iteration": {"exit_code": 0},
            "stop_reason": "single_run_finished",
        },
        completion_promise="PAPERBENCH_COMPLETE",
        solver=solver.shortname(),
        code_only=True,
    )

    assert status["submission_reproduce_sh_required"] is False
    assert status["submission_reproduce_sh_ok"] is True
    assert status["submission_reproduce_sh_message"] == ""
    assert status["effective_exit_code"] == 0

    non_code_only_status = solver._build_submission_finalization_status(
        raw_exit_code=0,
        reproduce_sh_ok=False,
        reproduce_sh_message="submission/reproduce.sh not found",
        loop_state={
            "completed": True,
            "iteration_count": 3,
            "last_iteration": {"exit_code": 0},
            "stop_reason": "single_run_finished",
        },
        completion_promise="PAPERBENCH_COMPLETE",
        solver=solver.shortname(),
        code_only=False,
    )

    assert non_code_only_status["submission_reproduce_sh_required"] is True
    assert non_code_only_status["submission_reproduce_sh_ok"] is False
    assert non_code_only_status["effective_exit_code"] == 65


def test_sglang_qwen_models_json_matches_expected_shape() -> None:
    payload = json.loads(
        _sglang_qwen_models_json(
            model_id="qwen35_35b_a3b",
            base_url="http://arex.autoresearch.eval.baai.ac.cn/",
            api_key="inspectai",
        )
    )

    provider = payload["providers"]["sglang-qwen"]
    assert provider["baseUrl"] == "http://arex.autoresearch.eval.baai.ac.cn/"
    assert provider["api"] == "openai-completions"
    assert provider["apiKey"] == "inspectai"
    assert provider["authHeader"] is True
    assert provider["models"][0]["id"] == "qwen35_35b_a3b"
    assert provider["models"][0]["name"] == "Qwen3.5-35B-A3B (SGLang)"
    assert provider["models"][0]["compat"]["thinkingFormat"] == "qwen-chat-template"
