from __future__ import annotations

import json
import importlib.util
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

import blobfile as bf
import pytest
from paperbench.grade import JudgeOutput
from paperbench.judge.graded_task_node import GradedTaskNode
from paperbench.nano.reproduction_success import reproduction_payload_succeeded
from paperbench.nano.structs import AgentOutput, PaperBenchResult, ReproductionMetadata
from paperbench.nano.utils import gather_eval_runs
from paperbench.nano.task import PBTask
from paperbench.nano.structs import JudgeConfig, ReproductionConfig
from paperbench.monitor.monitor import BasicMonitor
from paperbench.solvers.codexagent.solver import (
    CodexAgentSolver,
    SUBMISSION_FINALIZATION_FAILURE_EXIT_CODE,
)
from paperbench.solvers.utils import check_for_existing_run


def _load_official_case_scheduler_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "sota"
        / "scripts"
        / "official_case_scheduler.py"
    )
    spec = importlib.util.spec_from_file_location("official_case_scheduler", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reproduction_succeeded_requires_zero_exit_code() -> None:
    ok = ReproductionMetadata(
        is_valid_git_repo=True,
        git_log="x",
        repro_script_exists=True,
        files_before_reproduce="a",
        files_after_reproduce="b",
        timedout=False,
        repro_log="ok",
        repro_exit_code=0,
        executed_submission="submission_executed.tar.gz",
    )
    fail = ReproductionMetadata(
        is_valid_git_repo=True,
        git_log="x",
        repro_script_exists=True,
        files_before_reproduce="a",
        files_after_reproduce="b",
        timedout=False,
        repro_log="ok",
        repro_exit_code=1,
        executed_submission="submission_executed.tar.gz",
    )

    assert ok.reproduction_succeeded() is True
    assert fail.reproduction_succeeded() is False


def test_reproduction_succeeded_rejects_terminal_failure_even_with_zero_exit_code() -> None:
    fail = ReproductionMetadata(
        is_valid_git_repo=True,
        git_log="x",
        repro_script_exists=True,
        files_before_reproduce="a",
        files_after_reproduce="b",
        timedout=False,
        repro_log=(
            "Downloading data\n"
            "Traceback (most recent call last):\n"
            "RuntimeError: Failed to load Hugging Face dataset\n"
        ),
        repro_exit_code=0,
        executed_submission="submission_executed.tar.gz",
    )

    assert fail.reproduction_succeeded() is False


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {
                "repro_script_exists": True,
                "timedout": False,
                "repro_exit_code": 0,
                "executed_submission": "submission_executed.tar.gz",
                "repro_log": (
                    "Traceback (most recent call last):\n"
                    "RuntimeError: something optional failed\n"
                    "Finished. Outputs are in outputs\n"
                    "Reproduction complete. Outputs written to outputs\n"
                ),
            },
            True,
        ),
        (
            {
                "repro_script_exists": True,
                "timedout": False,
                "repro_exit_code": 0,
                "executed_submission": "submission_executed.tar.gz",
                "repro_log": (
                    "Traceback (most recent call last):\n"
                    "OSError: Can't load the model for 'microsoft/deberta-v3-base'\n"
                ),
            },
            False,
        ),
        (
            {
                "repro_script_exists": True,
                "timedout": False,
                "repro_exit_code": 0,
                "executed_submission": "submission_executed.tar.gz",
                "repro_log": (
                    "AttributeError: module 'torchvision.models' has no attribute 'Resnet18_Weights'\n"
                    "Full GPU reproduction failed with status 1; running explicit sanity fallback.\n"
                ),
            },
            False,
        ),
        (
            {
                "repro_script_exists": True,
                "timedout": False,
                "repro_exit_code": 0,
                "executed_submission": "submission_executed.tar.gz",
                "repro_log": (
                    "../aten/src/ATen/native/cuda/Loss.cu:250: "
                    "Assertion `t >= 0 && t < n_classes` failed.\n"
                    "terminate called after throwing an instance of 'c10::Error'\n"
                    "what(): CUDA error: device-side assert triggered\n"
                    "reproduce.sh: line 88:  1064 Aborted                 (core dumped)\n"
                ),
            },
            False,
        ),
    ],
)
def test_reproduction_payload_succeeded_uses_end_of_log_semantics(
    payload: dict[str, object],
    expected: bool,
) -> None:
    assert reproduction_payload_succeeded(payload) is expected


def test_judge_output_to_dict_includes_success() -> None:
    node = GradedTaskNode(
        id="leaf",
        requirements="x",
        weight=1,
        score=1.0,
        valid_score=True,
        task_category="Code Development",
        explanation="ok",
        judge_metadata=None,
        sub_tasks=[],
    )
    output = JudgeOutput(
        judge_type="simple",
        score=1.0,
        num_leaf_nodes=1,
        num_invalid_leaf_nodes=0,
        graded_at="now",
        graded_task_tree=node,
    )

    payload = output.to_dict()

    assert payload["success"] is True


async def test_check_for_existing_run_rejects_failed_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        submission_dir = run_dir / "submissions" / "2026-01-01T00-00-00-GMT"
        submission_dir.mkdir(parents=True, exist_ok=True)
        (submission_dir / "submission.tar.gz").write_bytes(b"x")
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "status": "done",
                    "created_at": 1,
                    "agent_finished_at": 2,
                    "last_updated": 2,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "run_id": "rid",
                    "time_start": 0,
                    "time_end": 1,
                    "error_msg": "Codex command exited with 1",
                    "runtime_in_seconds": 1,
                    "status_exists": True,
                }
            ),
            encoding="utf-8",
        )

        task = PBTask(
            question_id="q",
            attempt_id=0,
            prompt=[{"role": "user", "content": "x"}],
            paper_id="rice",
            run_id="rid",
            run_group_id="g",
            run_dir=str(run_dir),
            runs_dir=str(run_dir),
            target_duration_hr=None,
            reproduction=ReproductionConfig(skip_reproduction=True),
            judge=JudgeConfig(grade=False),
            monitor_config=BasicMonitor.Config(),
            save_cluster_output_to_host=False,
        )

        result = await check_for_existing_run(task)

        assert result is None
        assert task.skipped_rollout is False


def test_agent_output_rollout_succeeded_false_on_error() -> None:
    output = AgentOutput(
        run_id="rid",
        time_start=0,
        time_end=1,
        error_msg="boom",
        runtime_in_seconds=1,
        status_exists=True,
    )

    assert output.rollout_succeeded() is False


async def test_ensure_files_available_for_grading_preserves_rollout_status(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        status_path = run_dir / "status.json"
        status_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "created_at": 123,
                    "agent_finished_at": 456,
                    "last_updated": 456,
                }
            ),
            encoding="utf-8",
        )

        task = PBTask(
            question_id="q",
            attempt_id=0,
            prompt=[{"role": "user", "content": "x"}],
            paper_id="rice",
            run_id="rid",
            run_group_id="g",
            run_dir=str(run_dir),
            runs_dir=str(run_dir),
            target_duration_hr=None,
            reproduction=ReproductionConfig(skip_reproduction=True),
            judge=JudgeConfig(grade=False),
            monitor_config=BasicMonitor.Config(),
            save_cluster_output_to_host=False,
        )

        async def _noop_upload_heavy_logs(**_kwargs):
            return None

        monkeypatch.setattr(
            "paperbench.nano.task.upload_heavy_logs",
            _noop_upload_heavy_logs,
        )

        class DummyComputer:
            async def upload(self, *_args, **_kwargs):
                return None

        await task._ensure_files_available_for_grading(DummyComputer())

        with bf.BlobFile(str(status_path), "r") as f:
            payload = json.loads(f.read())
        assert payload["status"] == "failed"
        assert payload["created_at"] == 123
        assert payload["agent_finished_at"] == 456


def test_scheduler_result_summary_rejects_invalid_judged_run() -> None:
    module = _load_official_case_scheduler_module()

    summary = module.latest_result_summary("case18")

    assert summary["has_grade"] is True
    assert summary["judge_success"] is True
    assert summary["looks_complete"] is False


def test_scheduler_active_cases_prioritizes_live_processes_over_stale_finished_flag() -> None:
    module = _load_official_case_scheduler_module()

    module.load_json = lambda _path: {"official_finished": True, "official_wrapper_pid": 12345}
    module.load_launch_record = lambda _case_id: {"launcher_pid": 0}
    module.pid_alive = lambda pid: str(pid) == "12345"
    module.case_root = lambda _case_id: Path("/definitely/missing")

    active = module.active_cases([{"case_id": "caseX", "paper_id": "paperX"}])

    assert "caseX" in active


def test_scheduler_assigned_gpu_ids_preserves_gpu_for_live_launcher_despite_finished_flag() -> None:
    module = _load_official_case_scheduler_module()

    module.load_launch_record = lambda _case_id: {"gpu_id": "7", "launcher_pid": 555}
    module.load_json = lambda _path: {"official_finished": True, "official_wrapper_pid": 0}
    module.pid_alive = lambda pid: str(pid) == "555"
    module.case_root = lambda _case_id: Path("/definitely/missing")

    assigned = module.assigned_gpu_ids([{"case_id": "caseX", "paper_id": "paperX"}])

    assert assigned == {"caseX": "7"}


def test_scheduler_reconciles_stale_finished_status_for_invalid_run(tmp_path) -> None:
    module = _load_official_case_scheduler_module()

    case_root = tmp_path / "caseX"
    monitoring = case_root / "monitoring"
    monitoring.mkdir(parents=True, exist_ok=True)
    status_path = monitoring / "supervisor_status.json"
    status_path.write_text(
        json.dumps(
            {
                "case_id": "caseX",
                "phase": "paperbench_rollout_finished",
                "official_finished": True,
                "official_wrapper_pid": 0,
            }
        ),
        encoding="utf-8",
    )

    module.case_root = lambda _case_id: case_root
    module.supervisor_status_path = lambda _case_id: status_path
    module.load_json = lambda path: json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
    module.load_launch_record = lambda _case_id: {}
    module.pid_alive = lambda _pid: False
    module.latest_result_summary = lambda _case_id: {
        "looks_complete": False,
        "score": 0.0,
        "rollout_succeeded": False,
        "reproduction_succeeded": False,
        "judge_success": False,
    }
    reconciled_events = []
    module.event = lambda name, details: reconciled_events.append((name, details))

    payload = module.reconcile_supervisor_status("caseX", dry_run=False)

    assert payload is not None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["official_finished"] is False
    assert status["official_failed"] is True
    assert status["phase"] == "paperbench_rollout_invalidated"
    assert reconciled_events and reconciled_events[0][0] == "scheduler_reconciled_supervisor_status"


def _judge_output(score: float = 1.0) -> JudgeOutput:
    node = GradedTaskNode(
        id="leaf",
        requirements="x",
        weight=1,
        score=score,
        valid_score=True,
        task_category="Code Development",
        explanation="ok",
        judge_metadata=None,
        sub_tasks=[],
    )
    return JudgeOutput(
        judge_type="simple",
        score=score,
        num_leaf_nodes=1,
        num_invalid_leaf_nodes=0,
        graded_at="now",
        graded_task_tree=node,
    )


def _agent_output() -> AgentOutput:
    return AgentOutput(
        run_id="rid",
        time_start=0.0,
        time_end=1.0,
        runtime_in_seconds=1.0,
        error_msg=None,
        status_exists=True,
        agent_log_exists=True,
    )


def test_gather_eval_runs_accepts_skip_reproduction_results() -> None:
    result = PaperBenchResult(
        paper_id="rice",
        run_id="rid",
        submission_exists=True,
        skipped_reproduction=True,
        code_only=False,
        resources_provided=False,
        agent_output=_agent_output(),
        judge_output=_judge_output(),
        reproduction_metadata=None,
    )

    eval_runs = gather_eval_runs([result], n_runs=1)

    assert "rice" in eval_runs[0].paper_evaluations


def test_gather_eval_runs_accepts_code_only_results() -> None:
    result = PaperBenchResult(
        paper_id="rice",
        run_id="rid",
        submission_exists=True,
        skipped_reproduction=False,
        code_only=True,
        resources_provided=False,
        agent_output=_agent_output(),
        judge_output=_judge_output(),
        reproduction_metadata=None,
    )

    eval_runs = gather_eval_runs([result], n_runs=1)

    assert "rice" in eval_runs[0].paper_evaluations


def test_codex_iteration_command_uses_fresh_exec_not_resume() -> None:
    solver = CodexAgentSolver()

    class DummyComputer:
        captured: str | None = None

        async def upload(self, *_args, **_kwargs):
            return None

        async def send_shell_command(self, cmd: str):
            self.captured = cmd
            payload = {"iteration": 1, "exit_code": 0, "completed": False}
            return SimpleNamespace(
                unicode_output_best_effort=json.dumps(payload),
                exit_code=0,
                output=json.dumps(payload).encode("utf-8"),
            )

    computer = DummyComputer()

    import asyncio

    asyncio.run(
        solver._run_codex_iteration(
            computer=computer,
            prompt_path="/tmp/prompt.txt",
            iteration=1,
            time_limit_seconds=None,
        )
    )

    assert computer.captured is not None
    assert "codex exec resume" not in computer.captured
    assert "codex exec " in computer.captured


def test_default_codex_solver_has_loop_disabled() -> None:
    solver = CodexAgentSolver()

    assert solver.loop_enabled is False


def test_submission_finalization_fails_clean_exit_without_reproduce_sh() -> None:
    status = CodexAgentSolver._build_submission_finalization_status(
        raw_exit_code=0,
        reproduce_sh_ok=False,
        reproduce_sh_message="submission/reproduce.sh not found",
        loop_state={
            "completed": True,
            "iteration_count": 1,
            "stop_reason": "single_run_finished",
        },
        completion_promise="PAPERBENCH_COMPLETE",
        solver="codexagent",
    )

    assert status["raw_agent_exit_code"] == 0
    assert status["raw_agent_completed"] is True
    assert status["effective_exit_code"] == SUBMISSION_FINALIZATION_FAILURE_EXIT_CODE
    assert status["completed"] is False
    assert status["submission_reproduce_sh_ok"] is False
    assert "reproduce.sh not found" in CodexAgentSolver._submission_finalization_error(status)


def test_submission_finalization_preserves_agent_failure_code() -> None:
    status = CodexAgentSolver._build_submission_finalization_status(
        raw_exit_code=124,
        reproduce_sh_ok=False,
        reproduce_sh_message="submission/reproduce.sh not found",
        loop_state={
            "completed": False,
            "iteration_count": 1,
            "stop_reason": "single_run_timeout",
        },
        completion_promise="PAPERBENCH_COMPLETE",
        solver="piagent-vanilla",
    )

    assert status["effective_exit_code"] == 124
    assert status["completed"] is False


def test_submission_finalization_salvages_timeout_with_reproduce_sh() -> None:
    status = CodexAgentSolver._build_submission_finalization_status(
        raw_exit_code=124,
        reproduce_sh_ok=True,
        reproduce_sh_message="",
        loop_state={
            "completed": False,
            "iteration_count": 1,
            "stop_reason": "single_run_timeout",
        },
        completion_promise="PAPERBENCH_COMPLETE",
        solver="piagent-vanilla",
    )

    assert status["raw_agent_exit_code"] == 124
    assert status["effective_exit_code"] == 0
    assert status["salvaged_timeout_with_reproduce_sh"] is True
    assert status["completed"] is False
    assert CodexAgentSolver._submission_finalization_error(status) is None
    assert CodexAgentSolver._finalized_effective_exit_code(status, 124) == 0


@pytest.mark.parametrize(
    "last_error_excerpt",
    [
        'OpenAI API error (500): {"message":"Post \\"https://www.su8.codes/v1/chat/completions\\": EOF","type":"server_error","code":"internal_server_error"}',
        "Error Code internal_server_error: stream error: stream ID 7; INTERNAL_ERROR; received from peer",
        '<title>su8.codes | 520: Web server is returning an unknown error</title><span class="code-label">Error code 520</span>',
    ],
)
def test_codex_retryable_iteration_failure_covers_pi_transport_errors(
    last_error_excerpt: str,
) -> None:
    assert CodexAgentSolver._is_retryable_iteration_failure(
        {"last_error_excerpt": last_error_excerpt}
    )


async def test_grade_explicit_checkpoint_writes_separate_artifact(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        submission_dir = run_dir / "submissions" / "2026-01-01T00-00-00-GMT"
        submission_dir.mkdir(parents=True, exist_ok=True)
        submission_path = submission_dir / "submission.tar.gz"
        submission_path.write_bytes(b"x")

        task = PBTask(
            question_id="q",
            attempt_id=0,
            prompt=[{"role": "user", "content": "x"}],
            paper_id="rice",
            run_id="rid",
            run_group_id="g",
            run_dir=str(run_dir),
            runs_dir=str(run_dir),
            target_duration_hr=None,
            reproduction=ReproductionConfig(skip_reproduction=True),
            judge=JudgeConfig(grade=False),
            monitor_config=BasicMonitor.Config(),
            save_cluster_output_to_host=False,
        )

        async def _fake_grade_submission_checkpoint(
            _submission_path: str,
            *,
            agent_output: AgentOutput | None,
            require_rollout_success: bool,
            run_monitor: bool,
        ):
            del require_rollout_success, run_monitor
            return task._build_grade(
                submission_exists=True,
                agent_output=agent_output,
                judge_output=None,
                reproduction_metadata=None,
                monitor_ran=False,
                monitor_result=None,
                grader_log="explicit checkpoint graded",
            )

        monkeypatch.setattr(task, "_grade_submission_checkpoint", _fake_grade_submission_checkpoint)

        output_path = run_dir / "intermediate_grades" / "iteration_0001_grade.json"
        grade = await task.grade_explicit_checkpoint(
            str(submission_path),
            grade_output_path=str(output_path),
            agent_output=_agent_output(),
        )

        assert grade.grader_log == "explicit checkpoint graded"
        assert output_path.exists()
        assert not (run_dir / "grade.json").exists()


def test_valid_submission_checkpoint_requires_reproduce_sh_for_reproduction() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        submission_dir = run_dir / "submissions" / "2026-01-01T00-00-00-GMT"
        submission_dir.mkdir(parents=True, exist_ok=True)
        submission_path = submission_dir / "submission.tar.gz"
        payload_root = run_dir / "payload"
        (payload_root / "submission").mkdir(parents=True)
        (payload_root / "submission" / "requirements.txt").write_text("numpy\n", encoding="utf-8")
        with tarfile.open(submission_path, "w:gz") as archive:
            archive.add(payload_root / "submission", arcname="submission")

        task = PBTask(
            question_id="q",
            attempt_id=0,
            prompt=[{"role": "user", "content": "x"}],
            paper_id="rice",
            run_id="rid",
            run_group_id="g",
            run_dir=str(run_dir),
            runs_dir=str(run_dir),
            target_duration_hr=None,
            reproduction=ReproductionConfig(skip_reproduction=False),
            judge=JudgeConfig(grade=False),
            monitor_config=BasicMonitor.Config(),
            save_cluster_output_to_host=False,
        )

        ok, message = task._valid_submission_checkpoint(str(submission_path), _agent_output())

        assert not ok
        assert "reproduce.sh not found" in message


def test_valid_submission_checkpoint_allows_missing_reproduce_sh_when_reproduction_skipped() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        submission_dir = run_dir / "submissions" / "2026-01-01T00-00-00-GMT"
        submission_dir.mkdir(parents=True, exist_ok=True)
        submission_path = submission_dir / "submission.tar.gz"
        payload_root = run_dir / "payload"
        (payload_root / "submission").mkdir(parents=True)
        (payload_root / "submission" / "requirements.txt").write_text("numpy\n", encoding="utf-8")
        with tarfile.open(submission_path, "w:gz") as archive:
            archive.add(payload_root / "submission", arcname="submission")

        task = PBTask(
            question_id="q",
            attempt_id=0,
            prompt=[{"role": "user", "content": "x"}],
            paper_id="rice",
            run_id="rid",
            run_group_id="g",
            run_dir=str(run_dir),
            runs_dir=str(run_dir),
            target_duration_hr=None,
            reproduction=ReproductionConfig(skip_reproduction=True),
            judge=JudgeConfig(grade=False),
            monitor_config=BasicMonitor.Config(),
            save_cluster_output_to_host=False,
        )

        ok, message = task._valid_submission_checkpoint(str(submission_path), _agent_output())

        assert ok
        assert message == ""
