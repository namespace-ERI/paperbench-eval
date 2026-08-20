from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import blobfile as bf
import structlog.stdlib

from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerConfiguration,
    ComputerInterface,
    ComputerRuntime,
)
from paperbench.computer_utils import (
    put_submission_in_computer,
    start_computer_with_retry,
)
from paperbench.infra.alcatraz import tar_and_extract_from_computer
from paperbench.nano.structs import PBRuntimeConfig, ReproductionMetadata, ReproScriptRunOutcome
from paperbench.utils import get_agents_env_vars

logger = structlog.stdlib.get_logger(component=__name__)


SUBMISSION_TRANSIENT_CLEANUP = (
    "rm -rf "
    "venv .venv .venv-paperbench .paperbench_venv env "
    ".pytest_cache .mypy_cache .ruff_cache .tox .nox __pycache__"
    " && find . -type d \\( "
    "-name __pycache__ -o "
    "-name .pytest_cache -o "
    "-name .mypy_cache -o "
    "-name .ruff_cache -o "
    "-name .tox -o "
    "-name .nox -o "
    "-name .venv-paperbench -o "
    "-name .paperbench_venv -o "
    "-name .venv -o "
    "-name venv -o "
    "-name env "
    "\\) -prune -exec rm -rf {} +"
)

GRAPHICS_RUNTIME_BOOTSTRAP = (
    "if [ \"${PAPERBENCH_SKIP_GRAPHICS_RUNTIME_BOOTSTRAP:-0}\" != \"1\" ]; then"
    " if ! ldconfig -p 2>/dev/null | grep -q \"libEGL\\\\.so\\\\.1\""
    " || ! ldconfig -p 2>/dev/null | grep -q \"libGL\\\\.so\\\\.1\"; then"
    " echo \"[paperbench] missing GL/EGL runtime libraries; attempting bootstrap\";"
    " export DEBIAN_FRONTEND=noninteractive;"
    " if command -v sudo >/dev/null 2>&1 && [ \"$(id -u)\" -ne 0 ]; then SUDO=sudo; else SUDO=; fi;"
    " $SUDO apt-get update || true;"
    " $SUDO apt-get install -y --no-install-recommends"
    " libegl1 libgl1 libgles2 libglvnd0 libgl1-mesa-dri libosmesa6 mesa-utils || true;"
    " fi;"
    " else echo \"[paperbench] skipping graphics runtime bootstrap by environment\"; fi;"
    " python3 -c \"import ctypes.util; "
    "print(\\\"[paperbench] libEGL=%s\\\" % ctypes.util.find_library(\\\"EGL\\\")); "
    "print(\\\"[paperbench] libGL=%s\\\" % ctypes.util.find_library(\\\"GL\\\"))\""
)

TRANSIENT_REPRO_NETWORK_MARKERS = (
    "requests.exceptions.ProxyError",
    "urllib3.exceptions.ProxyError",
    "RemoteDisconnected",
    "Remote end closed connection without response",
    "ReadTimeoutError",
    "Read timed out",
    "The read operation timed out",
    "SSLEOFError",
    "SSLError",
    "UNEXPECTED_EOF_WHILE_READING",
    "IncompleteRead",
    "ProtocolError",
    "Connection broken:",
    "Connection refused",
    "Connection reset by peer",
    "Temporary failure in name resolution",
    "Max retries exceeded with url",
    "Could not install packages due to an OSError",
    "HTTP Error 502",
    "502 Bad Gateway",
    "Retryable HTTP 502",
    "HTTPSConnectionPool(host='huggingface.co'",
    "HTTPSConnectionPool(host=\"huggingface.co\"",
    "HTTPSConnectionPool(host='download.pytorch.org'",
    "HTTPSConnectionPool(host='download-r2.pytorch.org'",
    "HTTPSConnectionPool(host='pypi.org'",
    "HTTPSConnectionPool(host='files.pythonhosted.org'",
    "cas-bridge.xethub.hf.co",
    "xet-bridge-us",
)


def _looks_like_transient_repro_network_failure(*texts: str) -> bool:
    haystack = "\n".join(texts)
    return any(marker in haystack for marker in TRANSIENT_REPRO_NETWORK_MARKERS)


async def _cleanup_submission_transients(
    computer: ComputerInterface,
    submission_path: Path,
) -> None:
    await computer.send_shell_command(
        f"bash -c 'cd {submission_path} && {SUBMISSION_TRANSIENT_CLEANUP}'"
    )


async def run_reproduce_script(
    computer: ComputerInterface,
    submission_path: Path,
    run_group_id: str,
    runs_dir: str,
    run_id: str,
    timeout: float | None,
    use_py3_11: bool = False,  # defaults to 3.12
    make_venv: bool = False,
) -> ReproScriptRunOutcome:
    """
    Runs the reproduce.sh script in the submission directory.
    Optionally sets python3 to python3.11 and/or creates a virtual environment.
    """

    ctx_logger = logger.bind(
        run_group_id=run_group_id, runs_dir=runs_dir, run_id=run_id, destinations=["run"]
    )
    await _cleanup_submission_transients(computer, submission_path)
    # log rough start time of reproduction to a file
    await computer.check_shell_command(f"date +%s > {submission_path}/reproduce.log.creation_time")

    def build_reproduce_command(*, append_log: bool, attempt_number: int) -> str:
        cmd_str = f"bash -o pipefail -c 'cd {submission_path}"
        # remove existing venvs since they contain broken symlinks to python3
        cmd_str += f" && {SUBMISSION_TRANSIENT_CLEANUP}"
        cmd_str += f" && {GRAPHICS_RUNTIME_BOOTSTRAP}"
        if use_py3_11:
            cmd_str += (
                " && if command -v python3.11 >/dev/null 2>&1; then"
                " if update-alternatives --query python3 >/dev/null 2>&1; then"
                " update-alternatives --set python3 $(command -v python3.11) || true;"
                " else echo python3.11 present but no python3 update-alternatives entry;"
                " fi;"
                " else echo python3.11 requested but not available, letting reproduce.sh choose interpreter;"
                " fi"
            )
        if make_venv:
            cmd_str += (
                " && rm -rf /tmp/paperbench_reproduce_venv"
                " && python3 -m venv /tmp/paperbench_reproduce_venv"
                " && source /tmp/paperbench_reproduce_venv/bin/activate"
            )
        if append_log:
            cmd_str += (
                " && printf '\\n[paperbench] transient network retry attempt "
                f"{attempt_number}\\n' >> reproduce.log"
            )
        tee_args = "-a reproduce.log" if append_log else "reproduce.log"
        cmd_str += f" && bash reproduce.sh 2>&1 | tee {tee_args}'"
        return cmd_str

    retry_limit = max(
        1,
        int(os.getenv("PAPERBENCH_REPRO_INPLACE_NETWORK_RETRY_MAX_ATTEMPTS", "5")),
    )
    retry_sleep_seconds = max(
        0.0,
        float(os.getenv("PAPERBENCH_REPRO_INPLACE_NETWORK_RETRY_SLEEP_SECONDS", "20")),
    )
    deadline = None if timeout is None else time.time() + timeout
    repro_start_time = time.time()
    timedout = False
    repro_exit_code = None
    repro_log = "reproduce.log file not found"

    await computer.send_shell_command(f"rm -f {submission_path}/reproduce.log")

    for attempt_number in range(1, retry_limit + 1):
        remaining_timeout = None
        if deadline is not None:
            remaining_timeout = max(1.0, deadline - time.time())

        output_text = ""
        try:
            result = await asyncio.wait_for(
                computer.send_shell_command(
                    build_reproduce_command(
                        append_log=attempt_number > 1,
                        attempt_number=attempt_number,
                    )
                ),
                timeout=remaining_timeout,
            )
            repro_exit_code = result.exit_code
            output_text = result.output.decode("utf-8", errors="replace")
            ctx_logger.info(
                f"Reproduction script output (attempt {attempt_number}/{retry_limit}): {output_text}"
            )
        except asyncio.TimeoutError:
            timedout = True
            repro_exit_code = None

        result = await computer.send_shell_command(f"cat {submission_path}/reproduce.log")
        if result.exit_code != 0:
            repro_log = "reproduce.log file not found"
        else:
            repro_log = result.output.decode("utf-8", errors="replace")

        if timedout or repro_exit_code == 0:
            break

        transient_failure = _looks_like_transient_repro_network_failure(output_text, repro_log)
        if not transient_failure or attempt_number >= retry_limit:
            break

        ctx_logger.info(
            "Transient network failure detected during reproduction; retrying in-place",
            attempt_number=attempt_number,
            retry_limit=retry_limit,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        if deadline is not None and time.time() + retry_sleep_seconds >= deadline:
            break
        if retry_sleep_seconds > 0:
            await asyncio.sleep(retry_sleep_seconds)

    repro_execution_time = time.time() - repro_start_time

    await _cleanup_submission_transients(computer, submission_path)

    return ReproScriptRunOutcome(repro_execution_time, timedout, repro_log, repro_exit_code)


async def reproduce(
    computer: ComputerInterface,
    submission_path: Path,
    run_group_id: str,
    runs_dir: str,
    run_id: str,
    timeout: float | None = None,
    use_py3_11: bool = False,
    make_venv: bool = False,
) -> ReproductionMetadata:
    """
    args:
        computer: ComputerInterface on which the reproduce.sh script will be run
        submission_path: Path to the submission directory
        timeout: (optional) Timeout for the reproduce.sh script
        use_py3_11: (optional) Whether to switch python3 to python3.11 before running
        make_venv: (optional) Whether to create and use a virtualenv before running
    """
    ctx_logger = logger.bind(
        run_group_id=run_group_id, runs_dir=runs_dir, run_id=run_id, destinations=["run"]
    )
    # get git history for interest
    cmd_str = f"bash -c 'cd {submission_path} && git --no-pager log'"
    result = await computer.send_shell_command(cmd_str)
    exit_code, git_log = result.exit_code, result.output.decode("utf-8")
    # fine if this fails (e.g. if not a git repo), we just store that error message if so
    is_valid_git_repo = True
    if exit_code != 0:
        # typical errors: "not a git repository" or "does not have any commits"
        is_valid_git_repo = False

    result = await computer.check_shell_command(f"ls -la {submission_path}")
    files_before_reproduce = result.output.decode("utf-8")

    # does reproduce.sh exist?
    reproduce_script_path = submission_path / "reproduce.sh"
    result = await computer.send_shell_command(f"ls {reproduce_script_path}")
    repro_script_exists = result.exit_code == 0
    ctx_logger.info(f"Reproduce script exists at {reproduce_script_path}: {repro_script_exists}")
    if not repro_script_exists:
        return ReproductionMetadata(
            is_valid_git_repo=is_valid_git_repo,
            git_log=git_log,
            repro_script_exists=repro_script_exists,
            repro_log="reproduce.sh not found",
            files_before_reproduce=files_before_reproduce,
            files_after_reproduce=files_before_reproduce,
            timedout=False,
        )

    # sometimes git complains about `detected dubious ownership in repository` due to mismatching file ownership
    cmd_str = f"bash -c 'git config --global --add safe.directory {submission_path}'"
    await computer.send_shell_command(cmd_str)

    repro_outcome = await run_reproduce_script(
        computer=computer,
        submission_path=submission_path,
        run_group_id=run_group_id,
        runs_dir=runs_dir,
        run_id=run_id,
        timeout=timeout,
        use_py3_11=use_py3_11,
        make_venv=make_venv,
    )

    result = await computer.check_shell_command(f"ls -la {submission_path}")
    files_after_reproduce = result.output.decode("utf-8")

    result = await computer.send_shell_command(f"cd {submission_path} && git status")
    git_status = result.output.decode("utf-8")

    return ReproductionMetadata(
        is_valid_git_repo=is_valid_git_repo,
        git_log=git_log,
        repro_script_exists=repro_script_exists,
        repro_execution_time=repro_outcome.repro_execution_time,
        repro_log=repro_outcome.repro_log,
        files_before_reproduce=files_before_reproduce,
        files_after_reproduce=files_after_reproduce,
        git_status_after_reproduce=git_status,
        timedout=repro_outcome.timedout,
        repro_exit_code=repro_outcome.repro_exit_code,
        # will populate retried_results and executed_submission later
    )


async def reproduce_on_computer(
    computer_runtime: ComputerRuntime,
    computer_config: ComputerConfiguration,
    runtime_config: PBRuntimeConfig,
    submission_path: str,
    run_group_id: str,
    runs_dir: str,
    run_id: str,
    run_dir: str,
    submission_cluster_path: Path = Path("/submission"),
    output_cluster_path: Path = Path("/output"),
    timeout: float | None = None,
    use_py3_11: bool = False,
    make_venv: bool = False,
) -> ReproductionMetadata:
    """
    Reproduce a single submission on a computer.

    This script will spin up a pb-reproducer container on the computer to do the following:
    - Download the submission
    - Run the reproduction script
    - Save and return the reproduction results
    """
    # we're ensuring the env variables defined in agent.env are set on the computer
    computer_config = computer_config.model_copy(
        update={"environment": {**computer_config.environment, **get_agents_env_vars()}}
    )
    ctx_logger = logger.bind(
        run_group_id=run_group_id, runs_dir=runs_dir, run_id=run_id, destinations=["run"]
    )
    async with start_computer_with_retry(
        computer_runtime, computer_config, runtime_config.exception_types
    ) as computer:
        time_start = time.time()

        await computer.check_shell_command(
            f"mkdir -p {output_cluster_path} {submission_cluster_path}"
        )

        # Step 1: Put submission to /submission
        await put_submission_in_computer(
            computer,
            submission_path,
            run_group_id,
            runs_dir,
            run_id,
        )

        # Step 2: Run reproduction
        repro_metadata = await reproduce(
            computer=computer,
            submission_path=submission_cluster_path,
            run_group_id=run_group_id,
            runs_dir=runs_dir,
            run_id=run_id,
            timeout=timeout,
            use_py3_11=use_py3_11,
            make_venv=make_venv,
        )

        # Step 3: Save the reproduced submission itself
        timestamp = Path(submission_path).parts[-2]
        upload_from = output_cluster_path / "submission_executed.tar.gz"
        upload_to = bf.join(run_dir, "submissions", timestamp, "submission_executed.tar.gz")
        await tar_and_extract_from_computer(
            computer=computer,
            dir_path_on_computer=submission_cluster_path,
            tar_path_on_computer=upload_from,
            tar_path_on_target=upload_to,
            run_group_id=run_group_id,
            runs_dir=runs_dir,
            run_id=run_id,
            max_file_size="10M",
        )

        ctx_logger.info(f"Reproduced dir has been written: {upload_to}")
        repro_metadata = replace(repro_metadata, executed_submission=upload_to)

        # Step 4: Save metadata
        path_to_output = submission_path.replace(".tar.gz", "_executed_metadata.json")
        bf.write_bytes(path_to_output, json.dumps(repro_metadata.to_dict()).encode("utf-8"))

        time_end = time.time()
        ctx_logger.info(f"Reproduction completed in {time_end - time_start:.2f} seconds.")

        return repro_metadata


async def reproduce_on_computer_with_salvaging(
    computer_runtime: ComputerRuntime,
    computer_config: ComputerConfiguration,
    runtime_config: PBRuntimeConfig,
    submission_path: str,
    run_group_id: str,
    runs_dir: str,
    run_id: str,
    run_dir: str,
    submission_cluster_path: Path = Path("/submission"),
    output_cluster_path: Path = Path("/output"),
    timeout: float | None = None,
    retry_threshold: float = 0,
    default_use_py3_11: bool = False,
    default_make_venv: bool = False,
) -> ReproductionMetadata:
    """
    Reproduce a single submission on a computer,
    salvaging reproduce attempts by retrying with slightly different configurations.
    """
    valid_threshold = True if timeout is None else retry_threshold < timeout
    retries_enabled = retry_threshold > 0 and valid_threshold

    retry_options = [{"use_py3_11": default_use_py3_11, "make_venv": default_make_venv}]
    if retries_enabled:
        salvage_options = [
            {"use_py3_11": True, "make_venv": False},
            {"use_py3_11": False, "make_venv": True},
            {"use_py3_11": True, "make_venv": True},
            {"use_py3_11": False, "make_venv": False},
        ]
        for opts in salvage_options:
            if opts not in retry_options:
                retry_options.append(opts)

    repro_attempts: list[ReproductionMetadata] = []

    ctx_logger = logger.bind(
        run_group_id=run_group_id, runs_dir=runs_dir, run_id=run_id, destinations=["run"]
    )

    for opts in retry_options:
        ctx_logger.info(
            f"Executing reproduce.sh with py3_11={opts['use_py3_11']}"
            f" and make_venv={opts['make_venv']}"
        )
        repro_attempt = await reproduce_on_computer(
            computer_runtime=computer_runtime,
            computer_config=computer_config,
            runtime_config=runtime_config,
            submission_path=submission_path,
            run_group_id=run_group_id,
            runs_dir=runs_dir,
            run_id=run_id,
            run_dir=run_dir,
            submission_cluster_path=submission_cluster_path,
            output_cluster_path=output_cluster_path,
            timeout=timeout,
            use_py3_11=opts["use_py3_11"],
            make_venv=opts["make_venv"],
        )
        repro_attempts.append(repro_attempt)
        if _should_retry(retries_enabled, repro_attempt, retry_threshold):
            ctx_logger.info(
                f"Reproduction attempt ran for less than {retry_threshold} seconds,"
                " retrying with different configuration."
            )
            continue  # retry, with next configuration
        else:
            break  # this last attempt was it

    repro_metadata = repro_attempts[-1]
    repro_metadata = _populate_retried_results(repro_metadata, repro_attempts[:-1])

    return repro_metadata


def _should_retry(
    retries_enabled: bool, repro_attempt: ReproductionMetadata, retry_threshold: float
) -> bool:
    """helper for determining whether we should retry to run reproduce.sh"""
    execution_time = repro_attempt.repro_execution_time or 0
    return retries_enabled and execution_time < retry_threshold


def _populate_retried_results(
    repro_metadata: ReproductionMetadata, repro_attempts: list[ReproductionMetadata]
) -> ReproductionMetadata:
    """Populates a ReproductionMetadata.retried_results with info from previous attempts"""
    if len(repro_attempts) >= 1:
        retried = [
            ReproScriptRunOutcome(
                float(m.repro_execution_time or 0),
                m.timedout,
                m.repro_log,
                m.repro_exit_code,
            )
            for m in repro_attempts
        ]
        repro_metadata = replace(repro_metadata, retried_results=retried)
    return repro_metadata
