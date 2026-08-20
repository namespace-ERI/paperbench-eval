# PaperBench Evaluation Runtime

[中文版说明](README.zh-CN.md)

This repository is a standalone runtime for evaluating research-reproduction
agents with PaperBench. It contains the code required to run the complete
evaluation pipeline:

1. **Rollout**: an agent works in an isolated container and writes a submission.
2. **Reproduction**: a fresh container runs the submitted `reproduce.sh`.
3. **Grading**: a rubric-driven judge assigns partial credit from the executed
   submission.
4. **Judge evaluation**: optional evaluation of a judge against the included
   JudgeEval reference submissions.

The repository intentionally contains source code, Dockerfiles, and distilled
skill trees. The official task packages, JudgeEval assets, and optional
prebuilt Docker image archives live in a separate ModelScope dataset.

## Publication Scope

Included here:

- `paperbench/`: rollout, reproduction, grading, judge, monitor, registry, and
  solver implementations.
- `vendor/`: the minimal `nanoeval`, `nanoeval_alcatraz`, `alcatraz`, and
  `preparedness_turn_completer` source dependencies needed to run PaperBench
  without a sibling monorepo checkout.
- `experiments/splits/`: official `all`, `debug`, `dev`, `lite`, and `testing`
  split files.
- `sota/skills/`: distilled skill trees used by skill-enabled solvers.
- `paperbench/Dockerfile.base`, `paperbench/reproducer.Dockerfile`, and the
  Codex/PI Dockerfiles.
- focused tests for registry resolution, rollout plumbing, reproduction, and
  grading.

Deliberately excluded:

- generated trajectories, sessions, rollout logs, checkpoints, and exported
  archives;
- `record/`, `exports/`, `runs/`, `sota/state/`, and `sota/cases/`;
- task-generation code and all `factory/` content;
- local API keys, caches, manual input files, and temporary fixes;
- official paper packages and JudgeEval assets, which must be downloaded
  separately.

## Repository Layout

```text
paperbench/                         Core evaluation runtime
  nano/                             Nanoeval integration and pipeline orchestration
  solvers/                          Dummy, direct-submission, BasicAgent, Codex, PI solvers
  judge/                            Rubric tree, judge implementations, JudgeEval support
  reproduce.py                      Fresh-container reproduction logic
  scripts/                          Standalone judge and monitor CLIs
  Dockerfile.base                   Base rollout/judge image definition
  reproducer.Dockerfile             Fresh reproduction image definition
vendor/                             Runtime dependencies vendored for standalone installation
experiments/splits/                 Official and lightweight paper id lists
scripts/fetch_modelscope_assets.sh  Downloads external paper data and optional images
scripts/build_agent_images.sh       Builds the standard and optional rollout images
scripts/verify_release_scope.sh     Rejects tracked trajectories, data, factory, and credentials
```

## Prerequisites

- Linux host with Python 3.11 or newer.
- Git and Git LFS.
- Docker Engine. The invoking user must be allowed to access the Docker socket.
- `uv` for reproducible Python environment setup.
- NVIDIA driver and NVIDIA Container Toolkit only when rollout or reproduction
  should receive GPUs.
- API credentials for the rollout agent and for the rubric judge.

Confirm local Docker access before launching an evaluation:

```bash
docker info >/dev/null
docker run --rm hello-world
```

For GPU runs, also confirm that a container can see the selected GPU:

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

## Install

Clone the code repository and create the Python environment:

```bash
git clone https://github.com/namespace-ERI/paperbench-eval.git
cd paperbench-eval
uv sync
```

The vendored runtime packages are wired through `pyproject.toml`; no sibling
`../common` checkout is required.

Copy the environment template:

```bash
cp .env.example .env
```

At minimum, configure a rollout key and a judge key:

```dotenv
OPENAI_API_KEY=...
GRADER_OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
```

`GRADER_OPENAI_API_KEY` falls back to `OPENAI_API_KEY` when it is not set, but
separating them is useful when rollout and judging use different accounts or
models.

Load the environment for each shell:

```bash
set -a
source .env
set +a
```

Some target papers need the agent to access Hugging Face or an API during
reproduction. These credentials are deliberately separate from `.env` because
the runtime uploads this file into the rollout container:

```bash
cp paperbench/solvers/agent.env.example paperbench/solvers/agent.env
```

Populate only credentials that are permitted to be exposed to the agent, for
example:

```dotenv
OPENAI_API_KEY=...
HF_TOKEN=...
```

Never commit `.env` or `paperbench/solvers/agent.env`.

## Download Official Assets

The PaperBench registry expects:

```text
${PAPERBENCH_DATA_DIR}/papers/<paper_id>/
  config.yaml
  paper.pdf
  paper.md
  addendum.md
  judge.addendum.md
  blacklist.txt
  rubric.json
  assets/
```

The separate ModelScope dataset provides the paper packages under `data/`,
JudgeEval inputs under `data/judge_eval/`, and the optional prebuilt image
under `docker/images/`. The distilled skill trees are part of this GitHub
repository under `sota/skills/`.

After the ModelScope dataset is created, set its id once:

```bash
export PAPERBENCH_MODELSCOPE_DATASET="YuyangHu/paperbench-assets"
```

Download paper packages:

```bash
scripts/fetch_modelscope_assets.sh .paperbench-assets
export PAPERBENCH_ASSETS_DIR="$PWD/.paperbench-assets"
export PAPERBENCH_DATA_DIR="$PAPERBENCH_ASSETS_DIR/data"
```

Download JudgeEval inputs only when evaluating a judge implementation:

```bash
scripts/fetch_modelscope_assets.sh .paperbench-assets --with-judge-eval
```

The ModelScope release publishes only the preinstalled
`pb-env-codex:latest` image. The standard `pb-env:latest` and
`pb-reproducer:latest` images can be built locally from the Dockerfiles; they
are intentionally not included as large archives:

```bash
scripts/fetch_modelscope_assets.sh .paperbench-assets --with-images
```

The download directory is ignored by Git. Do not move its contents into this
repository's tracked `data/` path.

The skill trees are already in this repository under `sota/skills/`. Point
`paperbench.solver.skills_dir` at `sota/skills/<paper_id>/skill` when you want
Codex or PI to use them.

Verify the registry sees the expected paper packages:

```bash
uv run python - <<'PY'
from paperbench.paper_registry import paper_registry

for paper_id in paper_registry.list_paper_ids():
    paper = paper_registry.get_paper(paper_id)
    assert paper.paper_pdf.exists(), paper.paper_pdf
    assert paper.paper_md.exists(), paper.paper_md
    assert paper.rubric.exists(), paper.rubric
    print(paper.id)
PY
```

## Build Or Load Docker Images

The standard pipeline needs two images:

- `pb-env:latest` for rollout and local grading.
- `pb-reproducer:latest` for fresh-container execution of `reproduce.sh`.

Build them from source:

```bash
bash paperbench/scripts/build-docker-images.sh
```

For Codex or PI rollout, build the extended agent image:

```bash
scripts/build_agent_images.sh
```

This produces:

```text
pb-env:latest
pb-reproducer:latest
pb-env-codex:latest
```

The optional Code-Dev PI image is environment-specific because its Dockerfile
sets a default host proxy. Build it only when that workflow is required:

```bash
BUILD_CODEDEV_PI_IMAGE=1 \
RUNTIME_PROXY_URL=http://172.17.0.1:7895 \
scripts/build_agent_images.sh
```

When prebuilt archives are present in the ModelScope asset checkout, load one
with:

```bash
docker load -i "$PAPERBENCH_ASSETS_DIR/docker/images/<archive-name>.tar"
```

Run `docker images pb-env-codex` to verify the prebuilt agent image before a
Codex or PI rollout. For a full local pipeline, also build and verify
`pb-env:latest` and `pb-reproducer:latest`.

## Evaluation Stages And Outputs

The normal entrypoint runs rollout, reproduction, and grading in sequence:

```text
runs/<run_group_id>/
  group.log
  <paper_id>_<uuid>/
    run.log
    status.json
    metadata.json
    grade.json
    submissions/<timestamp>/submission.tar.gz
    submissions/<timestamp>/submission_executed.tar.gz
    submissions/<timestamp>/submission_executed_grader_output_0.json
```

All of these are generated runtime output. They are ignored and must not be
added to a source-control commit.

`grade.json` is the top-level result. It includes the paper id, rollout
metadata, reproduction metadata, full grader tree, aggregate score, and failure
diagnostics. A valid standard result requires a successful rollout, a runnable
executed submission, and successful judging.

## Smoke Test

Use the official one-paper `debug` split with the included dummy solver and
dummy judge before spending API budget:

```bash
export PAPERBENCH_DATA_DIR="$PWD/.paperbench-assets/data"
export GRADER_OPENAI_API_KEY="${GRADER_OPENAI_API_KEY:-not-used-by-dummy}"

uv run python -m paperbench.nano.entrypoint \
  paperbench.paper_split=debug \
  paperbench.solver=paperbench.solvers.dummy.solver:PaperBenchDummySolver \
  paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.solver.computer_runtime.env.pull_from_registry=false \
  paperbench.reproduction.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.reproduction.computer_runtime.env.pull_from_registry=false \
  paperbench.judge.scaffold=dummy \
  runner.max_retries=0 \
  runner.concurrency=1 \
  runner.recorder=nanoeval.json_recorder:json_recorder
```

If this fails before the agent starts, check `PAPERBENCH_DATA_DIR`, Docker
access, image tags, and `runs/<run_group_id>/group.log`.

## Run A Standard BasicAgent Rollout

The following is a conservative one-paper command. Replace the model and time
budget to match the provider's limits:

```bash
export PAPERBENCH_DATA_DIR="$PWD/.paperbench-assets/data"

uv run python -m paperbench.nano.entrypoint \
  paperbench.paper_split=pinn \
  paperbench.solver=paperbench.solvers.basicagent.solver:BasicAgentSolver \
  paperbench.solver.completer_config=paperbench.solvers.basicagent.completer:OpenAIResponsesTurnCompleterConfig \
  paperbench.solver.completer_config.model='gpt-5.5' \
  paperbench.solver.time_limit=21600 \
  paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.solver.computer_runtime.env.pull_from_registry=false \
  paperbench.solver.computer_runtime.env.is_nvidia_gpu_env=true \
  paperbench.reproduction.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.reproduction.computer_runtime.env.pull_from_registry=false \
  paperbench.reproduction.computer_runtime.env.is_nvidia_gpu_env=true \
  paperbench.reproduction.timeout=21600 \
  paperbench.judge.completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
  paperbench.judge.completer_config.model='gpt-5.5' \
  runner.max_retries=0 \
  runner.concurrency=1 \
  runner.recorder=nanoeval.json_recorder:json_recorder
```

For a CPU-only rollout, set both `is_nvidia_gpu_env=false` values and add:

```text
paperbench.num_gpus=0
paperbench.gpu_device_ids=[]
```

For a multi-paper split, use `paperbench.paper_split=lite`, `dev`, or `all`.
Set `runner.concurrency` only after confirming that the host has enough Docker,
GPU, API, and network capacity for concurrent agents.

## Codex And PI Rollout Modes

All rollout solvers use the same pipeline and output layout. The important
differences are the agent class, the image, the API endpoint, and whether a
distilled skill tree is uploaded into the container.

### Codex Vanilla

`CodexVanillaAgentSolver` uses the official PaperBench prompt and no skills:

```bash
uv run python -m paperbench.nano.entrypoint \
  paperbench.paper_split=pinn \
  paperbench.docker_image=pb-env-codex:latest \
  paperbench.solver=paperbench.solvers.codexagent_vanilla.solver:CodexVanillaAgentSolver \
  paperbench.solver.codex_model='gpt-5.5' \
  paperbench.solver.codex_base_url="$OPENAI_BASE_URL" \
  paperbench.solver.time_limit=21600 \
  paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.solver.computer_runtime.env.pull_from_registry=false \
  paperbench.reproduction.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.reproduction.computer_runtime.env.pull_from_registry=false \
  runner.concurrency=1 \
  runner.recorder=nanoeval.json_recorder:json_recorder
```

### Codex With Skills

The non-vanilla Codex solver uploads the selected `skill` directory:

```bash
export PAPERBENCH_SKILL_DIR="$PWD/sota/skills/pinn/skill"

uv run python -m paperbench.nano.entrypoint \
  paperbench.paper_split=pinn \
  paperbench.docker_image=pb-env-codex:latest \
  paperbench.solver=paperbench.solvers.codexagent.solver:CodexAgentSolver \
  paperbench.solver.skills_dir="$PAPERBENCH_SKILL_DIR" \
  paperbench.solver.codex_model='gpt-5.5' \
  paperbench.solver.time_limit=21600 \
  paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.solver.computer_runtime.env.pull_from_registry=false \
  paperbench.reproduction.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.reproduction.computer_runtime.env.pull_from_registry=false \
  runner.concurrency=1 \
  runner.recorder=nanoeval.json_recorder:json_recorder
```

The selected skill tree should match the target paper or an intentionally chosen
related-work collection. Do not point `skills_dir` at a broad directory
containing unrelated task outputs.

### PI Vanilla

`PiVanillaAgentSolver` uses PI with the official prompt and explicitly disables
skills:

```bash
uv run python -m paperbench.nano.entrypoint \
  paperbench.paper_split=pinn \
  paperbench.docker_image=pb-env-codex:latest \
  paperbench.solver=paperbench.solvers.piagent_vanilla.solver:PiVanillaAgentSolver \
  paperbench.solver.pi_provider='openai-compatible-provider' \
  paperbench.solver.pi_model='your-model-id' \
  paperbench.solver.pi_api_key_env=OPENAI_API_KEY \
  paperbench.solver.pi_upstream_base_url="$OPENAI_BASE_URL" \
  paperbench.solver.pi_relay_enabled=false \
  paperbench.solver.time_limit=21600 \
  paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.solver.computer_runtime.env.pull_from_registry=false \
  paperbench.reproduction.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.reproduction.computer_runtime.env.pull_from_registry=false \
  runner.concurrency=1 \
  runner.recorder=nanoeval.json_recorder:json_recorder
```

### PI With Skills

Use `PiAgentSolver` and give it one extracted skill tree:

```bash
export PAPERBENCH_SKILL_DIR="$PWD/sota/skills/pinn/skill"

uv run python -m paperbench.nano.entrypoint \
  paperbench.paper_split=pinn \
  paperbench.docker_image=pb-env-codex:latest \
  paperbench.solver=paperbench.solvers.piagent.solver:PiAgentSolver \
  paperbench.solver.skills_dir="$PAPERBENCH_SKILL_DIR" \
  paperbench.solver.pi_provider='openai-compatible-provider' \
  paperbench.solver.pi_model='your-model-id' \
  paperbench.solver.pi_api_key_env=OPENAI_API_KEY \
  paperbench.solver.pi_upstream_base_url="$OPENAI_BASE_URL" \
  paperbench.solver.pi_relay_enabled=false \
  paperbench.solver.time_limit=21600 \
  paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.solver.computer_runtime.env.pull_from_registry=false \
  paperbench.reproduction.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.reproduction.computer_runtime.env.pull_from_registry=false \
  runner.concurrency=1 \
  runner.recorder=nanoeval.json_recorder:json_recorder
```

### PI Code-Dev

`PiVanillaCodeDevAgentSolver` is a specialized workflow. It requires:

- `paperbench.judge.code_only=true`;
- `pb-env-codedev-pi:0.84.0-proxy`;
- the `sglang-qwen` PI provider;
- a reachable OpenAI-compatible Qwen endpoint.

It is not the normal full-reproduction benchmark path. In Code-Dev mode,
`reproduce.sh` is not required and grading examines the source submission only.
Use it only when a Code-Dev benchmark is explicitly intended.

## Grade Existing Submissions

To score pre-existing repositories without a rollout, arrange submissions as:

```text
submissions/
  pinn/
    submission/
  rice/
    submission/
```

Then use the direct-submission solver:

```bash
uv run python -m paperbench.nano.entrypoint \
  paperbench.paper_split=pinn \
  paperbench.solver=paperbench.solvers.direct_submission.solver:PBDirectSubmissionSolver \
  paperbench.solver.submissions_dir="$PWD/submissions" \
  paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.solver.computer_runtime.env.pull_from_registry=false \
  paperbench.reproduction.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.reproduction.computer_runtime.env.pull_from_registry=false \
  runner.concurrency=1 \
  runner.recorder=nanoeval.json_recorder:json_recorder
```

The normal scoring path still runs reproduction and grading. To perform a
standalone judge invocation on an already executed submission directory:

```bash
uv run python -m paperbench.scripts.run_judge \
  submission_path=/path/to/executed/submission \
  paper_id=pinn \
  judge=simple \
  out_dir=/path/to/judge-output \
  completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
  completer_config.model='gpt-5.5'
```

Use `judge=dummy` for a no-API plumbing check.

## Evaluate The Judge

JudgeEval assets are optional and are not needed for ordinary PaperBench
rollouts. After downloading them with `--with-judge-eval`, run:

```bash
export PAPERBENCH_DATA_DIR="$PWD/.paperbench-assets/data"

uv run python -m paperbench.scripts.run_judge_eval \
  judge=dummy \
  out_dir=judge_eval_results
```

For the LLM-backed simple judge, add a `completer_config` and model in the same
form as the standalone judge command.

## Resume, Retry, And Runtime Control

Every entrypoint invocation creates a `run_group_id`. To resume a partially
completed run group without adding new tasks:

```bash
uv run python -m paperbench.nano.entrypoint \
  paperbench.resume_run_group_id='<existing-run-group-id>' \
  paperbench.resume_no_extend=true \
  paperbench.paper_split=pinn \
  <the same solver, runtime, and judge configuration>
```

Useful controls:

- `runner.concurrency`: number of simultaneous PaperBench tasks.
- `runner.max_retries`: nanoeval-level retry count for system failures.
- `paperbench.solver.time_limit`: rollout budget in seconds.
- `paperbench.reproduction.timeout`: `reproduce.sh` time limit in seconds.
- `paperbench.target_duration_hr`: choose an archived intermediate agent
  checkpoint closest to a target duration for later reproduction and scoring.
- `paperbench.reproduction.skip_reproduction=true`: rollout and judge-only
  experiments. This is not a standard full reproduction score.
- `paperbench.judge.code_only=true`: source-only Code-Dev grading. This is not
  equivalent to normal execution-based PaperBench scoring.

Do not change solver, image, prompt mode, or judge configuration while resuming
a run group unless the goal is explicitly to produce a different experiment.

## Troubleshooting

### `PaperbenchDataNotFoundError`

`PAPERBENCH_DATA_DIR` must point at the directory that contains `papers/`, not
at a single paper directory:

```bash
export PAPERBENCH_DATA_DIR="$PWD/.paperbench-assets/data"
```

### Paper package is reported as an LFS pointer or missing

Re-run the asset download with Git LFS installed:

```bash
git lfs version
scripts/fetch_modelscope_assets.sh .paperbench-assets --force
```

### Docker is unavailable

The local runtime checks the Docker daemon before it creates tasks. Confirm both
that Docker is running and that the current user can access its socket:

```bash
docker info
id -nG
```

### GPU is not visible in rollout or reproduction

Check host `nvidia-smi`, then test `docker run --rm --gpus all ... nvidia-smi`.
Set `is_nvidia_gpu_env=true` on the solver and reproduction runtimes only after
the Docker GPU test succeeds.

### Rollout works but grading fails

Inspect the run's `grade.json`, `run.log`, and
`submission_executed_grader_output_0.json`. A successful rollout alone is not
a valid full score: the submission must also survive fresh-container execution
and produce judge-readable outputs.

### API endpoint is not reachable inside containers

Set `PB_PROXY_URL` only when the container can reach the configured proxy from
the Docker bridge. For ordinary public APIs, leave it unset. Keep API keys in
ignored env files rather than shell history or tracked scripts.

## Release Verification

Before pushing code, run:

```bash
scripts/verify_release_scope.sh
git status --short
```

The scope check fails if Git tracks data assets, trajectories, factory code,
state directories, or access-token-like strings.
