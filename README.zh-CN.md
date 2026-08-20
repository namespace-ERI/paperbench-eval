# PaperBench 评测运行时

[English README](README.md)

这个仓库只保存 PaperBench 的评测运行时代码，负责把一个研究复现
agent 的完整评测流程跑起来：

1. **Rollout**：在隔离容器中运行 agent，生成 submission。
2. **Reproduction**：使用全新的容器执行 submission 中的 `reproduce.sh`。
3. **Grading**：根据 rubric 对执行结果进行分项评分。
4. **JudgeEval**：可选地评估 judge 本身在参考提交上的表现。

代码、Dockerfile 和技能树放在 GitHub，较大的任务数据、JudgeEval
输入和预装 agent 镜像放在 ModelScope：

- GitHub：`https://github.com/namespace-ERI/paperbench-eval`
- ModelScope：`YuyangHu/paperbench-assets`

## 1. 发布内容

### GitHub 仓库中的内容

GitHub 仓库包含：

- `paperbench/`：rollout、reproduction、grading、judge、monitor、registry
  和各类 solver。
- `vendor/`：独立运行所需的 `nanoeval`、`nanoeval_alcatraz`、`alcatraz`
  和 `preparedness_turn_completer` 最小源码。
- `experiments/splits/`：`all`、`debug`、`dev`、`lite`、`testing` 等官方
  split 文件。
- `sota/skills/`：供 Codex/PI skill solver 使用的 distilled skill tree。
- `paperbench/Dockerfile.base`：标准 rollout/judge 基础镜像定义。
- `paperbench/reproducer.Dockerfile`：执行 `reproduce.sh` 的干净容器定义。
- `paperbench/solvers/codexagent/Dockerfile`：预装 Codex 和 PI 的 agent 镜像
  定义。
- `README.md` 和本文件：英文、中文运行说明。

以下内容明确不放入 GitHub：

- 生成的 trajectory、session、rollout log、checkpoint、run output。
- `record/`、`exports/`、`runs/`、`sota/state/`、`sota/cases/`。
- `factory/` 以及任务生成、数据生成代码。
- API key、缓存、临时输入文件和本地运行结果。

### ModelScope 数据集中的内容

ModelScope 数据集使用下面的目录布局：

```text
paperbench-assets/
├── data/
│   ├── papers/
│   │   └── <paper_id>/
│   │       ├── config.yaml
│   │       ├── paper.pdf
│   │       ├── paper.md
│   │       ├── addendum.md
│   │       ├── judge.addendum.md
│   │       ├── blacklist.txt
│   │       ├── rubric.json
│   │       └── assets/
│   └── judge_eval/
│       ├── <paper_or_topic>/
│       └── ...
└── docker/
    └── images/
        └── pb-env-codex-latest.tar
```

各目录的用途如下：

| ModelScope 路径 | 用途 | 是否为普通评测必需 |
| --- | --- | --- |
| `data/papers/` | 原题论文、题目元数据、rubric、addendum 和论文附件 | 是 |
| `data/judge_eval/` | JudgeEval 的参考提交和标签 | 只有评估 judge 时需要 |
| `docker/images/pb-env-codex-latest.tar` | 预装 Codex 0.138.0 和 PI 0.84.0 的 agent 镜像 | Codex/PI rollout 时需要 |

技能树已经随 GitHub 仓库发布，位于 `sota/skills/`，不需要从
ModelScope 另外下载。Dockerfile 快照和构建定义也都在 GitHub 仓库里，
ModelScope 只保留 `data/` 和可选镜像归档。

ModelScope **只上传一张预构建镜像**：

- `pb-env-codex:latest`：预装 `@openai/codex@0.138.0` 和
  `@earendil-works/pi-coding-agent@0.84.0`。

下面三张镜像不作为归档上传：

- `pb-env:latest`：标准 rollout/judge 基础镜像。
- `pb-reproducer:latest`：干净 reproduction 镜像。
- `pb-env-codedev-pi:0.84.0-proxy`：带环境特定代理的 Code-Dev/Qwen PI 镜像。

前两张官方镜像可以根据 GitHub 中的 Dockerfile 在目标机器本地构建；
最后一张镜像只适用于特定代理环境，因此不纳入这个发布包。

## 2. 安装 GitHub 代码

```bash
git clone https://github.com/namespace-ERI/paperbench-eval.git
cd paperbench-eval
uv sync
```

复制环境变量模板：

```bash
cp .env.example .env
```

至少配置 rollout 和 judge 所需的 API key：

```dotenv
OPENAI_API_KEY=...
GRADER_OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
```

加载环境变量：

```bash
set -a
source .env
set +a
```

如果某些论文的 reproduction 需要访问 Hugging Face 或其他服务，单独配置
agent 容器会收到的环境文件：

```bash
cp paperbench/solvers/agent.env.example paperbench/solvers/agent.env
```

只填写允许暴露给 agent 的变量。`.env` 和
`paperbench/solvers/agent.env` 都不会提交到 GitHub。

## 3. 下载 ModelScope 数据

### 推荐方式：使用仓库脚本

先设置数据集 id：

```bash
export PAPERBENCH_MODELSCOPE_DATASET="YuyangHu/paperbench-assets"
```

如果当前机器需要代理访问 ModelScope，可以先执行：

```bash
source /share/project/yuyang/workspace/setvpn.sh
```

下载普通 rollout/reproduction/grading 所需的论文包：

```bash
scripts/fetch_modelscope_assets.sh .paperbench-assets
```

如果还要运行 JudgeEval：

```bash
scripts/fetch_modelscope_assets.sh \
  .paperbench-assets \
  --with-judge-eval
```

如果希望同时下载预构建的 `pb-env-codex:latest`：

```bash
scripts/fetch_modelscope_assets.sh \
  .paperbench-assets \
  --with-judge-eval \
  --with-images
```

下载完成后设置两个路径：

```bash
export PAPERBENCH_ASSETS_DIR="$PWD/.paperbench-assets"
export PAPERBENCH_DATA_DIR="$PAPERBENCH_ASSETS_DIR/data"
```

### 下载后每个文件放在哪里

下载目录不应该移动到 GitHub 仓库的 `data/` 路径中。推荐保持下面的结构：

```text
paperbench-eval/
├── .paperbench-assets/
│   ├── data/
│   │   ├── papers/
│   │   └── judge_eval/
│   └── docker/
│       └── images/
│           └── pb-env-codex-latest.tar
├── paperbench/
└── sota/
    └── skills/
```

对应关系：

- `PAPERBENCH_DATA_DIR` 必须指向
  `.paperbench-assets/data`，因为运行时会在其下查找
  `.paperbench-assets/data/papers/<paper_id>/`。
- 原题论文和题目相关文件位于
  `.paperbench-assets/data/papers/<paper_id>/`。
- 某篇论文的 skill 位于 GitHub 仓库里的 `sota/skills/<paper_id>/skill/`。
  例如 `pinn` 的 skill 是 `sota/skills/pinn/skill/`。
- JudgeEval 输入位于
  `.paperbench-assets/data/judge_eval/`。
- 预构建镜像归档位于
  `.paperbench-assets/docker/images/pb-env-codex-latest.tar`。

技能树已经随 GitHub 仓库发布，不需要从 ModelScope 另外下载。
Dockerfile 快照和构建定义也都在 GitHub 仓库里，ModelScope 只保留
`data/` 和可选镜像归档。

### 校验论文包

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

如果报 `PaperbenchDataNotFoundError`，通常是
`PAPERBENCH_DATA_DIR` 指到了某一个 paper 目录，而不是包含 `papers/`
的 `data/` 目录。

## 4. 准备 Docker 镜像

### 加载 ModelScope 中唯一的预构建镜像

```bash
docker load \
  -i "$PAPERBENCH_ASSETS_DIR/docker/images/pb-env-codex-latest.tar"
docker image inspect pb-env-codex:latest
```

这个镜像可以用于：

- `CodexVanillaAgentSolver`
- `CodexAgentSolver`
- `PiVanillaAgentSolver`
- `PiAgentSolver`

### 本地构建两张官方镜像

如果要跑完整的 reproduction/grading 流程，目标机器仍然需要
`pb-env:latest` 和 `pb-reproducer:latest`。它们不从 ModelScope 下载，
而是从 GitHub 代码构建：

```bash
bash paperbench/scripts/build-docker-images.sh
```

构建后检查：

```bash
docker images pb-env pb-reproducer pb-env-codex
```

如果只使用预装 Codex/PI 镜像进行 rollout，则只需要加载
`pb-env-codex-latest.tar`。但完整流水线的 reproduction 默认使用
`pb-reproducer:latest`，不能用 `pb-env-codex:latest` 直接替代。

## 5. 运行评测

### Dummy smoke test

先用不消耗 API 的 dummy solver 做一篇 debug 任务的连通性测试：

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

### Codex vanilla

不上传 skill，使用官方 prompt：

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

### Codex + skill

先选择与目标论文对应的 skill tree：

```bash
export PAPERBENCH_SKILL_DIR="$PWD/sota/skills/pinn/skill"
```

然后运行：

```bash
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

`skills_dir` 要指向一个具体的 `skill/` 目录，不要直接指向包含多个
论文和历史输出的上层目录。

### PI + skill

预装镜像同时包含 PI，因此可以使用同一张镜像：

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

### split 选择

可用的 split 文件在 GitHub 的 `experiments/splits/` 中：

- `debug`：最小调试集。
- `lite`：轻量评测集。
- `dev`：开发评测集。
- `testing`：测试用 split。
- `all`：全部官方任务。

把命令中的 `paperbench.paper_split=pinn` 替换为目标 split 即可。
也可以直接使用某个论文 id，具体以当前代码的 split 解析规则为准。

## 6. 复现、评分和 JudgeEval

正常入口会依次执行 rollout、reproduction 和 grading，结果写入：

```text
runs/<run_group_id>/
├── group.log
└── <paper_id>_<uuid>/
    ├── run.log
    ├── status.json
    ├── metadata.json
    ├── grade.json
    └── submissions/
```

`grade.json` 是最终结果，包含 rollout 元数据、reproduction 元数据、
judge tree、总分和失败诊断。

如果要给已有 submission 评分，可以使用 direct-submission solver：

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

JudgeEval 只有在评估 judge 时需要下载：

```bash
scripts/fetch_modelscope_assets.sh \
  .paperbench-assets \
  --with-judge-eval

export PAPERBENCH_DATA_DIR="$PWD/.paperbench-assets/data"

uv run python -m paperbench.scripts.run_judge_eval \
  judge=dummy \
  out_dir=judge_eval_results
```

## 7. 断点续跑和排错

使用同样的 solver、镜像、prompt、judge 配置恢复一个已有 run group：

```bash
uv run python -m paperbench.nano.entrypoint \
  paperbench.resume_run_group_id='<existing-run-group-id>' \
  paperbench.resume_no_extend=true \
  paperbench.paper_split=pinn \
  <其余参数保持与原运行一致>
```

常见检查：

```bash
echo "$PAPERBENCH_DATA_DIR"
test -d "$PAPERBENCH_DATA_DIR/papers"
docker image inspect pb-env-codex:latest
docker image inspect pb-env:latest
docker image inspect pb-reproducer:latest
```

如果需要 GPU：

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

只在该测试成功后，把 solver 和 reproduction runtime 的
`is_nvidia_gpu_env` 设置为 `true`。

不要把 `record/`、`runs/`、导出的 submission 或 trajectory 复制回
GitHub 仓库，也不要把 API key 写入 README、shell script 或 tracked
文件。
