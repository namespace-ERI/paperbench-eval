#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/fetch_modelscope_assets.sh [destination] [--with-judge-eval] [--with-images] [--force]

Downloads PaperBench assets from the public ModelScope dataset using Git LFS.

By default this downloads:
  - data/papers: official paper packages used by rollout, reproduction, and grading
  - skills: paper-aligned distilled skill trees
  - docker/dockerfiles: exact Dockerfiles and image manifest

Options:
  --with-judge-eval  Also download data/judge_eval for judge-quality evaluation.
  --with-images      Also download docker/images. This can require many gigabytes.
  --force            Remove an existing destination before cloning.

Environment:
  PAPERBENCH_MODELSCOPE_DATASET  Required dataset id, for example owner/paperbench-assets.
EOF
}

DESTINATION="${1:-.paperbench-assets}"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  shift
fi

WITH_JUDGE_EVAL=false
WITH_IMAGES=false
FORCE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-judge-eval) WITH_JUDGE_EVAL=true ;;
    --with-images) WITH_IMAGES=true ;;
    --force) FORCE=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

DATASET_ID="${PAPERBENCH_MODELSCOPE_DATASET:-}"
if [[ -z "${DATASET_ID}" ]]; then
  echo "PAPERBENCH_MODELSCOPE_DATASET must be set to the ModelScope dataset id." >&2
  exit 2
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required." >&2
  exit 2
fi
if ! command -v git-lfs >/dev/null 2>&1 && ! command -v git >/dev/null 2>&1; then
  echo "git-lfs is required. Install Git LFS before downloading assets." >&2
  exit 2
fi

if [[ -e "${DESTINATION}" ]]; then
  if [[ "${FORCE}" != true ]]; then
    echo "Destination already exists: ${DESTINATION}. Use --force to replace it." >&2
    exit 2
  fi
  rm -rf "${DESTINATION}"
fi

REMOTE="https://www.modelscope.cn/datasets/${DATASET_ID}.git"
GIT_LFS_SKIP_SMUDGE=1 git clone "${REMOTE}" "${DESTINATION}"
git -C "${DESTINATION}" lfs install --local

INCLUDE_PATTERNS="data/papers/**,skills/**,docker/dockerfiles/**"
if [[ "${WITH_JUDGE_EVAL}" == true ]]; then
  INCLUDE_PATTERNS="${INCLUDE_PATTERNS},data/judge_eval/**"
fi
if [[ "${WITH_IMAGES}" == true ]]; then
  INCLUDE_PATTERNS="${INCLUDE_PATTERNS},docker/images/**"
fi

git -C "${DESTINATION}" lfs pull --include="${INCLUDE_PATTERNS}"

cat <<EOF
Assets downloaded to: ${DESTINATION}

For PaperBench commands:
  export PAPERBENCH_ASSETS_DIR="\$(cd "${DESTINATION}" && pwd)"
  export PAPERBENCH_DATA_DIR="\${PAPERBENCH_ASSETS_DIR}/data"

To use skills, pass an individual skill tree such as:
  paperbench.solver.skills_dir="\${PAPERBENCH_ASSETS_DIR}/skills/pinn/skill"
EOF
