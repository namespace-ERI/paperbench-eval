#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

bash paperbench/scripts/build-docker-images.sh
docker build --platform=linux/amd64 \
  -t pb-env-codex:latest \
  -f paperbench/solvers/codexagent/Dockerfile \
  .

if [[ "${BUILD_CODEDEV_PI_IMAGE:-0}" == "1" ]]; then
  docker build --platform=linux/amd64 \
    --build-arg "RUNTIME_PROXY_URL=${RUNTIME_PROXY_URL:-http://172.17.0.1:7895}" \
    -t pb-env-codedev-pi:0.84.0-proxy \
    -f sota/docker/pi_codedev/Dockerfile \
    .
fi
