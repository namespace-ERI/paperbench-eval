#!/usr/bin/env bash
set -euo pipefail

forbidden_prefixes=(
  "factory/"
  "record/"
  "exports/"
  "sota/cases/"
  "sota/state/"
  "data/"
  ".paperbench-assets/"
)

for prefix in "${forbidden_prefixes[@]}"; do
  if git ls-files -- "${prefix}" | grep -q .; then
    echo "Forbidden published path detected: ${prefix}" >&2
    git ls-files -- "${prefix}" >&2
    exit 1
  fi
done

if git grep -nE 'ghp_[A-Za-z0-9]{20,}|ms-[A-Za-z0-9-]{20,}' -- . ':!scripts/verify_release_scope.sh' >/dev/null; then
  echo "Potential access token detected in tracked content." >&2
  exit 1
fi

echo "Release scope check passed."
