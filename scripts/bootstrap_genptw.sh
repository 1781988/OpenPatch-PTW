#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${ROOT_DIR}/third_party"
GENPTW_DIR="${THIRD_PARTY}/GenPTW"
UPSTREAM_URL="https://github.com/GanZhenliang/GenPTW.git"
# Pin the inspected public release commit for reproducibility.
UPSTREAM_COMMIT="8c22fc04dcf9d896eaeabd034d09918588e82cb3"

mkdir -p "${THIRD_PARTY}"

if [[ ! -d "${GENPTW_DIR}/.git" ]]; then
  git clone "${UPSTREAM_URL}" "${GENPTW_DIR}"
fi

cd "${GENPTW_DIR}"
git fetch --all --tags
git checkout "${UPSTREAM_COMMIT}"

echo "GenPTW ready at: ${GENPTW_DIR}"
echo "Pinned commit: ${UPSTREAM_COMMIT}"
echo "Next: follow third_party/GenPTW/README.md to download VAE, ConvNeXt, LaMA and official checkpoints."
