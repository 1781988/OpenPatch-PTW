#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-openpatch}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_TAG="${CUDA_TAG:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.0}"

if command -v mamba >/dev/null 2>&1; then
  ENV_TOOL=mamba
elif command -v conda >/dev/null 2>&1; then
  ENV_TOOL=conda
else
  ENV_TOOL=""
fi

if [[ -n "${ENV_TOOL}" ]]; then
  if ! "${ENV_TOOL}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    "${ENV_TOOL}" create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip git unzip -y
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
else
  python3 -m venv "${ROOT_DIR}/.venv"
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
python -m pip install -r "${ROOT_DIR}/requirements.txt"
python -m pip install -e "${ROOT_DIR}"

python - <<'PY'
import torch
print("Python environment is ready")
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Activate later with: conda activate ${ENV_NAME}"
