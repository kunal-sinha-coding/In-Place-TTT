#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/ttt"
RULER_DIR="${SCRIPT_DIR}/../RULER"
WHEEL_NAME="flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
LOCAL_WHEEL="${SCRIPT_DIR}/${WHEEL_NAME}"
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${WHEEL_NAME}"
RULER_INPUT="${RULER_DIR}/scripts/data/synthetic/json/PaulGrahamEssays.json"
DATA_OUTPUT="${SCRIPT_DIR}/data/paul_graham_essays.jsonl"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

choose_python() {
  if command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi

  echo "python3 is required but was not found." >&2
  exit 1
}

PYTHON_BIN="$(choose_python)"

echo "Using interpreter: ${PYTHON_BIN}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python --version
pip install --upgrade pip setuptools wheel

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

if [[ -f "${LOCAL_WHEEL}" ]]; then
  FLASH_WHEEL="${LOCAL_WHEEL}"
else
  FLASH_WHEEL="/tmp/${WHEEL_NAME}"
  wget -O "${FLASH_WHEEL}" "${WHEEL_URL}"
fi

pip install "${FLASH_WHEEL}"

if [[ "${FLASH_WHEEL}" == "/tmp/${WHEEL_NAME}" ]]; then
  rm -f "${FLASH_WHEEL}"
fi

pip install "veomni @ git+https://github.com/ByteDance-Seed/VeOmni.git@9b91e164bea9e17f17ed490aab5e076c2335ca25"

pip install liger-kernel
pip install wandb torchdata blobfile datasets diffusers tiktoken timm
pip install transformers==4.57.3
pip install opt_einsum einops

python - <<'PY'
import json
import pathlib
import veomni

p = pathlib.Path(veomni.__file__).resolve().parents[1] / "veomni-0.1.0.dist-info" / "direct_url.json"
print("veomni file:", veomni.__file__)
print("direct_url:", json.loads(p.read_text()) if p.exists() else "not found")
PY

if [[ ! -d "${RULER_DIR}" ]]; then
  git clone https://github.com/NVIDIA/RULER "${RULER_DIR}"
fi

pushd "${RULER_DIR}/scripts/data/synthetic/json" >/dev/null
python download_paulgraham_essay.py
popd >/dev/null

if [[ ! -f "${RULER_INPUT}" ]]; then
  echo "Missing RULER input file: ${RULER_INPUT}" >&2
  echo "RULER download step did not produce the expected file." >&2
  exit 1
fi

python "${SCRIPT_DIR}/scripts/convert_paul_graham_essays.py" \
  --input "${RULER_INPUT}" \
  --output "${DATA_OUTPUT}"

echo
echo "Setup complete."
echo "Activate the environment with:"
echo "  source \"${VENV_DIR}/bin/activate\""
