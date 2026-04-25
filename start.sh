#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/ttt"
RULER_DIR="${SCRIPT_DIR}/../RULER"
TMPDIR="${SCRIPT_DIR}/.tmp"
XDG_CACHE_HOME="${SCRIPT_DIR}/.cache"
WHEEL_NAME="flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
LOCAL_WHEEL="${SCRIPT_DIR}/${WHEEL_NAME}"
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${WHEEL_NAME}"
RULER_INPUT="${RULER_DIR}/scripts/data/synthetic/json/PaulGrahamEssays.json"
DATA_OUTPUT="${SCRIPT_DIR}/data/paul_graham_essays.jsonl"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
export TMPDIR
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export XDG_CACHE_HOME
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${TRANSFORMERS_CACHE}"

echo "Repo directory: ${SCRIPT_DIR}"
echo "Virtualenv: ${VENV_DIR}"
echo "Temp directory: ${TMPDIR}"
echo "Cache directory: ${XDG_CACHE_HOME}"

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

node_major_version() {
  if ! command -v node >/dev/null 2>&1; then
    echo 0
    return
  fi

  node -p "process.versions.node.split('.')[0]" 2>/dev/null || echo 0
}

PYTHON_BIN="$(choose_python)"

echo "Using interpreter: ${PYTHON_BIN}"

apt-get update
apt-get install -y vim

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  apt-get install -y nodejs npm
fi

if (( $(node_major_version) < 18 )); then
  echo "Node.js 18+ is required for Codex, but found: $(node --version 2>/dev/null || echo missing)" >&2
  echo "Please install a newer Node.js runtime before running start.sh." >&2
  exit 1
fi

echo "Using Node.js: $(node --version)"
echo "Using npm: $(npm --version)"

npm install -g @openai/codex
hash -r

if ! command -v codex >/dev/null 2>&1; then
  echo "Codex install completed, but the 'codex' command is not on PATH." >&2
  exit 1
fi

codex --version

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python --version
pip cache purge >/dev/null 2>&1 || true
pip install --no-cache-dir --upgrade pip setuptools wheel

pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

if [[ -f "${LOCAL_WHEEL}" ]]; then
  FLASH_WHEEL="${LOCAL_WHEEL}"
else
  FLASH_WHEEL="${TMPDIR}/${WHEEL_NAME}"
  wget -O "${FLASH_WHEEL}" "${WHEEL_URL}"
fi

pip install --no-cache-dir "${FLASH_WHEEL}"

if [[ "${FLASH_WHEEL}" == "${TMPDIR}/${WHEEL_NAME}" ]]; then
  rm -f "${FLASH_WHEEL}"
fi

pip install --no-cache-dir "veomni @ git+https://github.com/ByteDance-Seed/VeOmni.git@9b91e164bea9e17f17ed490aab5e076c2335ca25"

pip install --no-cache-dir liger-kernel
pip install --no-cache-dir \
  wandb \
  tqdm \
  pyyaml \
  sentencepiece \
  safetensors \
  torchdata \
  blobfile \
  datasets \
  diffusers \
  tiktoken \
  timm
pip install --no-cache-dir transformers==4.57.3
pip install --no-cache-dir opt_einsum einops
pip cache purge >/dev/null 2>&1 || true
rm -rf "${TMPDIR:?}/"*

python - <<'PY'
import json
import pathlib
import veomni

p = pathlib.Path(veomni.__file__).resolve().parents[1] / "veomni-0.1.0.dist-info" / "direct_url.json"
print("veomni file:", veomni.__file__)
print("direct_url:", json.loads(p.read_text()) if p.exists() else "not found")
PY

if [[ ! -d "${RULER_DIR}" ]]; then
  git clone --depth 1 https://github.com/NVIDIA/RULER "${RULER_DIR}"
fi

pip install --no-cache-dir html2text beautifulsoup4

if [[ ! -f "${RULER_INPUT}" ]]; then
  pushd "${RULER_DIR}/scripts/data/synthetic/json" >/dev/null
  python download_paulgraham_essay.py
  popd >/dev/null
else
  echo "RULER input already exists at ${RULER_INPUT}; skipping download."
fi

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
echo "Codex CLI is available as: $(command -v codex)"
