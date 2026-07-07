#!/usr/bin/env bash
set -euo pipefail

# setup_vllm_qwen35_no_venv.sh
# Purpose: Try to run Qwen3.5 with vLLM on an NVIDIA 535 server driver using cuda-compat-12-9,
#          WITHOUT creating a separate virtualenv/conda environment.
#
# WARNING:
# - This modifies your user-level Python packages with pip --user.
# - It intentionally avoids a separate environment because requested, but this is more fragile.
# - If a system/global flash-attn exists in /usr/local/lib/python*/dist-packages, it can still win
#   depending on your PYTHONPATH/sys.path. This script tries to avoid that by clearing PYTHONPATH.
# - nvidia-smi may still report CUDA 12.5/12.x. That is normal; cuda-compat is user-space only.

PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_VERSION="${VLLM_VERSION:-0.17.0}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
CUDA_COMPAT_PATH="${CUDA_COMPAT_PATH:-/usr/local/cuda-12.9/compat}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"   # set to 1 to rebuild flash-attn from source in --user site
MAX_JOBS="${MAX_JOBS:-4}"

export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_CUDA_COMPATIBILITY_PATH="$CUDA_COMPAT_PATH"
export LD_LIBRARY_PATH="$CUDA_COMPAT_PATH:${LD_LIBRARY_PATH:-}"

# Avoid accidentally importing packages from manually-set paths.
unset PYTHONPATH || true

log() { echo -e "\n[setup] $*"; }

log "Python executable: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
log "Using CUDA compat path: $CUDA_COMPAT_PATH"

if [[ ! -d "$CUDA_COMPAT_PATH" ]]; then
  echo "ERROR: CUDA compat path does not exist: $CUDA_COMPAT_PATH" >&2
  echo "Install it first, e.g. sudo apt-get install cuda-compat-12-9" >&2
  exit 1
fi

log "CUDA compat files:"
ls -lh "$CUDA_COMPAT_PATH" | head -30 || true

log "Upgrading pip tooling in user site"
$PYTHON_BIN -m pip install --user -U pip setuptools wheel packaging ninja

log "Removing user-site flash-attn if present. System flash-attn may require admin/root to remove."
$PYTHON_BIN -m pip uninstall -y flash-attn flash_attn || true

log "Installing vLLM ${VLLM_VERSION} in user site"
# Extra PyTorch cu129 index is included because newer vLLM/Qwen3.5 stacks often need CUDA 12.9 PyTorch wheels.
$PYTHON_BIN -m pip install --user --no-cache-dir \
  "vllm==${VLLM_VERSION}" \
  --extra-index-url https://download.pytorch.org/whl/cu129

if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
  log "Rebuilding flash-attn against the currently installed Torch. This can take a while."
  MAX_JOBS="$MAX_JOBS" $PYTHON_BIN -m pip install --user --no-cache-dir --force-reinstall \
    --no-build-isolation --no-binary flash-attn flash-attn
else
  log "Skipping flash-attn rebuild. If you still get flash_attn_2_cuda undefined symbol, rerun with: INSTALL_FLASH_ATTN=1 bash $0"
fi

log "Environment/version diagnostic"
$PYTHON_BIN - <<'PY'
import os, sys, importlib.util, ctypes
print("python:", sys.executable)
print("sys.path first 10:")
for p in sys.path[:10]:
    print("  ", p)
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("VLLM_ENABLE_CUDA_COMPATIBILITY:", os.environ.get("VLLM_ENABLE_CUDA_COMPATIBILITY"))
print("VLLM_CUDA_COMPATIBILITY_PATH:", os.environ.get("VLLM_CUDA_COMPATIBILITY_PATH"))

try:
    lib = ctypes.CDLL("libcuda.so.1")
    print("Loaded libcuda.so.1:", lib)
except Exception as e:
    print("Failed to load libcuda.so.1:", repr(e))

for name in ["torch", "vllm", "transformers", "flash_attn"]:
    spec = importlib.util.find_spec(name)
    print(f"{name} spec:", spec.origin if spec else None)

try:
    import torch
    print("torch:", torch.__version__)
    print("torch cuda build:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch import/check failed:", repr(e))

try:
    import vllm
    print("vllm:", vllm.__version__)
except Exception as e:
    print("vllm import failed:", repr(e))

try:
    import transformers
    print("transformers:", transformers.__version__)
except Exception as e:
    print("transformers import failed:", repr(e))
PY

cat > ./run_qwen35_vllm.sh <<RUNEOF
#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH || true
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_CUDA_COMPATIBILITY_PATH="$CUDA_COMPAT_PATH"
export LD_LIBRARY_PATH="$CUDA_COMPAT_PATH:\${LD_LIBRARY_PATH:-}"

# Edit MODEL_PATH or pass MODEL_PATH=/path/to/model when running setup script.
vllm serve "$MODEL_PATH" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --reasoning-parser qwen3
RUNEOF
chmod +x ./run_qwen35_vllm.sh

log "Wrote ./run_qwen35_vllm.sh"
log "To run: ./run_qwen35_vllm.sh"
log "If you hit flash_attn undefined symbol again: INSTALL_FLASH_ATTN=1 bash $0"
