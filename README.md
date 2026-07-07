python3 -m pip install --user --no-cache-dir --force-reinstall \
  "vllm==0.17.0" \
  --extra-index-url https://download.pytorch.org/whl/cu129


```
# ================================
# Qwen3.5 + vLLM 0.17.0 + CUDA 12.9 compat clean reinstall
# No conda / no venv, user-level install
# ================================

# 0. Set CUDA 12.9 compat paths
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_CUDA_COMPATIBILITY_PATH=/usr/local/cuda-12.9/compat
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH

# Check compat path exists
ls -lh /usr/local/cuda-12.9/compat || {
  echo "ERROR: /usr/local/cuda-12.9/compat not found. Install cuda-compat-12-9 first."
  exit 1
}

# 1. Get user site-packages path
USER_SITE=$(python3 -m site --user-site)
echo "USER_SITE=$USER_SITE"

# 2. Uninstall incompatible / stale packages from user install
python3 -m pip uninstall -y \
  vllm vllm-flash-attn \
  torch torchvision torchaudio triton pytorch-triton torch-tensorrt \
  xformers \
  flash-attn flash_attn

# 3. Manually remove stale user-site package folders
# This is important because pip uninstall can leave old torch files behind.
rm -rf "$USER_SITE/torch" \
       "$USER_SITE/torch-"*.dist-info \
       "$USER_SITE/torchgen" \
       "$USER_SITE/functorch" \
       "$USER_SITE/torchvision" \
       "$USER_SITE/torchvision-"*.dist-info \
       "$USER_SITE/torchaudio" \
       "$USER_SITE/torchaudio-"*.dist-info \
       "$USER_SITE/triton" \
       "$USER_SITE/triton-"*.dist-info \
       "$USER_SITE/vllm" \
       "$USER_SITE/vllm-"*.dist-info \
       "$USER_SITE/xformers" \
       "$USER_SITE/xformers-"*.dist-info \
       "$USER_SITE/flash_attn" \
       "$USER_SITE/flash_attn-"*.dist-info \
       "$USER_SITE/flash_attn-"*.egg-info

# 4. Clear Torch / Triton compile caches
rm -rf ~/.cache/torch/inductor
rm -rf ~/.cache/torch_extensions
rm -rf ~/.triton
rm -rf ~/.cache/triton

# 5. Upgrade pip tooling
python3 -m pip install --user -U pip setuptools wheel packaging ninja

# 6. Install PyTorch cu129 stack expected by vLLM 0.17.0
python3 -m pip install --user --no-cache-dir --force-reinstall \
  "torch==2.10.0" \
  "torchvision==0.25.0" \
  "torchaudio==2.10.0" \
  --index-url https://download.pytorch.org/whl/cu129

# 7. Install vLLM 0.17.0
python3 -m pip install --user --no-cache-dir --force-reinstall \
  "vllm==0.17.0" \
  --extra-index-url https://download.pytorch.org/whl/cu129

# 8. Pin Transformers to avoid the AuxRequest / flex_attention issue
python3 -m pip install --user --no-cache-dir --force-reinstall --no-deps \
  "transformers==4.56.2"

# 9. Keep flashinfer at the vLLM-pinned version
python3 -m pip install --user --no-cache-dir --force-reinstall \
  "flashinfer-python==0.6.4"

# 10. Optional: install FlashAttention only if your model/vLLM path needs it.
# If this takes too long, you can skip this block first and try vLLM without flash-attn.
MAX_JOBS=4 python3 -m pip install --user --no-cache-dir --force-reinstall \
  --ignore-installed \
  --no-build-isolation \
  "flash-attn==2.8.3.post1"

# 11. Verify imports and paths
python3 - <<'PY'
import os, sys, site, importlib.util

print("python:", sys.executable)
print("user site:", site.getusersitepackages())
print("compat enabled:", os.environ.get("VLLM_ENABLE_CUDA_COMPATIBILITY"))
print("compat path:", os.environ.get("VLLM_CUDA_COMPATIBILITY_PATH"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))

import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torch file:", torch.__file__)
print("cuda available:", torch.cuda.is_available())

import transformers
print("transformers:", transformers.__version__)
print("transformers file:", transformers.__file__)

import vllm
print("vllm:", vllm.__version__)
print("vllm file:", vllm.__file__)

import flashinfer
print("flashinfer file:", flashinfer.__file__)

spec = importlib.util.find_spec("flash_attn")
print("flash_attn spec:", spec.origin if spec else None)
if spec:
    import flash_attn
    print("flash_attn version:", getattr(flash_attn, "__version__", "unknown"))

import torch._inductor
import torch._dynamo
print("torch inductor/dynamo import OK")
PY

# 12. Run vLLM
vllm serve /group-volume/.dataset-0af49006-85e2-3498-9b25-f78df9e3d8b4/datasets/users/jerry.li/models/Qwen3.5-9B \
  --dtype bfloat16 \
  --max-model-len 32768
```
