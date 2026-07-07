python3 -m pip install --user --no-cache-dir --force-reinstall \
  "vllm==0.17.0" \
  --extra-index-url https://download.pytorch.org/whl/cu129


```
USER_SITE=$(python3 -m site --user-site)
echo "$USER_SITE"

python3 -m pip uninstall -y torch torchvision torchaudio triton pytorch-triton torch-tensorrt

rm -rf "$USER_SITE/torch" \
       "$USER_SITE/torch-"*.dist-info \
       "$USER_SITE/torchgen" \
       "$USER_SITE/functorch" \
       "$USER_SITE/torchvision" \
       "$USER_SITE/torchvision-"*.dist-info \
       "$USER_SITE/torchaudio" \
       "$USER_SITE/torchaudio-"*.dist-info \
       "$USER_SITE/triton" \
       "$USER_SITE/triton-"*.dist-info

rm -rf ~/.cache/torch/inductor
rm -rf ~/.cache/torch_extensions
rm -rf ~/.triton
rm -rf ~/.cache/triton

python3 -m pip install --user --no-cache-dir --force-reinstall \
  "torch==2.10.0" \
  "torchvision==0.25.0" \
  "torchaudio==2.10.0" \
  --index-url https://download.pytorch.org/whl/cu129

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch file:", torch.__file__)
from torch._inductor.kernel import flex_attention
print("Torch flex_attention import OK")
PY


python3 -m pip install --user --no-cache-dir --force-reinstall \
  "vllm==0.17.0" \
  --extra-index-url https://download.pytorch.org/whl/cu129


python3 -m pip install --user --no-cache-dir --force-reinstall --no-deps \
  "transformers==4.56.2"


export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_CUDA_COMPATIBILITY_PATH=/usr/local/cuda-12.9/compat
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH
```
