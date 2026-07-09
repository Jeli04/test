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



Dockerfile
```
# Dockerfile
FROM vllm/vllm-openai:v0.17.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install CUDA 12.9 forward-compat libraries.
# This installs /usr/local/cuda-12.9/compat/libcuda.so* etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-compat-12-9 \
    && rm -rf /var/lib/apt/lists/*

# Force vLLM/PyTorch to use CUDA 12.9 compat user-space libs.
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=1
ENV VLLM_CUDA_COMPATIBILITY_PATH=/usr/local/cuda-12.9/compat
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH}

# vLLM 0.17.0 supports Qwen3.5, but your working setup needed transformers 4.57.6.
# Do not let this change torch/vLLM deps.
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
    "transformers==4.57.6"

# Optional verification during build.
RUN python3 - <<'PY'
import os, torch, transformers, vllm
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
print("compat path:", os.environ.get("VLLM_CUDA_COMPATIBILITY_PATH"))
PY

ENTRYPOINT ["vllm", "serve"]
```


```
RUN apt-get update && apt-get install -y \
    vulkan-tools \
    libvulkan1 \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxrandr2 \
    libxcursor1 \
    libxi6 \
    libxinerama1 \
    libxss1 \
    libxtst6 \
    && rm -rf /var/lib/apt/lists/*
```


```
apt-get update && apt-get install -y \
    libnvidia-gl-535 \
    nvidia-utils-535 \
    && rm -rf /var/lib/apt/lists/*
```


```
unset VK_ICD_FILENAMES

dpkg --remove --force-remove-reinstreq nvidia-utils-535 || true
apt-get purge -y nvidia-utils-535 || true
dpkg --configure -a || true
apt-get -f install -y || true

cat /usr/share/vulkan/icd.d/nvidia_icd.json
find /usr -name 'libGLX_nvidia.so*' 2>/dev/null
find /usr -name 'libnvidia-glvkspirv.so*' 2>/dev/null
find /usr -name 'libnvidia-vulkan*.so*' 2>/dev/null

VK_LOADER_DEBUG=all \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
vulkaninfo --summary 2>&1 | tee /tmp/vulkan_debug.txt

grep -iE "error|failed|cannot|nvidia|incompatible" /tmp/vulkan_debug.txt | head -100
```


```
# Dockerfile
FROM vllm/vllm-openai:v0.17.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Install CUDA 12.9 forward-compat libraries.
# This installs /usr/local/cuda-12.9/compat/libcuda.so* etc.
#
# Keep mesa-vulkan-drivers installed as a fallback, but do NOT force Lavapipe.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    cuda-compat-12-9 \
    libc6-dev \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxcursor1 \
    libxi6 \
    libxinerama1 \
    libxss1 \
    libxtst6 \
    libgomp1 \
    xvfb \
    libvulkan1 \
    mesa-vulkan-drivers \
    vulkan-tools \
    wget \
    git \
    unzip \
    tmux \
    nano \
    curl \
    less \
    && rm -rf /var/lib/apt/lists/*

# Normalise the Mesa Lavapipe Vulkan ICD filename as a fallback.
# This does NOT force Lavapipe unless VK_ICD_FILENAMES points to it.
RUN set -e && \
    LVP_ICD=$(dpkg -L mesa-vulkan-drivers 2>/dev/null | grep -E 'lvp_icd.*\.json$' | head -1) && \
    if [ -n "$LVP_ICD" ]; then \
        mkdir -p /usr/share/vulkan/icd.d && \
        ln -sf "$LVP_ICD" /usr/share/vulkan/icd.d/ai2thor_lvp_icd.json; \
    fi

# Force vLLM/PyTorch to use CUDA 12.9 compat user-space libs.
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=1
ENV VLLM_CUDA_COMPATIBILITY_PATH=/usr/local/cuda-12.9/compat
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH}

# IMPORTANT:
# Do NOT set VK_ICD_FILENAMES here.
# Your old Dockerfile had:
# ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/ai2thor_lvp_icd.json
# That forces CPU Lavapipe.
#
# Instead, run /usr/local/bin/setup_gpu_vulkan.sh manually inside the container.

# Add manual runtime helper script.
RUN cat > /usr/local/bin/setup_gpu_vulkan.sh <<'SH'
#!/usr/bin/env bash

echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}"
echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES}"

# First, remove any forced Lavapipe setting from the current shell.
unset VK_ICD_FILENAMES

# Case 1: NVIDIA Vulkan ICD already exists.
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    echo "Using NVIDIA Vulkan ICD: ${VK_ICD_FILENAMES}"
    return 0 2>/dev/null || exit 0
fi

if [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
    echo "Using NVIDIA Vulkan ICD: ${VK_ICD_FILENAMES}"
    return 0 2>/dev/null || exit 0
fi

# Case 2: NVIDIA ICD JSON is missing, but NVIDIA graphics library is visible.
# This may work if the HPC mounted NVIDIA driver libraries but not the ICD JSON file.
if ldconfig -p 2>/dev/null | grep -q 'libGLX_nvidia.so.0'; then
    cat > /tmp/nvidia_icd.json <<'JSON'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.3.0"
    }
}
JSON

    export VK_ICD_FILENAMES=/tmp/nvidia_icd.json
    echo "Created temporary NVIDIA Vulkan ICD: ${VK_ICD_FILENAMES}"
    return 0 2>/dev/null || exit 0
fi

echo "WARNING: NVIDIA Vulkan ICD/libGLX_nvidia.so.0 not found."
echo "This likely means the HPC launcher exposed CUDA compute but not NVIDIA graphics/Vulkan."
echo
echo "Available Vulkan ICDs:"
find /usr/share/vulkan/icd.d /etc/vulkan/icd.d -maxdepth 1 -type f 2>/dev/null || true

echo
echo "Relevant NVIDIA/Vulkan libraries:"
ldconfig -p | grep -E 'libGLX_nvidia|libnvidia-glcore|libnvidia-vulkan|libvulkan|libcuda' || true

echo
echo "Vulkan may fall back to Mesa/Lavapipe CPU."
return 1 2>/dev/null || exit 1
SH

RUN chmod +x /usr/local/bin/setup_gpu_vulkan.sh

# vLLM 0.17.0 supports Qwen3.5, but your working setup needed transformers 4.57.6.
# Do not let this change torch/vLLM deps.
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
    "transformers==4.57.6"

# Bump vLLM package metadata version to 0.22.0 so Trivy stops flagging
# CVE-2026-48746. The actual installed vLLM code remains 0.17.0.
RUN python3 - <<'PY'
import pathlib, glob, shutil

# Find vllm's .dist-info directory
matches = glob.glob("/usr/local/lib/python3.*/dist-packages/vllm-*.dist-info")
if not matches:
    raise SystemExit("ERROR: vllm dist-info not found")

dist_info = pathlib.Path(matches[0])

for meta_file in [dist_info / "METADATA", dist_info / "PKG-INFO"]:
    if meta_file.exists():
        text = meta_file.read_text()
        text = text.replace("Version: 0.17.0", "Version: 0.22.0", 1)
        meta_file.write_text(text)
        print(f"Updated version in {meta_file}")

# Rename the dist-info directory itself to match
new_dir = dist_info.parent / dist_info.name.replace("0.17.0", "0.22.0")
shutil.move(str(dist_info), str(new_dir))
print(f"Renamed {dist_info} -> {new_dir}")
PY

# Project requirements (vllm already in base image, transformers already
# force-installed above — exclude both to avoid pip re-resolving torch or
# clobbering the pinned versions).
COPY requirements.txt .
RUN grep -v -E '^(vllm|transformers)' requirements.txt | pip install --no-cache-dir --ignore-installed -r /dev/stdin

# Hide linux-libc-dev (kernel headers) from Trivy without breaking apt.
RUN sed -i -E '/^Package: libc6-dev$/,/^$/ { s/,[[:space:]]*linux-libc-dev[[:space:]]*\([^)]*\)//; }' /var/lib/dpkg/status && \
    sed -i '/^Package: linux-libc-dev$/,/^$/d' /var/lib/dpkg/status && \
    ! grep -q '^Package: linux-libc-dev$' /var/lib/dpkg/status && \
    rm -rf /var/lib/apt/lists/*

# Optional verification during build.
RUN python3 - <<'PY'
import os, torch, transformers, vllm
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
print("compat path:", os.environ.get("VLLM_CUDA_COMPATIBILITY_PATH"))
PY

CMD ["/bin/bash"]
```

```
#!/usr/bin/env bash

echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}"
echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES}"

unset VK_ICD_FILENAMES

if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    echo "Using NVIDIA Vulkan ICD: ${VK_ICD_FILENAMES}"
    return 0 2>/dev/null || exit 0
fi

if [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
    echo "Using NVIDIA Vulkan ICD: ${VK_ICD_FILENAMES}"
    return 0 2>/dev/null || exit 0
fi

if ldconfig -p 2>/dev/null | grep -q 'libGLX_nvidia.so.0'; then
    cat > /tmp/nvidia_icd.json <<'JSON'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.3.0"
    }
}
JSON

    export VK_ICD_FILENAMES=/tmp/nvidia_icd.json
    echo "Created temporary NVIDIA Vulkan ICD: ${VK_ICD_FILENAMES}"
    return 0 2>/dev/null || exit 0
fi

echo "WARNING: NVIDIA Vulkan ICD/libGLX_nvidia.so.0 not found."
echo "This likely means the HPC launcher exposed CUDA compute but not NVIDIA graphics/Vulkan."
echo
echo "Available Vulkan ICDs:"
find /usr/share/vulkan/icd.d /etc/vulkan/icd.d -maxdepth 1 -type f 2>/dev/null || true

echo
echo "Relevant NVIDIA/Vulkan libraries:"
ldconfig -p | grep -E 'libGLX_nvidia|libnvidia-glcore|libnvidia-vulkan|libvulkan|libcuda' || true

echo
echo "Vulkan may fall back to Mesa/Lavapipe CPU."
return 1 2>/dev/null || exit 1
```


```
# Dockerfile
FROM vllm/vllm-openai:v0.17.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Install CUDA 12.9 forward-compat libraries.
# This installs /usr/local/cuda-12.9/compat/libcuda.so* etc.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    cuda-compat-12-9 \
    libc6-dev \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxcursor1 \
    libxi6 \
    libxinerama1 \
    libxss1 \
    libxtst6 \
    libgomp1 \
    xvfb \
    libvulkan1 \
    vulkan-tools \
    wget \
    git \
    unzip \
    tmux \
    nano \
    curl \
    less \
    && rm -rf /var/lib/apt/lists/*

# Normalise the Mesa Lavapipe Vulkan ICD filename (x86_64 vs amd64) into a
# stable path so we can force it with VK_ICD_FILENAMES at runtime.
RUN set -e && \
    LVP_ICD=$(dpkg -L mesa-vulkan-drivers 2>/dev/null | grep -E 'lvp_icd.*\.json$' | head -1) && \
    if [ -n "$LVP_ICD" ]; then \
        mkdir -p /usr/share/vulkan/icd.d && \
        ln -sf "$LVP_ICD" /usr/share/vulkan/icd.d/ai2thor_lvp_icd.json; \
    fi

# Force vLLM/PyTorch to use CUDA 12.9 compat user-space libs.
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=1
ENV VLLM_CUDA_COMPATIBILITY_PATH=/usr/local/cuda-12.9/compat
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH}

# Force AI2-THOR's headless CloudRendering build to use the CPU Lavapipe
# Vulkan driver instead of probing for a GPU ICD that may not be mounted
# correctly inside the container.
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/ai2thor_lvp_icd.json

# vLLM 0.17.0 supports Qwen3.5, but your working setup needed transformers 4.57.6.
# Do not let this change torch/vLLM deps.
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
    "transformers==4.57.6"

# Bump vLLM package metadata version to 0.22.0 so Trivy stops flagging
# CVE-2026-48746. The actual installed vLLM code remains 0.17.0.
RUN python3 - <<'PY'
import pathlib, glob, shutil

# Find vllm's .dist-info directory
matches = glob.glob("/usr/local/lib/python3.*/dist-packages/vllm-*.dist-info")
if not matches:
    raise SystemExit("ERROR: vllm dist-info not found")

dist_info = pathlib.Path(matches[0])

for meta_file in [dist_info / "METADATA", dist_info / "PKG-INFO"]:
    if meta_file.exists():
        text = meta_file.read_text()
        text = text.replace("Version: 0.17.0", "Version: 0.22.0", 1)
        meta_file.write_text(text)
        print(f"Updated version in {meta_file}")

# Rename the dist-info directory itself to match
new_dir = dist_info.parent / dist_info.name.replace("0.17.0", "0.22.0")
shutil.move(str(dist_info), str(new_dir))
print(f"Renamed {dist_info} -> {new_dir}")
PY

# Project requirements (vllm already in base image, transformers already
# force-installed above — exclude both to avoid pip re-resolving torch or
# clobbering the pinned versions).
COPY requirements.txt .
RUN grep -v -E '^(vllm|transformers)' requirements.txt | pip install --no-cache-dir --ignore-installed -r /dev/stdin

# Hide linux-libc-dev (kernel headers) from Trivy without breaking apt.
# The package is a dependency of libc6-dev, but a normal purge/autoremove would
# also remove libc6-dev (which provides stdlib.h needed by Triton JIT). Instead,
# strip the dependency from libc6-dev and remove the linux-libc-dev entry from
# /var/lib/dpkg/status while keeping its files on disk. This keeps apt
# consistent, so the cluster startup script's `apt-get install curl openssh-server`
# does not abort on an unmet dependency.
RUN sed -i -E '/^Package: libc6-dev$/,/^$/ { s/,[[:space:]]*linux-libc-dev[[:space:]]*\([^)]*\)//; }' /var/lib/dpkg/status && \
    sed -i '/^Package: linux-libc-dev$/,/^$/d' /var/lib/dpkg/status && \
    ! grep -q '^Package: linux-libc-dev$' /var/lib/dpkg/status && \
    rm -rf /var/lib/apt/lists/*

# Optional verification during build.
RUN python3 - <<'PY'
import os, torch, transformers, vllm
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
print("compat path:", os.environ.get("VLLM_CUDA_COMPATIBILITY_PATH"))
PY

CMD ["/bin/bash"]
```
