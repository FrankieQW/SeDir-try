#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="${1:-${ENV_NAME:-SeDir}}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Using existing conda environment: ${ENV_NAME}"
else
  echo "Creating conda environment: ${ENV_NAME}"
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

echo "Installing CUDA 11.7 toolkit and Linux compilers used by MC3D-AD extensions..."
conda install -y \
  -c nvidia/label/cuda-11.7.0 \
  -c conda-forge \
  cuda-toolkit=11.7 \
  gcc_linux-64=9.5 \
  gxx_linux-64=9.5

echo "Installing project Python dependencies..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${ROOT_DIR}/requirements.txt"
python -m pip install pytest

echo "Installing PyTorch 1.13.0 for CUDA 11.7..."
python -m pip install --force-reinstall \
  torch==1.13.0+cu117 \
  torchvision==0.14.0+cu117 \
  torchaudio==0.13.0 \
  --extra-index-url https://download.pytorch.org/whl/cu117

echo "Installing PointNet++ CUDA ops..."
python -m pip install \
  "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

echo "Installing KNN_CUDA..."
python -m pip install --upgrade \
  "https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl"

echo
echo "Environment ${ENV_NAME} is installed."
echo "Before training, run:"
echo "  conda activate ${ENV_NAME}"
echo "  source ${ROOT_DIR}/env.sh"
