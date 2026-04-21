# GuuGo training container.
#
# Based on NVIDIA's NGC PyTorch image, which ships a torch build
# compiled with Blackwell (sm_100/sm_120) kernels - the plain PyPI
# torch wheels do not include those, so on DGX Spark / Blackwell this
# image is the supported path. We do NOT create a venv: the NGC image
# already provides a properly-configured system Python with torch,
# CUDA, cuDNN, and NCCL wired up, and `pip install torch` would in
# fact break it by replacing that build with a stock wheel.
#
# Build:
#   docker build -t guugo-train .
#
# Run:
#   docker run --rm -it --gpus all --ipc=host \
#       -v "$PWD":/workspace -w /workspace \
#       guugo-train \
#       python scripts/automated_training.py
#
# Notes:
#   --gpus all   expose every GPU; use --gpus '"device=0"' to pin one.
#   --ipc=host   required for torch.multiprocessing shared memory; the
#                default /dev/shm in docker is 64 MB and the worker
#                pool will die with bus errors. Alternative: --shm-size=8g.
#   -v $PWD:/workspace  keeps checkpoints/ and replay/ on the host so
#                they survive container restarts and can be rsync'd.

ARG NGC_PYTORCH_TAG=25.03-py3
FROM nvcr.io/nvidia/pytorch:${NGC_PYTORCH_TAG}

# Fail the build if anyone tries to sneak torch in.
# We only need the lightweight runtime bits; torch + numpy are already
# in the base image. pygame is intentionally skipped - the GUI is never
# run from inside a GPU container.
RUN pip install --no-cache-dir \
        "pytest>=8.0"

WORKDIR /workspace

# Default to an interactive shell so the container is a handy dev box;
# overridden at `docker run` time when you want to launch the script
# directly.
CMD ["bash"]
