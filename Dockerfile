# 1. Base Image: NVIDIA CUDA 12.4 Runtime on Ubuntu 22.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 2. Prevent timezone/keyboard prompts from freezing the build
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install Python 3.10, pip, and system essentials
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Alias python and pip for convenience
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 5. Set the working directory
WORKDIR /workspace

# 6. Copy ONLY the requirements file to keep the image perfectly clean
COPY requirements.txt /workspace/

# 7. Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# 8. Install requirements, forcing PyTorch to pull the CUDA 12.4 GPU binaries
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124