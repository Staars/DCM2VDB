#!/bin/bash
# Download GPU acceleration wheels for bundling with Blender extension

set -e

WHEELS_DIR="extension/wheels"
mkdir -p "$WHEELS_DIR"

echo "Downloading GPU acceleration wheels..."
echo "======================================"

# MLX for macOS ARM64 (Apple Silicon)
echo ""
echo "1. MLX for macOS ARM64 (M1/M2/M3/M4)..."
python3 -m pip download mlx==0.30.3 \
    --platform macosx_14_0_arm64 \
    --python-version 311 \
    --only-binary=:all: \
    --no-deps \
    -d "$WHEELS_DIR"

python3 -m pip download mlx-metal==0.30.3 \
    --platform macosx_14_0_arm64 \
    --python-version 311 \
    --only-binary=:all: \
    --no-deps \
    -d "$WHEELS_DIR"

# CuPy for Windows (CUDA 12.x)
echo ""
echo "2. CuPy for Windows (NVIDIA CUDA 12.x)..."
python3 -m pip download cupy-cuda12x==13.6.0 \
    --platform win_amd64 \
    --python-version 311 \
    --only-binary=:all: \
    --no-deps \
    -d "$WHEELS_DIR"

# CuPy for Linux x86_64 (CUDA 12.x)
echo ""
echo "3. CuPy for Linux x86_64 (NVIDIA CUDA 12.x)..."
python3 -m pip download cupy-cuda12x==13.6.0 \
    --platform manylinux2014_x86_64 \
    --python-version 311 \
    --only-binary=:all: \
    --no-deps \
    -d "$WHEELS_DIR"

echo ""
echo "======================================"
echo "Download complete!"
echo ""
echo "Downloaded wheels:"
ls -lh "$WHEELS_DIR"/*.whl | awk '{print "  " $9 " - " $5}'

echo ""
echo "Total size:"
du -sh "$WHEELS_DIR"

echo ""
echo "Next steps:"
echo "1. Update extension/__init__.py to install these wheels"
echo "2. Test in Blender"
