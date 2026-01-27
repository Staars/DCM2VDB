#!/bin/bash
# Setup isolated environment for MedSAM2 conversion using uv

set -e

echo "=========================================="
echo "Setting up MedSAM2 Conversion Environment"
echo "=========================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ uv is installed"

# Create virtual environment
echo ""
echo "Creating virtual environment with uv..."
uv venv .venv

echo "✓ Virtual environment created at .venv"

# Activate and install dependencies
echo ""
echo "Installing dependencies..."
source .venv/bin/activate

uv pip install -r requirements_conversion.txt

echo "✓ Dependencies installed"

# Clone SAM2 repository to cache
echo ""
echo "Cloning SAM2 repository to cache..."
SAM2_DIR="../.cache/sam2_repo"

if [ -d "$SAM2_DIR" ]; then
    echo "✓ SAM2 repository already exists at $SAM2_DIR"
else
    git clone https://github.com/facebookresearch/segment-anything-2.git "$SAM2_DIR"
    echo "✓ SAM2 repository cloned to $SAM2_DIR"
fi

# Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the conversion:"
echo "  python convert_medsam2.py"
echo ""
echo "To deactivate:"
echo "  deactivate"
