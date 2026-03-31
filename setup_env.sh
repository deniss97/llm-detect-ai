#!/bin/bash
# Setup script for translation and generation pipeline
# This script installs required dependencies

set -e

echo "=============================================="
echo "Setting up environment for translation/generation"
echo "=============================================="

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found!"
    exit 1
fi

# Install from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip3 install -r requirements.txt --quiet

# Verify installation
echo ""
echo "Verifying installation..."

python3 -c "
import sys
try:
    from deep_translator import GoogleTranslator
    print('✓ deep_translator installed')
except ImportError:
    print('✗ deep_translator NOT installed')
    sys.exit(1)

try:
    import pandas
    print('✓ pandas installed')
except ImportError:
    print('✗ pandas NOT installed')
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM
    print('✓ transformers installed')
except ImportError:
    print('✗ transformers NOT installed')
    sys.exit(1)

print('')
print('All dependencies installed successfully!')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Usage:"
echo "  # Quick test (3 essays, 1 variation each)"
echo "  python3 ./code/r_clm/translate_and_generate.py \\"
echo "      --input datasets/Датасет.csv \\"
echo "      --model_path /tmp/models/r_clm_v2/last \\"
echo "      --max_essays 3 \\"
echo "      --num_variations 1"
echo ""
echo "  # Full generation (all essays, 5 variations each)"
echo "  python3 ./code/r_clm/translate_and_generate.py \\"
echo "      --input datasets/Датасет.csv \\"
echo "      --model_path /tmp/models/r_clm_v2/last \\"
echo "      --max_essays -1 \\"
echo "      --num_variations 5"
echo ""
