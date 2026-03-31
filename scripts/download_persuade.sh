#!/bin/bash
# Script to download PERSUADE 2.0 dataset for CLM training
# Alternative: nbroad/persaude-corpus-2 (most popular alternative)
# Original: https://www.kaggle.com/datasets/marklvl/persuade-20-human-scores (403 Forbidden)
# Alternative: https://www.kaggle.com/datasets/nbroad/persaude-corpus-2

set -e

DATASETS_DIR="/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets"
TMP_DIR="/tmp/datasets/persuade"

echo "=== Downloading PERSUADE 2.0 Dataset (Alternative) ==="
echo "Using: nbroad/persaude-corpus-2 (4064 downloads, 131 votes)"
echo "Temporary storage: $TMP_DIR"
echo "Final location: $DATASETS_DIR"
echo ""

# Create directories
mkdir -p "$TMP_DIR"
mkdir -p "$DATASETS_DIR"

# Check if dataset already exists
PERSUADE_CSV="$DATASETS_DIR/persuade_2.0_human_scores_demo_id_github.csv"

if [ ! -f "$PERSUADE_CSV" ]; then
    echo "Downloading PERSUADE dataset from Kaggle..."
    
    # Download from Kaggle (alternative dataset)
    kaggle datasets download -d nbroad/persaude-corpus-2 -p "$TMP_DIR"
    
    # Extract
    echo "Extracting dataset..."
    unzip -o "$TMP_DIR/persaude-corpus-2.zip" -d "$TMP_DIR"
    
    # List files to see what we have
    echo "Files in extracted archive:"
    ls -la "$TMP_DIR"
    
    # Find CSV files
    echo ""
    echo "CSV files found:"
    find "$TMP_DIR" -name "*.csv" -type f
    
    # Copy the main CSV file (adjust name based on actual file)
    # The alternative dataset may have different column names
    FIRST_CSV=$(find "$TMP_DIR" -name "*.csv" -type f | head -1)
    
    if [ -n "$FIRST_CSV" ]; then
        cp "$FIRST_CSV" "$PERSUADE_CSV"
        echo ""
        echo "Dataset copied to: $PERSUADE_CSV"
    else
        echo "ERROR: No CSV file found in archive!"
        exit 1
    fi
    
    # Cleanup zip
    rm -f "$TMP_DIR/persaude-corpus-2.zip"
    
else
    echo "PERSUADE dataset already exists at: $PERSUADE_CSV"
fi

echo ""
echo "=== Dataset Preview ==="
echo "First 3 lines:"
head -3 "$PERSUADE_CSV" | cut -c1-300
echo ""

echo "=== Column Names ==="
head -1 "$PERSUADE_CSV" | tr ',' '\n' | nl
echo ""

echo "=== Dataset Ready for CLM Training ==="
echo ""
echo "NOTE: You may need to adjust column names in ai_dataset.py to match this dataset"
