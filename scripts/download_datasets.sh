#!/bin/bash
# Script to download large external datasets to /tmp and create symlinks
# This ensures datasets persist in tmp during training sessions
# Datasets will survive SSH disconnects when downloaded to /tmp

set -e

TMP_BASE="/tmp/datasets/external"
LINK_BASE="/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/external"

echo "=== Downloading External Datasets ==="
echo "Temporary storage: $TMP_BASE"
echo "Symlink location: $LINK_BASE"
echo ""

# Create directories
mkdir -p "$TMP_BASE"
mkdir -p "$LINK_BASE"

# --- Dataset 1: ai_mix_for_ranking (ai-bin7-mix-v1) ---
# For ranking model (DeBERTa-large) and embed model
echo "=== 1. Downloading ai_mix_for_ranking (ai-bin7-mix-v1) ==="
RANKING_ZIP="$TMP_BASE/ai-bin7-mix-v1.zip"
RANKING_DIR="$TMP_BASE/ai_mix_for_ranking"

if [ ! -d "$RANKING_DIR" ]; then
    echo "Downloading ranking dataset from Kaggle..."
    kaggle datasets download -d conjuring92/ai-bin7-mix-v1 -p "$TMP_BASE"
    mkdir -p "$RANKING_DIR"
    unzip -o "$RANKING_ZIP" -d "$RANKING_DIR"
    rm "$RANKING_ZIP"
    echo "Downloaded and extracted to $RANKING_DIR"
else
    echo "Ranking dataset already exists at $RANKING_DIR"
fi

# Create symlink
ln -sf "$RANKING_DIR" "$LINK_BASE/ai_mix_for_ranking"
echo "Symlink created: $LINK_BASE/ai_mix_for_ranking -> $RANKING_DIR"
echo ""

# --- Dataset 2: ai_mix_v16 ---
# For Detect Mix v16 model (Mistral-7B + LoRA r=8, 16 epochs)
echo "=== 2. Downloading ai_mix_v16 ==="
MIX16_ZIP="$TMP_BASE/ai-mix-v16.zip"
MIX16_DIR="$TMP_BASE/ai_mix_v16"

if [ ! -d "$MIX16_DIR" ]; then
    echo "Downloading mix_v16 dataset from Kaggle..."
    kaggle datasets download -d conjuring92/ai-mix-v16 -p "$TMP_BASE"
    mkdir -p "$MIX16_DIR"
    unzip -o "$MIX16_ZIP" -d "$MIX16_DIR"
    rm "$MIX16_ZIP"
    echo "Downloaded and extracted to $MIX16_DIR"
else
    echo "Mix v16 dataset already exists at $MIX16_DIR"
fi

# Create symlink
ln -sf "$MIX16_DIR" "$LINK_BASE/ai_mix_v16"
echo "Symlink created: $LINK_BASE/ai_mix_v16 -> $MIX16_DIR"
echo ""

# --- Dataset 3: ai_mix_v26 ---
# For Detect Mix v26 model (Mistral-7B + LoRA r=16, 16 epochs) and Embed model
echo "=== 3. Downloading ai_mix_v26 ==="
MIX26_ZIP="$TMP_BASE/ai-mix-v26.zip"
MIX26_DIR="$TMP_BASE/ai_mix_v26"

if [ ! -d "$MIX26_DIR" ]; then
    echo "Downloading mix_v26 dataset from Kaggle..."
    kaggle datasets download -d conjuring92/ai-mix-v26 -p "$TMP_BASE"
    mkdir -p "$MIX26_DIR"
    unzip -o "$MIX26_ZIP" -d "$MIX26_DIR"
    rm "$MIX26_ZIP"
    echo "Downloaded and extracted to $MIX26_DIR"
else
    echo "Mix v26 dataset already exists at $MIX26_DIR"
fi

# Create symlink
ln -sf "$MIX26_DIR" "$LINK_BASE/ai_mix_v26"
echo "Symlink created: $LINK_BASE/ai_mix_v26 -> $MIX26_DIR"
echo ""

echo "=== Download Complete ==="
echo ""
echo "Datasets location (symlinks):"
ls -la "$LINK_BASE/"
echo ""
echo "Actual data location:"
ls -la "$TMP_BASE/"
echo ""
echo "=== Ready for Training ==="
