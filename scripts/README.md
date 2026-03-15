# Training Scripts for Persistent Execution

These scripts are designed to be run with `qwarium-agent proc spawn` for persistent training that survives SSH session disconnects.

## Available Scripts

| Script | Purpose | Config Directory |
|--------|---------|------------------|
| `run_r_clm.sh` | CLM model training | `conf/r_clm/` |
| `run_r_clm_from_scratch.sh` | CLM model training from scratch | `conf/r_clm/` |
| `run_r_detect.sh` | AI detection model training | `conf/r_detect/` |
| `run_r_ranking.sh` | Ranking model training | `conf/r_ranking/` |
| `run_r_embed.sh` | Embedding model training | `conf/r_embed/` |
| `run_r_dpo.sh` | DPO (Direct Preference Optimization) training | `conf/r_dpo/` |

## Usage

### Basic Usage
```bash
# Make scripts executable (first time only)
chmod +x /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/*.sh

# Run with default config
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_detect.sh
```

### With Custom Config
```bash
# Specify config name (without .yaml extension)
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_detect.sh conf_r_detect_competition
```

### Examples

```bash
# Run r_detect with competition config
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_detect.sh conf_r_detect_competition

# Run r_detect with mini config
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_detect.sh conf_r_detect_mini

# Run r_ranking with large config
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_ranking.sh conf_r_ranking_large

# Run r_embed with default config
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_embed.sh

# Run r_clm with tiny_llama config
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_clm.sh conf_r_clm_tiny_llama
```

## Output Locations

### Models (Persistent Storage)
- r_detect: `/qwarium/home/d.a.lanovenko/models/r_detect_<config_name>/`
- r_ranking: `/qwarium/home/d.a.lanovenko/models/r_ranking_<config_name>/`
- r_embed: `/qwarium/home/d.a.lanovenko/models/r_embed_<config_name>/`
- r_dpo: `/qwarium/home/d.a.lanovenko/models/r_dpo_<config_name>/`
- r_clm: `/tmp/hydra_r_clm/` (Hydra output)

### Logs (Persistent Storage)
All logs are saved to: `/qwarium/home/d.a.lanovenko/llm-detect-ai/logs/`

## Environment Variables

The following environment variables are set automatically:
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` - Memory optimization
- `HF_HOME=/tmp/huggingface_cache` - HuggingFace cache location
- `HF_DATASETS_CACHE=/tmp/hf_datasets_cache` - Datasets cache location

## Available Configs

### r_detect
- `conf_r_detect_competition` - Competition model
- `conf_r_detect_mini` - Mini model
- `conf_r_detect_mix_v16` - Mix v16 model
- `conf_r_detect_mix_v26` - Mix v26 model
- `conf_r_detect_transfer` - Transfer learning model

### r_clm / r_clm_from_scratch
- `conf_r_clm` - Default CLM config
- `conf_r_clm_tiny_llama` - Tiny Llama
- `conf_r_clm_lite_llama` - Lite Llama
- `conf_r_clm_llama13b` - Llama 13B
- `conf_r_clm_gpt2` - GPT-2
- `conf_r_clm_mistral_persuade` - Mistral Persuade
- And more...

### r_ranking
- `conf_r_ranking_large` - Large ranking model

### r_embed
- `conf_r_embed` - Default embedding model

### r_dpo
- `conf_r_dpo` - Default DPO config
