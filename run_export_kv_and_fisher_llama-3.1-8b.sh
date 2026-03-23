#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7
set -euo pipefail

OUTPUT_DIR="/home/zz359/workspace-CQ-zzy/LLMSim-CQ-zzy/output/llama-3.1-8b-8c8b"

python export_kv_and_fisher.py \
  --model "meta-llama/Meta-Llama-3.1-8B" \
  --output_dir "${OUTPUT_DIR}" \
  --num_samples 16 \
  --max_seq_len 2048 \
  --num_coupled_channels 8 \
  --num_bits 8 \
  --dataset "wikitext" \
  --dataset_config "wikitext-2-raw-v1"