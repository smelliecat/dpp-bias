#!/usr/bin/env bash
# Demonstration-placement sweep from "Where to show Demos in Your Prompt:
# A Positional Bias of In-Context Learning" (EMNLP 2025).
#
# For each model: eight tasks x four demonstration positions (ssp, esp, sum, eum),
# five demonstrations, 200 questions per task.
#
# Usage:
#   bash scripts/run_placement_study.sh                      # the ten models in the paper
#   bash scripts/run_placement_study.sh Qwen_1.5B Qwen_7B    # a subset (keys from jsons/model_templates.json)
#
# Environment variables (all optional):
#   OUTPUT_ROOT   where results are written            (default: outputs)
#   TASKS_FILE    evaluation sets                       (default: jsons/tasks200_full.json if it exists,
#                                                        otherwise jsons/tasks200.json)
#   SIZE          questions per task                    (default: 200)
#
# Gated checkpoints need a Hugging Face login: run `huggingface-cli login` once, or
# export HF_TOKEN in your shell. Never write a token into this file.
#
# On a SLURM cluster, submit this script with your own sbatch options, e.g.
#   sbatch --gres=gpu:1 --mem=100G --time=3-00:00:00 scripts/run_placement_study.sh

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/src/runner.py" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"

# runner.py imports the bundled library as `src.llamafactory`, so the repository root must be importable.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=(
        Qwen_1.5B
        Qwen_7B
        LLAMA3_3B
        LLAMA3_8B
        Cohere_8B
        Mistral_7B
        Cohere_32B
        LLAMA3_70B
        Mistral_8x7B
        Qwen_72B
    )
fi

SIZE="${SIZE:-200}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
if [ -z "${TASKS_FILE:-}" ]; then
    if [ -f jsons/tasks200_full.json ]; then
        TASKS_FILE=jsons/tasks200_full.json
    else
        TASKS_FILE=jsons/tasks200.json
        echo "Note: jsons/tasks200_full.json not found, so ag_news, xsum and cnn_dailymail will be skipped." >&2
        echo "      Run 'python scripts/rebuild_news_sets.py' first to include them." >&2
    fi
fi
DATE="$(date "+%m%d%y_%H_%M_%S")"

python --version
python src/runner.py \
    --models "${MODELS[@]}" \
    --prompts samppm \
    --tasks ag_news mnli arc mmlu squad gsm8k xsum cnn_dailymail \
    --prompt_structure icld_ssp icld_esp icld_sum icld_eum \
    --prompt_file jsons/prompts.json \
    --model_template_file jsons/model_templates.json \
    --gen_config_file jsons/gen_configs.json \
    --input_path "${TASKS_FILE}" \
    --size "${SIZE}" \
    --output_path "${OUTPUT_ROOT}/results_${SIZE}_${DATE}"
