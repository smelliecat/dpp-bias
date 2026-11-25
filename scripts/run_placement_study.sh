#!/usr/bin/env bash
# Demonstration-placement sweep from "Where to show Demos in Your Prompt:
# A Positional Bias of In-Context Learning" (EMNLP 2025).
#
# For each model: up to eight tasks x four demonstration positions (ssp, esp, sum, eum),
# five demonstrations, 200 questions per task. Runs on one machine with an NVIDIA GPU;
# no cluster or job scheduler is needed.
#
# Usage:
#   bash scripts/run_placement_study.sh --smoke                   # quick check: Qwen_1.5B, MNLI, 5 questions
#   bash scripts/run_placement_study.sh Qwen_1.5B Qwen_7B         # chosen models, all tasks
#   bash scripts/run_placement_study.sh --tasks "mnli arc" Qwen_7B
#   bash scripts/run_placement_study.sh --all                     # the ten models in the paper
#   bash scripts/run_placement_study.sh --help
#
# Options:
#   --smoke            small run to check your setup (Qwen_1.5B unless you name models, MNLI, 5 questions)
#   --all              run the ten models in the paper (needs ~50 GB of GPU memory for the 70B-class ones)
#   --tasks "A B ..."  tasks to run (default: the eight paper tasks that are in the task file)
#   --size N           questions per task (default: 200, the paper setting; 5 with --smoke)
#   --output DIR       results root (default: outputs)
#   --tasks-file F     evaluation sets (default: jsons/tasks200_full.json if present, else jsons/tasks200.json)
#   --skip-check       do not run scripts/check_setup.py first
#   --dry-run          run the checks and print the command, but do not start the run
#
# The same settings can be given as environment variables: TASKS, SIZE, OUTPUT_ROOT,
# TASKS_FILE, PYTHON (the interpreter to use, default python3 or python).
#
# Gated checkpoints (Llama, Mistral, Aya) need a Hugging Face login: accept the terms on each
# model page, then run `huggingface-cli login` once, or export HF_TOKEN. Never write a token
# into this file.
#
# On a SLURM cluster you can submit this script as is, with your own sbatch options, e.g.
#   sbatch --gres=gpu:1 --mem=64G --time=2-00:00:00 scripts/run_placement_study.sh --all

set -euo pipefail

die() { echo "Error: $*" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

# Find the repository root. Under sbatch the script runs from a spool copy, so fall back to the
# submit directory when the script's own directory is not inside the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -f "${REPO_ROOT}/src/runner.py" ]; then
    for d in "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
        if [ -n "$d" ] && [ -f "$d/src/runner.py" ]; then REPO_ROOT="$d"; break; fi
    done
fi
[ -f "${REPO_ROOT}/src/runner.py" ] || die "cannot find src/runner.py. Run this script from the dpp-bias repository."
cd "${REPO_ROOT}"

PAPER_MODELS=(Qwen_1.5B Qwen_7B LLAMA3_3B LLAMA3_8B Cohere_8B Mistral_7B Cohere_32B LLAMA3_70B Mistral_8x7B Qwen_72B)
PAPER_TASKS="ag_news mnli arc mmlu squad gsm8k xsum cnn_dailymail"

SMOKE=0; ALL=0; SKIP_CHECK=0; DRY_RUN=0
MODELS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)    usage; exit 0 ;;
        --smoke)      SMOKE=1 ;;
        --all)        ALL=1 ;;
        --tasks)      [ "$#" -ge 2 ] || die "--tasks needs a value"; TASKS="$2"; shift ;;
        --size)       [ "$#" -ge 2 ] || die "--size needs a value"; SIZE="$2"; shift ;;
        --output)     [ "$#" -ge 2 ] || die "--output needs a value"; OUTPUT_ROOT="$2"; shift ;;
        --tasks-file) [ "$#" -ge 2 ] || die "--tasks-file needs a value"; TASKS_FILE="$2"; shift ;;
        --skip-check) SKIP_CHECK=1 ;;
        --dry-run)    DRY_RUN=1 ;;
        -*)           die "unknown option '$1'. See --help." ;;
        *)            MODELS+=("$1") ;;
    esac
    shift
done

# Python interpreter.
if [ -z "${PYTHON:-}" ]; then
    if command -v python >/dev/null 2>&1; then PYTHON=python
    elif command -v python3 >/dev/null 2>&1; then PYTHON=python3
    else die "no python found. Activate the environment from the README (conda activate dpp-bias) or set PYTHON."; fi
fi

# Models.
if [ "${#MODELS[@]}" -eq 0 ]; then
    if [ "$ALL" -eq 1 ]; then
        MODELS=("${PAPER_MODELS[@]}")
    elif [ "$SMOKE" -eq 1 ]; then
        MODELS=(Qwen_1.5B)
    else
        die "name at least one model (e.g. Qwen_1.5B), or pass --smoke for a quick check or --all for the paper's ten models. See --help."
    fi
fi

# Task file.
if [ -z "${TASKS_FILE:-}" ]; then
    if [ -f jsons/tasks200_full.json ]; then TASKS_FILE=jsons/tasks200_full.json; else TASKS_FILE=jsons/tasks200.json; fi
fi
[ -f "${TASKS_FILE}" ] || die "task file '${TASKS_FILE}' not found."

# Tasks. By default, run the paper tasks that are in the task file, and say which are skipped.
if [ "$SMOKE" -eq 1 ]; then
    TASKS="${TASKS:-mnli}"
    SIZE="${SIZE:-5}"
fi
if [ -z "${TASKS:-}" ]; then
    AVAILABLE="$("$PYTHON" -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))))' "${TASKS_FILE}")" \
        || die "could not read ${TASKS_FILE}."
    TASKS=""; SKIPPED=""
    for t in $PAPER_TASKS; do
        case " $AVAILABLE " in
            *" $t "*) TASKS="$TASKS $t" ;;
            *)        SKIPPED="$SKIPPED $t" ;;
        esac
    done
    TASKS="${TASKS# }"
    if [ -n "$SKIPPED" ]; then
        echo "Note: skipping${SKIPPED}: not in ${TASKS_FILE}." >&2
        echo "      News text is not shipped; run 'python scripts/rebuild_news_sets.py' once to include them." >&2
    fi
fi
[ -n "$TASKS" ] || die "no tasks to run."
read -r -a TASK_LIST <<< "$TASKS"

SIZE="${SIZE:-200}"
case "$SIZE" in ''|*[!0-9]*) die "--size must be a positive whole number, got '$SIZE'." ;; esac
[ "$SIZE" -gt 0 ] || die "--size must be at least 1."
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
RUN_NAME="$([ "$SMOKE" -eq 1 ] && echo smoke || echo results)_${SIZE}_$(date "+%Y%m%d_%H%M%S")"
OUTPUT_PATH="${OUTPUT_ROOT}/${RUN_NAME}"

# runner.py imports the bundled library as `src.llamafactory`, so the repository root must be importable.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Models: ${MODELS[*]}"
echo "Tasks:  ${TASK_LIST[*]}"
echo "Size:   ${SIZE} questions per task, 4 positions"
echo "Data:   ${TASKS_FILE}"
echo "Output: ${OUTPUT_PATH}"
echo

if [ "$SKIP_CHECK" -eq 0 ]; then
    echo "Checking setup (scripts/check_setup.py; skip with --skip-check)..."
    "$PYTHON" scripts/check_setup.py --models "${MODELS[@]}" --tasks "${TASK_LIST[@]}" --tasks_file "${TASKS_FILE}" \
        || die "setup check failed; see the messages above."
    echo
fi

CMD=("$PYTHON" src/runner.py
    --models "${MODELS[@]}"
    --prompts samppm
    --tasks "${TASK_LIST[@]}"
    --prompt_structure icld_ssp icld_esp icld_sum icld_eum
    --prompt_file jsons/prompts.json
    --model_template_file jsons/model_templates.json
    --gen_config_file jsons/gen_configs.json
    --input_path "${TASKS_FILE}"
    --size "${SIZE}"
    --output_path "${OUTPUT_PATH}")

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run; would execute:"
    printf '  %q' "${CMD[@]}"; echo
    exit 0
fi

mkdir -p "${OUTPUT_PATH}"
set +e
"${CMD[@]}" 2>&1 | tee "${OUTPUT_PATH}/run.log"
status="${PIPESTATUS[0]}"
set -e
[ "$status" -eq 0 ] || die "runner exited with status ${status}; the log is ${OUTPUT_PATH}/run.log"

echo
echo "Done. Results are in ${OUTPUT_PATH}/<MODEL>/ (log: ${OUTPUT_PATH}/run.log)."
echo "Summarize with: $PYTHON src/analysis.py --results_dir ${OUTPUT_PATH} --output_path ${OUTPUT_PATH}_analysis --recursive"
