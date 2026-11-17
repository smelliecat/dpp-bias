# dpp-bias

Code and evaluation sets for the paper

> **Where to show Demos in Your Prompt: A Positional Bias of In-Context Learning**
> Kwesi Adu Cobbina and Tianyi Zhou. EMNLP 2025.
> [Paper](https://aclanthology.org/2025.emnlp-main.1503/) · [DOI](https://doi.org/10.18653/v1/2025.emnlp-main.1503)

In-context learning is usually studied by changing *which* demonstrations go into a prompt or
the order among them. This study changes only *where* the block of demonstrations sits. The
instruction, the demonstrations and the question stay the same; the block moves between four
places in a chat prompt. That move alone shifts accuracy and flips individual predictions, an
effect the paper calls **DPP bias** (demos' position in prompt).

This repository holds the pipeline used for the study: it builds the four prompt variants, runs
them through open-weight chat models, extracts an answer from each response, and scores every
model, task and position.

## The four positions

| Flag (`--prompt_structure`) | Name in the paper | Where the demonstration block goes |
|---|---|---|
| `icld_ssp` | ssp, start of system prompt | system prompt, before the instruction |
| `icld_esp` | esp, end of system prompt | system prompt, after the instruction |
| `icld_sum` | sum, start of user message | user message, before the question |
| `icld_eum` | eum, end of user message | user message, after the question |

The prompt assembly is in `get_formatted_prompt` and `process_questions` in `src/runner.py`.

## Repository layout

```
src/runner.py               builds prompts, queries the model, extracts answers, scores each position
src/evaluation_metrics.py   accuracy, exact match / F1, ROUGE, BERTScore and text statistics
src/analysis.py             aggregates saved results into one error-analysis summary
src/plotter.py              exploratory plots of the per-position scores
src/data_maker.py           draws demonstrations and questions from the source datasets
src/llamafactory/           pinned, unmodified copy of LLaMA Factory (model loading and chat templates)
jsons/prompts.json          instructions (the study uses the shared one, `--prompts samppm`)
jsons/model_templates.json  checkpoint, quantization and chat template for each model key
jsons/gen_configs.json      decoding settings for each model and task
jsons/tasks200.json         evaluation sets: 5 demonstrations and 200 questions per task
jsons/news_sets.sha256.json fingerprints of the AG News, CNN/DailyMail and XSum sets (see "Data")
scripts/run_placement_study.sh   the full sweep
scripts/rebuild_news_sets.py     rebuilds the three news sets from the public datasets
```

## Setup

```bash
git clone https://github.com/smelliecat/dpp-bias.git
cd dpp-bias
conda create -n dpp-bias python=3.12 -y && conda activate dpp-bias
# Install the PyTorch build for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pip install flash-attn --no-build-isolation   # optional; the model templates ask for FlashAttention-2
```

Some checkpoints are gated on the Hugging Face Hub. Accept their terms on the model pages, then
run `huggingface-cli login` or export `HF_TOKEN` in your shell.

`runner.py` imports the bundled library as `src.llamafactory`, so run it from the repository root
with the root on `PYTHONPATH` (the sweep script does this for you).

## Quick check

One small model, one task, five questions, all four positions:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
python src/runner.py \
    --models Qwen_1.5B \
    --tasks mnli \
    --prompts samppm \
    --prompt_structure icld_ssp icld_esp icld_sum icld_eum \
    --prompt_file jsons/prompts.json \
    --model_template_file jsons/model_templates.json \
    --gen_config_file jsons/gen_configs.json \
    --input_path jsons/tasks200.json \
    --size 5 \
    --output_path outputs/quick_check
```

## Running the sweep

```bash
python scripts/rebuild_news_sets.py         # once, to add AG News, XSum and CNN/DailyMail (see "Data")
bash scripts/run_placement_study.sh         # ten models x eight tasks x four positions
bash scripts/run_placement_study.sh Qwen_7B # or any subset of model keys
```

Models (keys from `jsons/model_templates.json`):

| Key | Checkpoint |
|---|---|
| `Qwen_1.5B` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `Qwen_7B` | `Qwen/Qwen2.5-7B-Instruct` |
| `Qwen_72B` | `Qwen/Qwen2.5-72B-Instruct` |
| `LLAMA3_3B` | `meta-llama/Llama-3.2-3B-Instruct` |
| `LLAMA3_8B` | `meta-llama/Llama-3.1-8B-Instruct` |
| `LLAMA3_70B` | `meta-llama/Llama-3.3-70B-Instruct` |
| `Mistral_7B` | `mistralai/Mistral-7B-Instruct-v0.1` |
| `Mistral_8x7B` | `mistralai/Mixtral-8x7B-Instruct-v0.1` |
| `Cohere_8B` | `CohereForAI/aya-expanse-8b` |
| `Cohere_32B` | `CohereForAI/aya-expanse-32b` |

As configured in `jsons/model_templates.json`, each checkpoint is loaded through LLaMA Factory's
`ChatModel` with 4-bit bitsandbytes quantization. Decoding is greedy (temperature 0), with up to
50 new tokens for AG News, MNLI, ARC and MMLU and up to 500 for SQuAD, GSM8K, XSum and
CNN/DailyMail (`jsons/gen_configs.json`).

Tasks:

| Flag (`--tasks`) | Source dataset on the Hub | Score |
|---|---|---|
| `ag_news` | `wangrongsheng/ag_news` | accuracy |
| `mnli` | `nyu-mll/multi_nli` | accuracy |
| `arc` | `allenai/ai2_arc` (ARC-Challenge) | accuracy |
| `mmlu` | `cais/mmlu` | accuracy |
| `gsm8k` | `openai/gsm8k` | accuracy |
| `squad` | `rajpurkar/squad` | exact match, F1 |
| `xsum` | `EdinburghNLP/xsum` | ROUGE, BERTScore |
| `cnn_dailymail` | `abisee/cnn_dailymail` (3.0.0) | ROUGE, BERTScore |

`jsons/tasks200.json` also contains an SST-2 set (`sst2`), which the runner supports but which is
not one of the eight tasks reported in the paper.

## Outputs

```
<output_path>/<MODEL>/<MODEL>_<task>_full_samppm.json   one file per task
<output_path>/<MODEL>/<MODEL>_all_results.json          every task for that model
```

A task file has one entry per position (`icld_ssp`, `icld_esp`, `icld_sum`, `icld_eum`), each with
the `demonstrations` used, the `questions`, and `metric_scores`. Every question keeps its original
fields and gains `model_answer` (the raw response), `predicted_answer`, and `cleaned_answer` (the
label or text extracted from the response; this is what gets scored, with the raw response as the
fallback when nothing can be extracted).

To summarize or plot a finished run:

```bash
python src/analysis.py --results_dir outputs/<run> --output_path outputs/<run>_analysis --recursive
python src/plotter.py --results_dir outputs/<run>      # writes to ./result_plots/
```

## Scope of this release

The release covers prompt construction, generation, answer extraction and per-position scoring
for the four positions. The paper's two summary metrics are derived from these outputs:
accuracy-change compares each position with a zero-shot run of the same model and task, and
prediction-change counts the answers that flip relative to the `sum` position. The zero-shot
runs and the scripts that turn the saved outputs into the paper's tables and figures are not
included here.

Other things to know:

- `runner.py` does not fix a random seed. With `--size 200` it uses every question in the task
  file, so the only thing left to chance is the order in which questions are processed.
- `--attention_analysis` and `--visualize_attention` call `ChatModel.chat_with_attentions`, which
  upstream LLaMA Factory does not provide. They are not used in the sweep and will not run with
  the bundled copy.
- `--ablation` reruns each task with one demonstration removed at a time; `--randomize_demos`
  shuffles the order of the demonstrations.

## Data

`jsons/tasks200.json` is the reference copy of the evaluation sets for MMLU, SQuAD, SST-2, MNLI,
ARC and GSM8K: five demonstrations and 200 questions per task. They are shipped, rather than
regenerated, so that the exact questions are fixed. They are small samples of public datasets and
remain under the terms of those datasets; see the dataset pages for the Hub ids above (SST-2 is
`stanfordnlp/sst2`).

News text is not redistributed here. For AG News, XSum and CNN/DailyMail,
`jsons/news_sets.sha256.json` keeps each demonstration and question with its news text replaced
by a SHA-256 fingerprint. `scripts/rebuild_news_sets.py` downloads the three datasets, finds the
texts with those fingerprints, checks each rebuilt set against its recorded fingerprint, and
writes `jsons/tasks200_full.json` with all nine tasks. That file is git-ignored; please keep it
out of commits.

`src/data_maker.py` shows how sets like these are drawn. Without `--seed`, every call draws a
different sample. Its builders currently label ARC and GSM8K with different task types from the
shipped file, which changes how the runner words those questions, so use the shipped file to
match the paper.

## Citation

```bibtex
@inproceedings{cobbina-zhou-2025-show,
    title = "Where to show Demos in Your Prompt: A Positional Bias of In-Context Learning",
    author = "Cobbina, Kwesi Adu and Zhou, Tianyi",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1503/",
    doi = "10.18653/v1/2025.emnlp-main.1503",
    pages = "29560--29593",
}
```

## License and acknowledgements

The code is released under the Apache License 2.0 (`LICENSE`, `NOTICE`).

Model loading and chat templating use [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory).
`src/llamafactory/` is an unmodified copy of that project's package at commit
[`0583d06`](https://github.com/hiyouga/LLaMA-Factory/commit/0583d0667648edebde89f6d034b1011e604901ec),
under its own Apache 2.0 license (`src/llamafactory/LICENSE`).
