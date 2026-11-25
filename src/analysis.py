import os
import json
import argparse
import logging
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Optional: If you want classification metrics like precision, recall, F1
# from sklearn, you can do:
try:
    from sklearn.metrics import (
        accuracy_score, 
        precision_score, 
        recall_score, 
        f1_score,
        confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning("scikit-learn is not installed. Some metrics (precision, recall, F1) will be skipped.")
    SKLEARN_AVAILABLE = False

# Optional: If you want to compute ROUGE for summarization:
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    logging.warning("rouge_score library not installed. ROUGE will be skipped.")
    ROUGE_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description="Perform error analysis on saved LLM outputs with multiple metrics.")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory containing model subfolders with JSON results."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the aggregated error analysis."
    )
    parser.add_argument(
        "--recursive",
        action='store_true',
        help="If set, search recursively for JSON files in results_dir."
    )
    return parser.parse_args()


def load_json_files(results_dir: str, recursive: bool=False) -> List[str]:
    """
    Search for .json files in 'results_dir' (and subdirs if recursive=True).
    Returns a list of absolute file paths.
    """
    json_paths = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith(".json"):
                json_paths.append(os.path.join(root, file))
        if not recursive:
            break
    return json_paths


# ------------------------------------------------------------------------
# METRIC HELPERS
# ------------------------------------------------------------------------

def compute_classification_metrics(
    gold_labels: List[str],
    pred_labels: List[str]
) -> Dict[str, float]:
    """
    Compute standard classification metrics (Accuracy, Precision, Recall, F1).
    Uses scikit-learn if available. Otherwise, only accuracy is computed.
    """
    metrics = {}
    # Simple Accuracy
    correct = sum(g == p for g, p in zip(gold_labels, pred_labels))
    total = len(gold_labels)
    accuracy = correct / total if total > 0 else 0.0
    metrics["accuracy"] = accuracy

    if SKLEARN_AVAILABLE:
        # Convert str labels to a consistent set for precision/recall/f1 if possible
        # For multi-class, you can set average="weighted" or "macro"
        metrics["precision"] = precision_score(gold_labels, pred_labels, average="macro", zero_division=0)
        metrics["recall"]    = recall_score(gold_labels, pred_labels, average="macro", zero_division=0)
        metrics["f1"]        = f1_score(gold_labels, pred_labels, average="macro", zero_division=0)
        # You could also compute confusion_matrix if needed:
        # cm = confusion_matrix(gold_labels, pred_labels)
        # metrics["confusion_matrix"] = cm.tolist()  # Convert to list for JSON serialization
    return metrics


def compute_token_overlap_metrics(
    gold_answers: List[str],
    pred_answers: List[str]
) -> Dict[str, float]:
    """
    For short-answer QA (extractive or open-domain), compute a simple
    Exact Match (EM) and token-level F1.
    """
    from collections import Counter
    
    total = len(gold_answers)
    exact_count = 0
    sum_precision = 0.0
    sum_recall = 0.0

    for gold, pred in zip(gold_answers, pred_answers):
        # Check exact match (case-insensitive)
        if gold.strip().lower() == pred.strip().lower():
            exact_count += 1

        # Token-level F1
        gold_toks = gold.lower().split()
        pred_toks = pred.lower().split()
        gold_cnt = Counter(gold_toks)
        pred_cnt = Counter(pred_toks)
        
        # Intersection for overlap
        overlap = sum((gold_cnt & pred_cnt).values())
        precision = overlap / len(pred_toks) if len(pred_toks) > 0 else 0
        recall = overlap / len(gold_toks) if len(gold_toks) > 0 else 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        
        sum_precision += precision
        sum_recall += recall
    
    em_score = exact_count / total if total > 0 else 0.0
    avg_precision = sum_precision / total if total > 0 else 0.0
    avg_recall = sum_recall / total if total > 0 else 0.0
    if avg_precision + avg_recall > 0:
        avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        avg_f1 = 0.0
    
    return {
        "exact_match": em_score,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1
    }


def compute_rouge_scores(
    gold_texts: List[str],
    pred_texts: List[str]
) -> Dict[str, float]:
    """
    For summarization or other generative tasks, compute ROUGE-L (and optionally ROUGE-1, ROUGE-2).
    Requires `rouge_score` library.
    """
    if not ROUGE_AVAILABLE:
        return {"message": "ROUGE not computed (rouge_score not installed)."}

    # Initialize the scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregated_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    total = len(gold_texts)

    for gold, pred in zip(gold_texts, pred_texts):
        scores = scorer.score(gold, pred)
        aggregated_scores["rouge1"] += scores["rouge1"].fmeasure
        aggregated_scores["rouge2"] += scores["rouge2"].fmeasure
        aggregated_scores["rougeL"] += scores["rougeL"].fmeasure

    # Compute average
    if total > 0:
        for k in aggregated_scores:
            aggregated_scores[k] /= total
    return aggregated_scores


def compute_code_metrics(
    gold_code: List[str],
    pred_code: List[str]
) -> Dict[str, float]:
    """
    Placeholder function for code generation metrics.
    Typically you'd run each `pred_code` snippet with a test harness
    and measure pass@k or exact match. We'll do a simple exact match here.
    """
    correct = 0
    total = len(gold_code)
    for g, p in zip(gold_code, pred_code):
        # If you want a looser check, you can parse out function bodies or disregard comments, etc.
        if g.strip() == p.strip():
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"exact_match_accuracy": accuracy}


# ------------------------------------------------------------------------
# MAIN ANALYSIS LOGIC
# ------------------------------------------------------------------------

def analyze_json_file(json_file: str) -> Dict:
    """
    Load a single JSON result file and compute relevant metrics for each sub-task,
    which might be nested in the JSON structure.
    Returns a dictionary of aggregated metric results and 'wrong examples' if relevant.
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    analysis_results = {}

    # The stored structure might have multiple levels: 
    # data[task][ablation_label][structure] = { "demonstrations": ..., "questions": [...] }
    # or simpler. We'll handle nested dicts generically.

    if not isinstance(data, dict):
        logging.warning(f"File {json_file} has unexpected data format (not a dict).")
        return {}

    # Recursive or multi-level approach:
    analysis_results = traverse_and_analyze(data, [])

    return analysis_results


def traverse_and_analyze(
    node, 
    path_keys: List[str]
) -> Dict:
    """
    Recursively walk the nested dictionary structure. 
    Whenever we find a leaf with `questions`, we compute metrics based on 
    the guessed or known task type. Return a dictionary with 
    path-based keys => computed metrics.
    """
    results = {}

    if isinstance(node, dict):
        # If this node has 'questions' and 'demonstrations', let's analyze
        if "questions" in node and "demonstrations" in node:
            # Perform metric analysis
            analysis_key = "-".join(path_keys)  # e.g., "mmlu-abl_1-icld_ssp"
            questions = node["questions"]
            if not questions:
                return results

            # We'll guess the task type by looking at the first item
            sample_q = questions[0]
            task_type = guess_task_type(sample_q)

            # Collect gold answers vs. model answers
            gold_list = []
            pred_list = []
            # For code or summarization tasks, we store the full strings to compare with specialized metrics
            gold_texts = []
            pred_texts = []

            for q in questions:
                gold = str(q.get("answer", "")).strip()
                pred = str(q.get("model_answer", "")).strip()

                gold_list.append(gold)
                pred_list.append(pred)

                gold_texts.append(gold)
                pred_texts.append(pred)

            # Decide which metrics to compute
            metrics = {}

            if task_type in ("cls", "mcq", "nli"):
                # Use classification metrics
                class_metrics = compute_classification_metrics(gold_list, pred_list)
                metrics.update(class_metrics)

            elif task_type in ("qac", "qa"):
                # Use token-overlap metrics (Exact Match, F1)
                qa_metrics = compute_token_overlap_metrics(gold_list, pred_list)
                metrics.update(qa_metrics)

            elif task_type == "gen_summ":
                # Summarization => ROUGE
                rouge_metrics = compute_rouge_scores(gold_texts, pred_texts)
                metrics.update(rouge_metrics)

            elif task_type == "code":
                # Code => exact match or pass@k if you have a harness
                code_metrics = compute_code_metrics(gold_texts, pred_texts)
                metrics.update(code_metrics)

            else:
                # For unknown or purely generative tasks, do a minimal metric or skip
                # You can also add BERTScore here if you want semantic similarity
                metrics["message"] = f"No specialized metrics for task_type '{task_type}'."

            results[analysis_key] = {
                "task_type": task_type,
                **metrics
            }

        else:
            # If not a leaf, keep traversing
            for key, val in node.items():
                sub_path = path_keys + [key]
                sub_results = traverse_and_analyze(val, sub_path)
                # Merge sub-results
                results.update(sub_results)

    elif isinstance(node, list):
        # If it's a list, there's no direct tasks to analyze; skip
        return results

    return results


def guess_task_type(question_entry: dict) -> str:
    """
    Heuristically guess the task type from the question entry.
    You might store 'task_type' explicitly in each question in your data,
    in which case you could just return question_entry['task_type'].
    """
    # If there's 'sentence1' and 'sentence2', it's possibly an NLI or classification
    if "sentence1" in question_entry and "sentence2" in question_entry:
        return "nli"  # or "cls"
    # If there's 'options', it might be multiple-choice
    if "options" in question_entry:
        return "mcq"
    # If there's 'context', might be a QA context
    if "context" in question_entry:
        return "qac"
    # If there's a 'prompt' but no 'options', maybe it's a generative or code
    if "prompt" in question_entry:
        # If you know it's code vs. summary from your data, you can check more fields
        # e.g. if "language" in question_entry or something. We'll guess text gen.
        return "gen_summ"
    # If there's 'code' or special markers, guess code
    # else it's unknown:
    return "unknown"


def main():
    args = parse_args()
    logging.info(f"Loading results from: {args.results_dir}")

    json_files = load_json_files(args.results_dir, recursive=args.recursive)
    logging.info(f"Found {len(json_files)} JSON files to analyze.")

    # We'll aggregate results across all files
    aggregated_results = {}

    for jf in json_files:
        logging.info(f"Analyzing {jf} ...")
        analysis_dict = analyze_json_file(jf)
        aggregated_results[jf] = analysis_dict

    # Save final aggregated results
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, "error_analysis_summary.json")
    with open(output_file, "w") as f:
        json.dump(aggregated_results, f, indent=4)
    logging.info(f"Error analysis results saved to {output_file}")


if __name__ == "__main__":
    main()
