# evaluation_metrics.py

import evaluate
from typing import List, Dict
from bert_score import BERTScorer
import numpy as np
import re
import json
from typing import List, Dict, Any

def is_single_json(output: str) -> bool:
    """
    Checks if the given string is exactly one valid JSON object.
    
    Parameters:
        output (str): The string to evaluate.
    
    Returns:
        bool: True if output is exactly one valid JSON, False otherwise.
    """
    output = output.strip()
    if not output:
        return False  # Empty output is not valid JSON

    decoder = json.JSONDecoder()
    try:
        # Attempt to decode one JSON object from the string
        obj, index = decoder.raw_decode(output)
        # If there is extra non-whitespace content, it's invalid (e.g., multiple JSONs)
        if output[index:].strip():
            return False
        return True
    except json.JSONDecodeError:
        return False

def evaluate_json_outputs(outputs: List[str]) -> Dict[str, Any]:
    """
    Evaluates a list of model outputs for whether they are exactly one valid JSON.
    Returns individual scores and an aggregated final score.

    Parameters:
        outputs (List[str]): A list of output strings to evaluate.

    Returns:
        Dict[str, Any]: A dictionary with keys:
            - "individual_scores": A dict mapping each sample index to its score (1 for valid JSON, 0 otherwise).
            - "final_score": The average score across all samples.
    """
    individual_scores = {}
    for i, output in enumerate(outputs):
        score = 1 if is_single_json(output) else 0
        individual_scores[f'sample_{i}'] = score
    
    final_score = sum(individual_scores.values()) / len(individual_scores) if individual_scores else 0
    
    return {
        "individual_scores": individual_scores,
        "final_score": final_score
    }

def compute_accuracy(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Uses Hugging Face 'evaluate' to compute accuracy for classification or reasoning tasks.
    """
    letter_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "": 9}
    

    def convert(value):
        # Handle None and strings that are empty or "None"
        if value is None:
            return -1
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.lower() == "none":
                return -1
            if stripped.upper() in letter_mapping:
                return letter_mapping[stripped.upper()]
            try:
                return int(stripped)
            except ValueError:
                return -1
        return value


    # Convert both predictions and references.
    converted_predictions = [convert(p) for p in predictions]
    converted_references = [convert(r) for r in references]

    accuracy_metric = evaluate.load("accuracy")
    results = accuracy_metric.compute(
        references=converted_references,
        predictions=converted_predictions
    )
    return {"accuracy": results["accuracy"]}


def compute_rouge_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE and BERTScore for summarization tasks (e.g., CNN/DM, XSum).
    """
    rouge_metric = evaluate.load("rouge")
    bertscore_metric = evaluate.load("bertscore")
    scorer = BERTScorer(
        lang="en",
        model_type="roberta-large",
        num_layers=17,
        batch_size=8,
        rescale_with_baseline=True
    )

    rouge_scores = rouge_metric.compute(
        references=references,
        predictions=predictions,
        use_stemmer=True
    )
    bertscore_scores = bertscore_metric.compute(
        references=references,
        predictions=predictions,
        model_type="bert-base-uncased"
    )



    combined_results = {}
    for k, v in rouge_scores.items():
        combined_results[f"rouge_{k}"] = v
    combined_results["bertscore_precision"] = bertscore_scores["precision"]
    combined_results["bertscore_recall"] = bertscore_scores["recall"]
    combined_results["bertscore_f1"] = bertscore_scores["f1"]
    combined_results["mean_precision"] = float(np.mean(bertscore_scores["precision"]))
    combined_results["mean_recall"] = float(np.mean(bertscore_scores["recall"]))
    combined_results["mean_f1"] = float(np.mean(bertscore_scores["f1"]))
    return combined_results


def compute_exact_match_f1(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute Exact Match (EM) and F1 for QA tasks (e.g., SQuAD, NaturalQuestions).
    """
    squad_metric = evaluate.load("squad")

    # Convert each prediction/reference into SQuAD-compatible format:
    formatted_preds = []
    formatted_refs = []
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        formatted_preds.append({"id": str(i), "prediction_text": pred})
        formatted_refs.append({"id": str(i), "answers": {"text": [ref], "answer_start": [0]}})

    results = squad_metric.compute(
        predictions=formatted_preds,
        references=formatted_refs
    )
    return {"exact_match": results["exact_match"], "f1": results["f1"]}


def compute_pass_at_1(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Placeholder for pass@1 for coding tasks (HumanEval, MBPP).
    """
    pass_count = 0
    for pred, ref in zip(predictions, references):
        # Very naive check for demonstration:
        if pred.strip() == ref.strip():
            pass_count += 1
    pass_at_1 = pass_count / len(predictions) if len(predictions) > 0 else 0.0
    return {"pass@1": pass_at_1}


def compute_length_controlled_win_rate(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Placeholder for a custom 'length-controlled win rate' metric for instruction tasks.
    """
    return {"length_controlled_win_rate": 0.0}


def count_words(text):
    words = re.findall(r'\w+', text)
    return max(1, len(words))

def count_sentences(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    return len(sentences)

def count_syllables(word):
    word = word.lower()
    vowels = 'aeiouy'
    syllables = 0
    if word and word[0] in vowels:
        syllables += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i-1] not in vowels:
            syllables += 1
    if word.endswith('e'):
        syllables = max(1, syllables - 1)
    return syllables or 1

def total_syllables(text):
    words = re.findall(r'\w+', text)
    return sum(count_syllables(word) for word in words)

def count_complex_words(text):
    words = re.findall(r'\w+', text)
    complex_words = [word for word in words if count_syllables(word) >= 3]
    return len(complex_words)

def coleman_liau_index(references: List[str]) -> List[float]:
    results = []
    for text in references:
        letters = len(re.findall(r'[a-zA-Z]', text))
        words = count_words(text)
        sentences = count_sentences(text)
        L = letters / words * 100
        S = sentences / words * 100
        score = 0.0588 * L - 0.296 * S - 15.8
        results.append(round(score, 2))
    return results

def flesch_kincaid_reading_ease(references: List[str]) -> List[float]:
    results = []
    for text in references:
        words = count_words(text)
        sentences = count_sentences(text)
        syllables = total_syllables(text)
        score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        results.append(round(score, 2))
    return results

def gunning_fog_index(references: List[str]) -> List[float]:
    results = []
    for text in references:
        words = count_words(text)
        sentences = count_sentences(text)
        complex_words = count_complex_words(text)
        score = 0.4 * ((words / sentences) + 100 * (complex_words / words))
        results.append(round(score, 2))
    return results

def automated_readability_index(references: List[str]) -> List[float]:
    results = []
    for text in references:
        characters = len(re.findall(r'\w', text))
        words = count_words(text)
        sentences = count_sentences(text)
        score = 4.71 * (characters / words) + 0.5 * (words / sentences) - 21.43
        results.append(round(score, 2))
    return results

def unique_word_count(references: List[str]) -> List[int]:
    results = []
    for text in references:
        words = re.findall(r'\w+', text.lower())
        results.append(len(set(words)))
    return results


def evaluate_task(task_name: str, predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Given a task name, run the appropriate metric(s) on the predictions and references.
    """
    task_name_lower = task_name.lower()

    # Classification or reasoning => accuracy
    if task_name_lower in ["sst2", "ag_news", "mnli", "arc", "mmlu", "gsm8k"]:
        accuracy = compute_accuracy(predictions, references)
        # Add Coleman-Liau index
        coleman_liau = coleman_liau_index(predictions)
        flesch_kincaid = flesch_kincaid_reading_ease(predictions)
        gunning_fog = gunning_fog_index(predictions)
        automated_readability = automated_readability_index(predictions)
        unique_word = unique_word_count(predictions)
        return {
            "accuracy": accuracy["accuracy"],
            "coleman_liau_index": coleman_liau,
            "flesch_kincaid_reading_ease": flesch_kincaid,
            "gunning_fog_index": gunning_fog,
            "automated_readability_index": automated_readability,
            "unique_word_count": unique_word
        }
    
        # return compute_accuracy(predictions, references)

    # Summarization => ROUGE, BERTScore
    elif task_name_lower in ["cnn_dailymail", "xsum"]:
        # Add Coleman-Liau index
        coleman_liau = coleman_liau_index(predictions)
        flesch_kincaid = flesch_kincaid_reading_ease(predictions)
        gunning_fog = gunning_fog_index(predictions)
        automated_readability = automated_readability_index(predictions)
        unique_word = unique_word_count(predictions)
        results = compute_rouge_bertscore(predictions, references)
        results["coleman_liau_index"] = coleman_liau
        results["flesch_kincaid_reading_ease"] = flesch_kincaid
        results["gunning_fog_index"] = gunning_fog
        results["automated_readability_index"] = automated_readability
        results["unique_word_count"] = unique_word
        return results
        # return compute_rouge_bertscore(predictions, references)

    # QA => EM, F1
    elif task_name_lower in ["squad", "naturalquestions"]:
        # Add Coleman-Liau index
        coleman_liau = coleman_liau_index(predictions)
        flesch_kincaid = flesch_kincaid_reading_ease(predictions)
        gunning_fog = gunning_fog_index(predictions)
        automated_readability = automated_readability_index(predictions)
        unique_word = unique_word_count(predictions)
        results = compute_exact_match_f1(predictions, references)
        results["coleman_liau_index"] = coleman_liau
        results["flesch_kincaid_reading_ease"] = flesch_kincaid
        results["gunning_fog_index"] = gunning_fog
        results["automated_readability_index"] = automated_readability
        results["unique_word_count"] = unique_word
        return results
        # return compute_exact_match_f1(predictions, references)

    # Coding => pass@1
    elif task_name_lower in ["humaneval", "mbpp"]:
        # Add Coleman-Liau index
        coleman_liau = coleman_liau_index(predictions)
        flesch_kincaid = flesch_kincaid_reading_ease(predictions)
        gunning_fog = gunning_fog_index(predictions)
        automated_readability = automated_readability_index(predictions)
        unique_word = unique_word_count(predictions)
        results = compute_pass_at_1(predictions, references)
        results["coleman_liau_index"] = coleman_liau
        results["flesch_kincaid_reading_ease"] = flesch_kincaid
        results["gunning_fog_index"] = gunning_fog
        results["automated_readability_index"] = automated_readability
        results["unique_word_count"] = unique_word
        return results
        # return compute_pass_at_1(predictions, references)

    # Instruction => length-controlled win rate
    elif task_name_lower in ["alpaca_eval"]:
        return compute_length_controlled_win_rate(predictions, references)

    else:
        return {"error": f"Task '{task_name}' not recognized."}
