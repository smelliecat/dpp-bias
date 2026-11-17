import json
import random
import argparse
from datasets import load_dataset
import os

def build_mnli_task(num_demonstrations, num_questions, example_set='validation_matched'):
    """
    Build a multiple-choice task using MNLI.
    Each example uses:
      - "question": the premise,
      - "options": a list of labels,
      - "answer": the label,
      - "subject": the genre.
    """
    options = ["neutral", "entailment", "contradiction"]
    # Convert datasets to lists for sampling
    dataset_demos = list(load_dataset("nyu-mll/multi_nli", split="train"))
    dataset_questions = list(load_dataset("nyu-mll/multi_nli", split=example_set))
    
    selected_demos = []
    selected_questions = []
    
    for sample in random.sample(dataset_demos, min(num_demonstrations, len(dataset_demos))):
        selected_demos.append({
            "question": sample["premise"],
            "options": options,
            "answer": options[sample["label"]],
            "answer_index": sample["label"],
            "subject": sample["genre"]
        })
    for sample in random.sample(dataset_questions, min(num_questions, len(dataset_questions))):
        selected_questions.append({
            "question": sample["premise"],
            "options": options,
            "answer": options[sample["label"]],
            "answer_index": sample["label"],
            "subject": sample["genre"]
        })
    return {
        "task_type": "cls",
        "classes": options,
        "demonstrations": selected_demos,
        "questions": selected_questions
    }

def build_mmlu_task(num_demonstrations, num_questions):
    """
    Build a multiple-choice task using MMLU.
    Each example uses:
      - "question", "options", "answer", "subject"
    """
    dataset_demos = list(load_dataset("cais/mmlu", "all", split="auxiliary_train"))
    dataset_questions = list(load_dataset("cais/mmlu", "all", split="validation"))
    
    selected_demos = []
    selected_questions = []
    
    for sample in random.sample(dataset_demos, min(num_demonstrations, len(dataset_demos))):
        selected_demos.append({
            "question": sample["question"],
            "options": sample["choices"],
            "answer_index": sample["answer"],
            "answer": sample["choices"][sample["answer"]],
            "subject": sample["subject"]
        })
    for sample in random.sample(dataset_questions, min(num_questions, len(dataset_questions))):
        selected_questions.append({
            "question": sample["question"],
            "options": sample["choices"],
            "answer_index": sample["answer"],
            "answer": sample["choices"][sample["answer"]],
            "subject": sample["subject"]
        })
    return {
        "task_type": "res",
        "classes": [],
        "demonstrations": selected_demos,
        "questions": selected_questions
    }

def build_arc_task(num_demonstrations, num_questions):
    """
    Build a multiple-choice task using AAC.
    Each example uses:
      - "question", "options", "answer", "subject"
    """
    dataset_demos = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train"))
    dataset_questions = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"))
    
    selected_demos = []
    selected_questions = []
    
    for sample in random.sample(dataset_demos, min(num_demonstrations, len(dataset_demos))):
        selected_demos.append({
            "question": sample["question"],
            "options": sample["choices"]["text"],
            "options_letters": sample["choices"]["label"],
            "answer_index": ord(sample["answerKey"].upper()) - ord('A'),
            "answer": [sample["answerKey"]],
            "subject": sample["id"]
        })
    for sample in random.sample(dataset_questions, min(num_questions, len(dataset_questions))):
        selected_questions.append({
            "question": sample["question"],
            "options": sample["choices"]["text"],
            "options_letters": sample["choices"]["label"],
            "answer_index": ord(sample["answerKey"].upper()) - ord('A'),
            "answer": [sample["answerKey"]],
            "subject": sample["id"]
        })
    return {
        "task_type": "cls",
        "classes": [],
        "demonstrations": selected_demos,
        "questions": selected_questions
    }

def build_xsum_task(num_demonstrations, num_questions):
    """
    Build a summarization task using XSum.
    Each example uses:
      - "prompt", "context", "answer"
    """
    dataset = list(load_dataset("EdinburghNLP/xsum", split="train", trust_remote_code=True))
    question_dataset = list(load_dataset("EdinburghNLP/xsum", split="validation", trust_remote_code=True))
    
    demos = []
    questions = []
    
    for sample in random.sample(dataset, min(num_demonstrations, len(dataset))):
        demos.append({
            "prompt": "Write a concise summary of the following article.",
            "context": sample["document"],
            "answer": sample["summary"]
        })
    for sample in random.sample(question_dataset, min(num_questions, len(question_dataset))):
        questions.append({
            "prompt": "Write a concise summary of the following article.",
            "context": sample["document"],
            "answer": sample["summary"]
        })
    return {
        "task_type": "gen_summ",
        "demonstrations": demos,
        "questions": questions
    }

def build_sst_task(num_demonstrations, num_questions):
    """
    Build a multiple-choice task for SST2.
    Each example uses:
      - "question": the sentence,
      - "options": ["positive", "negative"],
      - "answer": the label,
      - "subject": "sst2"
    """
    dataset = list(load_dataset("stanfordnlp/sst2", split="train"))
    dataset_questions = list(load_dataset("stanfordnlp/sst2", split="validation"))
    
    demos = []
    questions = []
    options = ["positive", "negative"]
    for sample in random.sample(dataset, min(num_demonstrations, len(dataset))):
        demos.append({
            "question": sample["sentence"],
            "options": ["positive", "negative"],
            "answer": options[sample["label"]],
            "answer_index": sample["label"],
            "subject": "sst2"
        })
    for sample in random.sample(dataset_questions, min(num_questions, len(dataset_questions))):
        questions.append({
            "question": sample["sentence"],
            "options": ["positive", "negative"],
            "answer": options[sample["label"]],
            "answer_index": sample["label"],
            "subject": "sst2"
        })
    return {
        "task_type": "cls",
        "classes": ["positive", "negative"],
        "demonstrations": demos,
        "questions": questions
    }

def build_agnews_task(num_demonstrations, num_questions):
    """
    Build a multiple-choice task for AG News.
    Each example uses:
      - "question": the news text,
      - "options": ["World", "Sports", "Business", "Sci/Tech"],
      - "answer": the label,
      - "subject": "ag_news"
    """
    dataset = list(load_dataset("wangrongsheng/ag_news", split="train"))
    question_dataset = list(load_dataset("wangrongsheng/ag_news", split="test"))
    
    demos = []
    questions = []
    options = ["World", "Sports", "Business", "Sci/Tech"]
    for sample in random.sample(dataset, min(num_demonstrations, len(dataset))):
        demos.append({
            "question": sample["text"],
            "options": ["World", "Sports", "Business", "Sci/Tech"],
            "answer_index": sample["label"],
            "answer": options[sample["label"]],
            "subject": "ag_news"
        })
    for sample in random.sample(question_dataset, min(num_questions, len(question_dataset))):
        questions.append({
            "question": sample["text"],
            "options": ["World", "Sports", "Business", "Sci/Tech"],
            "answer_index": sample["label"],
            "answer": options[sample["label"]],
            "subject": "ag_news"
        })
    return {
        "task_type": "cls",
        "classes": ["World", "Sports", "Business", "Sci/Tech"],
        "demonstrations": demos,
        "questions": questions
    }

def build_cnn_dailymail_task(num_demonstrations, num_questions):
    """
    Build a summarization task using CNN/DailyMail.
    Each example uses:
      - "prompt", "context", "answer"
    """
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    train_data = list(dataset["train"])
    val_data = list(dataset["validation"])
    
    demos = []
    questions = []
    for sample in random.sample(train_data, min(num_demonstrations, len(train_data))):
        demos.append({
            "prompt": "Write a concise summary of the following article.",
            "context": sample["article"],
            "answer": sample["highlights"]
        })
    for sample in random.sample(val_data, min(num_questions, len(val_data))):
        questions.append({
            "prompt": "Write a concise summary of the following article.",
            "context": sample["article"],
            "answer": sample["highlights"]
        })
    return {
        "task_type": "gen_summ",
        "demonstrations": demos,
        "questions": questions
    }

def build_squad_task(num_demonstrations, num_questions):
    """
    Build a QA task using SQuAD.
    Each example uses:
      - "context", "question", "answer"
    """
    dataset = load_dataset("rajpurkar/squad")
    train_data = list(dataset["train"])
    val_data = list(dataset["validation"])
    
    demos = []
    questions = []
    for sample in random.sample(train_data, min(num_demonstrations, len(train_data))):
        answer = sample["answers"]["text"][0] if sample["answers"]["text"] else ""
        demos.append({
            "context": sample["context"],
            "question": sample["question"],
            "answer": answer
        })
    for sample in random.sample(val_data, min(num_questions, len(val_data))):
        answer = sample["answers"]["text"][0] if sample["answers"]["text"] else ""
        questions.append({
            "context": sample["context"],
            "question": sample["question"],
            "answer": answer
        })
    return {
        "task_type": "qac",
        "demonstrations": demos,
        "questions": questions
    }

def build_gsm8k_task(num_demonstrations, num_questions):
    """
    Build a math problem-solving task using GSM8K.
    Each example uses:
      - "question", "answer"
    """
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    question_dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    demos = []
    questions = []
    for sample in random.sample(list(dataset), min(num_demonstrations, len(dataset))):
        demos.append({
            "question": sample["question"],
            "answer": sample["answer"]
        })

    for sample in random.sample(list(question_dataset), min(num_questions, len(question_dataset))):
        questions.append({
            "question": sample["question"],
            "answer": sample["answer"]
        })

    return {
        "task_type": "res",
        "demonstrations": demos,
        "questions": questions
    }


# Order matches the task order in jsons/tasks200.json.
TASK_BUILDERS = {
    "mmlu": build_mmlu_task,
    "cnn_dailymail": build_cnn_dailymail_task,
    "squad": build_squad_task,
    "ag_news": build_agnews_task,
    "sst2": build_sst_task,
    "mnli": build_mnli_task,
    "xsum": build_xsum_task,
    "arc": build_arc_task,
    "gsm8k": build_gsm8k_task,
}

def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation data file using Hugging Face datasets."
    )
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file path.")
    parser.add_argument("--num_demos", type=int, default=5, help="Number of demonstrations per task.")
    parser.add_argument("--num_questions", type=int, default=1, help="Number of questions per task.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_BUILDERS),
        choices=list(TASK_BUILDERS),
        help="Tasks to build (default: all)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for Python's random module. Without it every call draws a different sample."
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    data = {}
    # Build tasks; each task is keyed by a short name.
    for task in args.tasks:
        data[task] = TASK_BUILDERS[task](args.num_demos, args.num_questions)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # Save the generated data to a JSON file.
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Data file saved to {args.output_file}")

if __name__ == "__main__":
    main()
