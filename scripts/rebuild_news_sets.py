#!/usr/bin/env python
"""Rebuild the AG News, CNN/DailyMail and XSum evaluation sets from the public datasets.

This repository does not redistribute news text. For these three tasks,
jsons/news_sets.sha256.json keeps every demonstration and question with its
news text replaced by a SHA-256 fingerprint. This script downloads the three
datasets from the Hugging Face Hub, finds the texts with those fingerprints,
and writes one task file that holds all nine tasks (the six shipped in
jsons/tasks200.json plus the three rebuilt here).

    python scripts/rebuild_news_sets.py

The output, jsons/tasks200_full.json, is listed in .gitignore. Please do not
commit it.
"""
import argparse
import hashlib
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = ("demonstrations", "questions")


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha256(obj):
    return sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False))


def iter_rows(spec):
    """Yield every row of the dataset as a tuple of the texts in spec["columns"] order."""
    from datasets import load_dataset

    columns = list(spec["columns"].values())
    for split in spec["splits"]:
        args = [spec["dataset"]] + ([spec["config"]] if spec.get("config") else [])
        dataset = load_dataset(*args, split=split, **spec.get("load_kwargs", {}))
        for row in dataset:
            yield tuple(row[column] for column in columns)


def rebuild_task(name, spec, rows):
    """Fill the fingerprinted fields of spec["task"] from `rows`.

    Returns (task_dict, number_of_entries_filled, problems).
    """
    fields = list(spec["columns"])  # entry fields that hold news text; the first one identifies the row
    template = spec["task"]

    def key_of(entry):
        return tuple(entry[field]["sha256"] for field in fields)

    wanted = {}  # fingerprint of the first field -> full fingerprint tuples still needed
    for part in PARTS:
        for entry in template[part]:
            key = key_of(entry)
            wanted.setdefault(key[0], set()).add(key)

    found = {}  # fingerprint of the first field -> {full fingerprint tuple -> texts}
    open_keys = sum(len(keys) for keys in wanted.values())
    for texts in rows:
        first = sha256(texts[0])
        if first not in wanted:
            continue
        key = (first,) + tuple(sha256(text) for text in texts[1:])
        candidates = found.setdefault(first, {})
        if key not in candidates:
            candidates[key] = texts
            if key in wanted[first]:
                open_keys -= 1
                if open_keys == 0:
                    break

    problems = []
    filled = 0
    task = {}
    for part_or_key, value in template.items():
        if part_or_key not in PARTS:
            task[part_or_key] = value
            continue
        task[part_or_key] = []
        for position, entry in enumerate(value):
            key = key_of(entry)
            candidates = found.get(key[0])
            if not candidates:
                problems.append(f"{name}/{part_or_key}[{position}]: text not found in the dataset")
                continue
            texts = candidates.get(key)
            if texts is None:
                texts = next(iter(candidates.values()))
                problems.append(
                    f"{name}/{part_or_key}[{position}]: text found, but its {fields[1:]} differs "
                    f"from the one used in the study"
                )
            rebuilt = dict(entry)
            for field, text in zip(fields, texts):
                rebuilt[field] = text
            task[part_or_key].append(rebuilt)
            filled += 1

    if not problems and canonical_sha256(task) != spec["task_sha256"]:
        problems.append(f"{name}: rebuilt task does not match the recorded fingerprint")
    return task, filled, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--manifest",
        default=os.path.join(REPO_ROOT, "jsons", "news_sets.sha256.json"),
        help="Fingerprint file for the news tasks."
    )
    parser.add_argument(
        "--base",
        default=os.path.join(REPO_ROOT, "jsons", "tasks200.json"),
        help="Task file with the tasks that ship with the repository."
    )
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "jsons", "tasks200_full.json"),
        help="Where to write the nine-task file."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="News tasks to rebuild (default: every task in the manifest)."
    )
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(args.base, encoding="utf-8") as f:
        tasks = json.load(f)

    names = args.tasks or list(manifest["tasks"])
    unknown = [n for n in names if n not in manifest["tasks"]]
    if unknown:
        parser.error(f"not in the manifest: {', '.join(unknown)}")

    all_problems = []
    for name in names:
        spec = manifest["tasks"][name]
        print(f"[{name}] searching {spec['dataset']} ...", flush=True)
        task, filled, problems = rebuild_task(name, spec, iter_rows(spec))
        total = sum(len(spec["task"][part]) for part in PARTS)
        print(f"[{name}] rebuilt {filled} of {total} entries", flush=True)
        all_problems.extend(problems)
        tasks[name] = task

    order = [t for t in manifest.get("task_order", []) if t in tasks]
    order += [t for t in tasks if t not in order]
    tasks = {t: tasks[t] for t in order}

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)
    print(f"Wrote {args.output} ({', '.join(tasks)})")

    if all_problems:
        print("\nThe rebuilt file differs from the sets used in the study:", file=sys.stderr)
        for problem in all_problems:
            print(f"  - {problem}", file=sys.stderr)
        sys.exit(1)
    print("Every rebuilt set matches the fingerprint of the set used in the study.")


if __name__ == "__main__":
    main()
