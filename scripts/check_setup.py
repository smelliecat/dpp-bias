#!/usr/bin/env python
"""Check that this machine can run the placement study, before anything is downloaded.

    python scripts/check_setup.py                         # the smoke-test model and task
    python scripts/check_setup.py --models Qwen_7B LLAMA3_8B --tasks mnli arc

It checks, in order:
  1. the configs know every model key and every model/task pair you asked for;
  2. the task file contains the tasks (and says how to add the three news tasks);
  3. the Python packages the runner needs are importable;
  4. PyTorch can see a CUDA GPU, and roughly how much memory each model needs;
  5. your Hugging Face login can reach each checkpoint (several are gated).

It exits with status 1 on the first class of problem that would stop a run, and prints
what to do about it. Warnings (for example a GPU that is probably too small) do not stop it.
Set SKIP_HF_CHECK=1 to skip step 5, for example when the checkpoints are already cached
and you are offline.
"""
import argparse
import importlib
import importlib.util
import json
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_TASKS = {"ag_news", "xsum", "cnn_dailymail"}

# Rough figures for the configured setup (4-bit bitsandbytes, up to ~1.5k tokens of context).
# vram_gb: GPU memory to load the model and generate; disk_gb: the full-precision download in
# the Hugging Face cache (models are quantized when loaded, so the full weights are fetched).
# Treat them as guides, not guarantees.
MODEL_NEEDS = {
    "Qwen_1.5B":    {"vram_gb": 4,  "disk_gb": 4,   "gated": False},
    "LLAMA3_1B":    {"vram_gb": 4,  "disk_gb": 3,   "gated": True},
    "LLAMA3_3B":    {"vram_gb": 6,  "disk_gb": 7,   "gated": True},
    "Qwen_7B":      {"vram_gb": 10, "disk_gb": 16,  "gated": False},
    "Mistral_7B":   {"vram_gb": 10, "disk_gb": 15,  "gated": True},
    "LLAMA3_8B":    {"vram_gb": 10, "disk_gb": 16,  "gated": True},
    "LLAMA2_7B":    {"vram_gb": 10, "disk_gb": 14,  "gated": True},
    "Cohere_8B":    {"vram_gb": 10, "disk_gb": 17,  "gated": True},
    "Cohere_32B":   {"vram_gb": 26, "disk_gb": 65,  "gated": True},
    "Mistral_8x7B": {"vram_gb": 32, "disk_gb": 95,  "gated": True},
    "LLAMA3_70B":   {"vram_gb": 48, "disk_gb": 142, "gated": True},
    "LLAMA3_70B_":  {"vram_gb": 48, "disk_gb": 142, "gated": True},
    "Nemotron_70B": {"vram_gb": 48, "disk_gb": 142, "gated": False},
    "Qwen_72B":     {"vram_gb": 50, "disk_gb": 146, "gated": False},
}

REQUIRED_PACKAGES = [
    ("torch", "PyTorch: install the build for your CUDA version from https://pytorch.org/get-started/locally/"),
    ("transformers", "pip install -r requirements.txt"),
    ("accelerate", "pip install -r requirements.txt"),
    ("peft", "pip install -r requirements.txt"),
    ("trl", "pip install -r requirements.txt"),
    ("datasets", "pip install -r requirements.txt"),
    ("bitsandbytes", "pip install -r requirements.txt (needed for the 4-bit loading in jsons/model_templates.json)"),
    ("evaluate", "pip install -r requirements.txt"),
    ("rouge_score", "pip install -r requirements.txt"),
    ("bert_score", "pip install -r requirements.txt"),
    ("json_repair", "pip install -r requirements.txt"),
    ("sklearn", "pip install -r requirements.txt"),
    ("huggingface_hub", "pip install -r requirements.txt"),
]


class Report:
    def __init__(self):
        self.errors = 0

    def ok(self, msg):
        print(f"  ok    {msg}")

    def warn(self, msg):
        print(f"  WARN  {msg}")

    def fail(self, msg):
        self.errors += 1
        print(f"  FAIL  {msg}")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def check_configs(r, models, tasks, tasks_file):
    print("Configs")
    templates = load_json(os.path.join(REPO_ROOT, "jsons/model_templates.json"))
    gen = load_json(os.path.join(REPO_ROOT, "jsons/gen_configs.json"))
    for m in models:
        if m not in templates:
            r.fail(f"unknown model key '{m}'. Known keys: {', '.join(sorted(templates))}")
            continue
        missing = [t for t in tasks if f"{m}_{t}" not in gen]
        if missing:
            r.fail(f"jsons/gen_configs.json has no entry for {', '.join(m + '_' + t for t in missing)}; "
                   f"add one (copy an existing {m}_* entry) or the runner will stop")
        else:
            r.ok(f"{m} -> {templates[m]['model_name_or_path']}")

    print("Task file")
    path = tasks_file if os.path.isabs(tasks_file) else os.path.join(REPO_ROOT, tasks_file)
    if not os.path.isfile(path):
        r.fail(f"{tasks_file} not found")
        return templates
    data = load_json(path)
    for t in tasks:
        if t in data:
            r.ok(f"{t}: {len(data[t].get('demonstrations', []))} demonstrations, "
                 f"{len(data[t].get('questions', []))} questions")
        elif t in NEWS_TASKS:
            r.fail(f"{t} is not in {tasks_file}. News text is not shipped: run "
                   f"'python scripts/rebuild_news_sets.py' to create jsons/tasks200_full.json, "
                   f"or leave {t} out")
        else:
            r.fail(f"{t} is not in {tasks_file}. Tasks in it: {', '.join(sorted(data))}")
    return templates


def check_packages(r):
    print("Python packages")
    if sys.version_info < (3, 9):
        r.fail(f"Python {sys.version.split()[0]} is too old; use 3.10 to 3.12")
    else:
        r.ok(f"Python {sys.version.split()[0]}")
    missing = []
    for mod, hint in REQUIRED_PACKAGES:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as e:
            if e.name != mod.split(".")[0]:  # the package is there but one of its dependencies is not
                r.fail(f"{mod} is installed but cannot import {e.name}. Fix: {hint}")
            missing.append(mod)
        except Exception as e:  # a broken binary, such as a bitsandbytes built for another CUDA
            missing.append(mod)
            first = str(e).splitlines()[0] if str(e) else ""
            r.fail(f"{mod} is installed but fails to import ({type(e).__name__}: {first})")
    not_installed = [m for m in missing if importlib.util.find_spec(m) is None]
    if "torch" in not_installed:
        r.fail("PyTorch is not installed. Install the build for your CUDA version from "
               "https://pytorch.org/get-started/locally/ before the requirements")
    others = [m for m in not_installed if m != "torch"]
    if others:
        r.fail(f"not installed: {', '.join(others)}. Fix: pip install -r requirements.txt")
    if not missing:
        r.ok("all required packages import")
    try:
        importlib.import_module("flash_attn")
        r.ok("flash-attn installed (optional)")
    except Exception:
        print("  info  flash-attn not installed; LLaMA Factory falls back to the default attention. "
              "This is fine, just slower.")
    return "torch" not in missing


def check_gpu(r, models):
    print("GPU")
    import torch

    if not torch.cuda.is_available():
        r.fail("PyTorch cannot see a CUDA GPU (torch.cuda.is_available() is False). The model configs "
               "load checkpoints in 4-bit with bitsandbytes, which needs an NVIDIA GPU. Check that "
               "nvidia-smi works and that you installed a CUDA build of PyTorch "
               f"(this one is {torch.__version__}, CUDA {torch.version.cuda}).")
        return
    n = torch.cuda.device_count()
    total = 0.0
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        gb = p.total_memory / 1024 ** 3
        total += gb
        r.ok(f"cuda:{i} {p.name}, {gb:.0f} GB")
    for m in models:
        need = MODEL_NEEDS.get(m)
        if need and need["vram_gb"] > total:
            r.warn(f"{m} needs roughly {need['vram_gb']} GB of GPU memory and this machine has "
                   f"{total:.0f} GB in total; it will probably run out of memory")


def check_disk(r, models):
    need = sum(MODEL_NEEDS.get(m, {}).get("disk_gb", 0) for m in models)
    if not need:
        return
    print("Disk")
    cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    probe = cache
    while not os.path.exists(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    free = shutil.disk_usage(probe).free / 1024 ** 3
    msg = (f"these models download about {need} GB into {cache} ({free:.0f} GB free there; "
           f"set HF_HOME to use another disk). Already-cached models do not count again.")
    if free < need:
        r.warn(msg)
    else:
        r.ok(msg)


def check_hf_access(r, models, templates):
    print("Hugging Face access")
    if os.environ.get("SKIP_HF_CHECK") or os.environ.get("HF_HUB_OFFLINE") == "1":
        print("  info  skipped (SKIP_HF_CHECK or HF_HUB_OFFLINE is set)")
        return
    try:
        from huggingface_hub import HfApi, get_token
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except Exception:
        r.warn("huggingface_hub is not importable, so access to gated checkpoints was not checked")
        return
    token = get_token()
    if token:
        r.ok("logged in to the Hugging Face Hub")
    else:
        gated = [m for m in models if MODEL_NEEDS.get(m, {}).get("gated")]
        if gated:
            r.fail(f"not logged in, and these models are gated: {', '.join(gated)}. Accept each model's terms on its "
                   "Hub page, then run 'huggingface-cli login' (or 'hf auth login') or export HF_TOKEN")
            return
        print("  info  not logged in; fine for the ungated models you selected")
    api = HfApi()
    for m in models:
        if m not in templates:
            continue
        repo = templates[m]["model_name_or_path"]
        try:
            if hasattr(api, "auth_check"):
                api.auth_check(repo, token=token)
            else:
                api.model_info(repo, token=token)
            r.ok(f"can access {repo}")
        except GatedRepoError:
            r.fail(f"{repo} is gated and this account has not been granted access yet. "
                   f"Request it at https://huggingface.co/{repo} and wait for approval")
        except RepositoryNotFoundError:
            r.fail(f"{repo} was not found, or the token cannot see it")
        except Exception as e:
            r.warn(f"could not check {repo} ({type(e).__name__}); are you offline? "
                   "Set SKIP_HF_CHECK=1 to skip this step")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["Qwen_1.5B"])
    ap.add_argument("--tasks", nargs="+", default=["mnli"])
    ap.add_argument("--tasks_file", default="jsons/tasks200.json")
    ap.add_argument("--configs_only", action="store_true",
                    help="only check the configs and the task file (no imports, GPU or network)")
    a = ap.parse_args()

    r = Report()
    templates = check_configs(r, a.models, a.tasks, a.tasks_file)
    if not a.configs_only:
        if check_packages(r):
            check_gpu(r, a.models)
        check_disk(r, a.models)
        check_hf_access(r, a.models, templates)

    print()
    if r.errors:
        print(f"{r.errors} problem(s) found. Fix them and run this check again.")
        sys.exit(1)
    print("Setup looks good.")


if __name__ == "__main__":
    main()
