import os
import json
import csv
import random
import string
import logging
import argparse
from tqdm import tqdm
import numpy as np
import torch
import math
from evaluation_metrics import evaluate_task
import string
import re
import torch.nn.functional as F  
import sys


try:
    # Optional: json_repair can help fix minor JSON formatting errors.
    import json_repair
except ImportError:
    json_repair = None



# New imports for visualization
import matplotlib.pyplot as plt
import seaborn as sns

from src.llamafactory.chat import ChatModel
from src.llamafactory.extras.misc import torch_gc

# -------------------------------------------------------------------
# Global Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------

def load_eval_set(data_path, tasks, size=10):
    """
    Load evaluation data from a JSON file, and sample up to `size` questions per task.
    """
    with open(data_path, 'r') as file:
        data = json.load(file)

    eval_set = {}

    if tasks == "all":
        selected_task = data.keys()
    else:
        selected_task = tasks

    for task in selected_task:
        if task in data:
            questions = data[task].get("questions", [])
            sample_size = min(size, len(questions))
            sampled_questions = random.sample(questions, sample_size) if sample_size > 0 else []

            eval_set[task] = {
                **data[task],
                "questions": sampled_questions
            }
        else:
            logging.warning(f"Task '{task}' not found in data.")

    return eval_set


def load_prompts(prompt_file):
    """
    Load a dictionary of prompts from a JSON file.
    """
    with open(prompt_file, 'r') as file:
        prompts = json.load(file)
    return prompts


def load_model_templates(model_template_file):
    """
    Load a dictionary of model templates (e.g., arguments or config) from a JSON file.
    """
    with open(model_template_file, 'r') as file:
        model_template = json.load(file)
    return model_template


def save_generations(outputs, save_path, model, prompt_type, save_as="all", ablation_idx=None):
    """
    Save model outputs in CSV, JSON, or JSONL with error catching.
    """
    logging.info(f"Saving outputs for model '{model}' with prompt type '{prompt_type}'")
    model_path = os.path.join(save_path, model)
    
    # Create directory with error catching.
    try:
        os.makedirs(model_path, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create directory {model_path}: {e}")
        return

    formats = ["csv", "json", "jsonl"] if save_as == "all" else save_as
    data_list = [{"task": key, **value} for key, value in outputs.items()]
    ablation_suffix = f"_abl{ablation_idx}" if ablation_idx is not None else ""

    # Save CSV file.
    if "csv" in formats and len(data_list) > 0:
        csv_path = os.path.join(model_path, f"{model}_{prompt_type}{ablation_suffix}.csv")
        try:
            with open(csv_path, mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=data_list[0].keys())
                writer.writeheader()
                writer.writerows(data_list)
            logging.info(f"Outputs saved as CSV at {csv_path}")
        except Exception as e:
            logging.error(f"Error saving CSV to {csv_path}: {e}")

    # Save JSON file.
    if "json" in formats:
        json_path = os.path.join(model_path, f"{model}_{prompt_type}{ablation_suffix}.json")
        try:
            with open(json_path, mode="w") as file:
                json.dump(outputs, file, indent=4)
            logging.info(f"Outputs saved as JSON at {json_path}")
        except Exception as e:
            logging.error(f"Error saving JSON to {json_path}: {e}")

    # Save JSONL file.
    if "jsonl" in formats:
        jsonl_path = os.path.join(model_path, f"{model}_{prompt_type}{ablation_suffix}.jsonl")
        try:
            with open(jsonl_path, mode="w") as file:
                for entry in data_list:
                    file.write(json.dumps(entry) + "\n")
            logging.info(f"Outputs saved as JSONL at {jsonl_path}")
        except Exception as e:
            logging.error(f"Error saving JSONL to {jsonl_path}: {e}")

def get_prompt_key(prompt_type, model, task):
    """
    Map a prompt_type to the correct key in the prompts dictionary.
    """
    if prompt_type == 'seppm':
        return f"{model}_prompt"
    elif prompt_type == 'seppt':
        return f"{task}_prompt"
    elif prompt_type == 'seppmt':
        return f"{model}_{task.upper()}_prompt"
    elif prompt_type == 'samppt':
        return "all_task_prompt"
    elif prompt_type == 'samppm':
        return "all_model_prompt"
    else:
        logging.warning(f"Unrecognized prompt type: {prompt_type}")
        return None


def format_demonstrations(demonstrations, task_type):
    """
    Convert demonstration examples into a text block.
    """
    formatted_demos = "Examples\n"
    if task_type == "cls":
        for demo in demonstrations:
            formatted_options = "\n".join(
                f"{letter}. {option}"
                for letter, option in zip(string.ascii_uppercase, demo["options"])
            )
            letter = chr(ord("A") + demo["answer_index"])
            qprompt = (
                f"Which class from the Options does the Premise belong to?\n"
                f"Premise: {demo['question']}\n"
                f"Options: \n{formatted_options}\n\n"
                f"""```json{{ "Answer": "{letter}" }}```\n\n"""
                f"===\n"
                # f"Answer: {demo["options"][demo['answer_index']]}\n\n"
            )
            formatted_demos += qprompt
    elif task_type in ["gen", "gen_summ"]:
        for demo in demonstrations:
            formatted_demos += (
                f"Prompt: {demo['prompt']}\n Context:\n{demo['context']}\n"
                f"""```json{{ "Answer": "{demo['answer']}" }}```\n\n"""
                f"===\n"
                # f"Answer: {demo['answer']}\n\n"
            )
    elif task_type == "qac":
        for demo in demonstrations:
            qprompt = (
                f"Context:\n{demo['context']}\n\n"
                f"Question:\n{demo['question']}\n\n"
                f"""```json{{ "Answer": "{demo['answer']}" }}```\n\n"""
                f"===\n"
                # f"Answer: {demo['answer']}\n\n"
            )
            formatted_demos += qprompt
    elif task_type == "mcq":
        for demo in demonstrations:
            formatted_options = "\n".join(
                f"{letter}. {option}"
                for letter, option in zip(string.ascii_uppercase, demo["options"])
            )
            letter = chr(ord("A") + demo["answer_index"])
            formatted_demos += (
                f"Question: {demo['question']}\n\n"
                f"Options\n{formatted_options}\n\n"
                f"""```json{{ "Answer": "{letter}" }}```\n\n"""
                f"===\n"
                # f"Answer: {demo["options"][demo['answer_index']]}\n\n"
            )
    elif task_type == "res":
        for demo in demonstrations:
            formatted_options = "\n".join(
                f"{letter}. {option}"
                for letter, option in zip(string.ascii_uppercase, demo["options"])
            )
            letter = chr(ord("A") + demo["answer_index"])
            formatted_demos += (
                f"Question: {demo['question']}\n"
                f"Options\n{formatted_options}\n\n"
                f"""```json{{ "Answer": "{letter}" }}```\n\n"""
                f"===\n"
            )
            # f"Answer: {demo["options"][demo['answer_index']]}\n\n"
    else:
        logging.warning(f"Unknown task type '{task_type}'. No formatting applied.")
    return formatted_demos


def get_formatted_prompt(prompt, formatted_demos, structure):
    """
    Combine the base prompt with the demonstration text block.
    """
    if structure == "icld_ssp":
        return f"Use the demos below as examples on how to answer the question\n{formatted_demos}\n{prompt}"
    elif structure == "icld_esp":
        return f"{prompt}\nUse the demos below as examples on how to answer the question\n{formatted_demos}"
    elif structure in ["icld_sum", "icld_eum"]:
        return f"{prompt}"
    else:
        return prompt  # Default


def approximate_token_length(text):
    """
    A heuristic to approximate token length.
    """
    return len(text.split())


def analyze_attention(attentions, tokens, config, **context):
    """
    Enhanced attention analysis with proper tuple handling
    """
    results = {
        'layer_data': {},
        'aggregations': {},
        'token_alignment': tokens
    }

    if config['layers'] == "all":
        layer_indices = list(range(len(attentions)))
    else:
        layer_indices = config['layers']

    total_layers = len(attentions)
    for layer_idx in layer_indices:
        if not (-total_layers <= layer_idx < total_layers):
            logging.warning(f"Layer index {layer_idx} out of range (total layers: {total_layers}). Skipping.")
            continue

        # Get layer attention tuple (could contain self-attn + cross-attn)
        layer_tuple = attentions[layer_idx]
        
        # Convert to numpy arrays and handle tuple structure
        layer_attentions = []
        for attn in layer_tuple:
            if isinstance(attn, torch.Tensor):
                # Convert tensor to numpy array
                layer_attentions.append(attn.detach().cpu().float().numpy())
            elif isinstance(attn, tuple):
                # Handle nested tuples (e.g., cross-attention layers)
                for sub_attn in attn:
                    if isinstance(sub_attn, torch.Tensor):
                        layer_attentions.append(sub_attn.detach().cpu().numpy())
            else:
                logging.warning(f"Unexpected attention format at layer {layer_idx}: {type(attn)}")

        # Stack attention heads for analysis
        try:
            # Combine all attention matrices for the layer
            layer_att = np.stack(layer_attentions, axis=0)  # Shape: (num_heads, seq_len, seq_len)
        except ValueError as e:
            logging.error(f"Could not stack attention matrices: {str(e)}")
            continue

        # Head selection
        if config['heads'] != 'all':
            layer_att = layer_att[config['heads']]
            
        # Store raw attention
        results['layer_data'][layer_idx] = layer_att
        
        # Calculate aggregations
        if config['aggregation'] == 'mean':
            results['aggregations'][layer_idx] = np.mean(layer_att, axis=0)
        elif config['aggregation'] == 'max':
            results['aggregations'][layer_idx] = np.max(layer_att, axis=0)
            
    return results

def visualize_attention_heatmaps(analysis_results, output_path, task, question_id,
                               save_dir='attention_visualizations', heatmap_size=(35, 20), 
                               dpi=300, max_heads=16):
    try:
        vis_paths = []
        for layer_idx, layer_data in analysis_results['layer_data'].items():
            if len(layer_data.shape) == 4:
                logging.warning(f"Percentage of zeros before KV averaging: {np.mean(layer_data == 0) * 100:.2f}%")
                logging.warning(f"Layer {layer_idx} KV Groups - Before Averaging:")
                logging.warning(f"Min: {np.min(layer_data)}, Max: {np.max(layer_data)}, Mean: {np.mean(layer_data)}")
                sys.stdout.flush()

                print(f"Percentage of zeros before KV averaging: {np.mean(layer_data == 0) * 100:.2f}%")
                print(f"Layer {layer_idx} KV Groups - Before Averaging:")
                print(f"Min: {np.min(layer_data)}, Max: {np.max(layer_data)}, Mean: {np.mean(layer_data)}")

                weights = layer_data.sum(axis=-1, keepdims=True)
                weights /= weights.sum(axis=1, keepdims=True) + 1e-10
                
                layer_data = (layer_data * weights).sum(axis=1)

                # layer_data = layer_data.mean(axis=1)  # Average over key-value groups
                logging.warning(f"Percentage of zeros after KV averaging: {np.mean(layer_data == 0) * 100:.2f}%")
                logging.warning(f"Layer {layer_idx} KV Groups - After Averaging:")
                logging.warning(f"Min: {np.min(layer_data)}, Max: {np.max(layer_data)}, Mean: {np.mean(layer_data)}")
                sys.stdout.flush()

                print(f"Percentage of zeros after KV averaging: {np.mean(layer_data == 0) * 100:.2f}%")
                print(f"Layer {layer_idx} KV Groups - After Averaging:")
                print(f"Min: {np.min(layer_data)}, Max: {np.max(layer_data)}, Mean: {np.mean(layer_data)}")

            layer_data = np.squeeze(layer_data)  # Shape becomes (num_heads, seq_len, seq_len)
            # Verify dimensions after squeezing
            if len(layer_data.shape) != 3:
                logging.warning(f"Invalid attention shape {layer_data.shape} for layer {layer_idx}")
                continue
                
            num_heads = layer_data.shape[0]
            if num_heads > max_heads:
                logging.warning(f"Skipping visualization for layer {layer_idx} with {num_heads} heads")
                continue

            # Create subplots
            fig, axes = plt.subplots(math.ceil(num_heads/4), 4, figsize=heatmap_size)
            axes = axes.flatten()

            for head_idx in range(num_heads):
                # Get 2D attention matrix for this head
                head_matrix = layer_data[head_idx]
                
                if head_matrix.ndim != 2:
                    head_matrix = np.squeeze(head_matrix)
                    if head_matrix.ndim != 2:
                        logging.warning(f"Unexpected head matrix shape {head_matrix.shape} for layer {layer_idx} head {head_idx}")
                        continue
                
                sns.heatmap(
                    head_matrix,
                    ax=axes[head_idx],
                    cmap="viridis",
                    xticklabels=50,
                    yticklabels=50
                )
                axes[head_idx].set_title(f"Layer {layer_idx} Head {head_idx}")
                axes[head_idx].tick_params(axis='both', labelsize=6)

            # Cleanup and saving
            for ax in axes[num_heads:]:
                ax.axis('off')
                
            plt.tight_layout()
            unique_filename = f"attention_{task}_q_{question_id}_L_{layer_idx}.png"
            vis_dir = os.path.join(output_path, save_dir)
            os.makedirs(vis_dir, exist_ok=True)
            # vis_path = os.path.join(vis_dir, f"attention_L{layer_idx}.png")
            vis_path = os.path.join(vis_dir, unique_filename)
            plt.savefig(vis_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            vis_paths.append(vis_path) 

        return vis_paths
            
    except Exception as e:
        logging.error(f"Visualization failed: {str(e)}")
        return None

import matplotlib.colors as mcolors

def visualize_heatmap(
    analysis_results,
    output_path,
    task,
    question_id,
    do_layers=True,
    save_dir='attention_visualizations_222',
    heatmap_size=(12, 10),
    dpi=800,
    max_heads=16
):
    """
    Creates one heatmap per layer, where the heatmap has:
      - rows = heads
      - columns = token positions
    by averaging each head's attention matrix over its 'query' axis.

    Parameters:
      analysis_results: Dict containing attention data in analysis_results['layer_data'] (or 'aggregations')
      output_path:      Directory to save visualizations
      task:             Name of the task (used in filename)
      question_id:      Identifier for the question (used in filename)
      do_layers:        Whether to use 'layer_data' or 'aggregations' from analysis_results
      save_dir:         Subdirectory to store output
      heatmap_size:     (width, height) of the figure in inches
      dpi:              Dots per inch for saving the figure
      max_heads:        Skip layers with more heads than this

    Returns:
      List of file paths to the saved heatmaps
    """
    try:
        vis_paths = []
        # Choose the appropriate data source
        if do_layers:
            process = analysis_results['layer_data'].items()
        else:
            process = analysis_results['aggregations'].items()
        logging.info(f"Processing {len(process)} layers for visualization.")
        logging.info(f"Process shape: {process}")

        for layer_idx, layer_data in process:
            # Squeeze out extra dimensions
            if layer_data.ndim > 3:
                layer_data = layer_data.squeeze()

            # layer_data should now be (num_heads, seq_len, seq_len) in a typical self-attention scenario,
            # or possibly (num_heads, something, something) if rectangular.
            num_heads = layer_data.shape[0]
            if num_heads > max_heads:
                logging.warning(
                    f"Skipping layer {layer_idx} because it has {num_heads} heads (> {max_heads})."
                )
                continue
            
            # We'll aggregate each head's 2D matrix into a 1D vector of length seq_len
            # by averaging over the query axis (axis=0).
            # So if shape is (seq_len, seq_len), we do .mean(axis=0).
            # That leaves shape (seq_len,) for each head.
            aggregated_heads = []
            for head_idx in range(num_heads):
                head_matrix = layer_data[head_idx]
                # Ensure it's 2D
                logging.info(f"Layer {layer_idx} has {num_heads} heads and shape {head_matrix.shape}.")
                if head_matrix.ndim != 2:
                    head_matrix = head_matrix.squeeze()
                    if head_matrix.ndim != 2:
                        logging.warning(
                            f"Unexpected shape {head_matrix.shape} for layer {layer_idx}, head {head_idx}"
                        )
                        continue
                
                # Average across the "query" axis to get a 1D distribution over tokens
                row_vector = head_matrix.mean(axis=0)  # shape: (seq_len,)
                aggregated_heads.append(row_vector)

            if not aggregated_heads:
                # If we couldn't parse any heads, skip plotting
                logging.warning(f"No valid heads found for layer {layer_idx}.")
                continue

            # Stack into shape (num_heads, seq_len)
            aggregated_matrix = np.vstack(aggregated_heads)

            # Plot as a single heatmap: rows = heads, columns = token positions
            plt.figure(figsize=heatmap_size)
            sns.heatmap(
                aggregated_matrix,
                cmap="viridis",
                norm=mcolors.LogNorm(),  # Log scale can help if values vary widely
                cbar=True,
                xticklabels=50,  # or provide your token labels if available
                yticklabels=[f"H{h}" for h in range(num_heads)]
            )
            plt.title(f"Layer {layer_idx} - Heads vs. Tokens\n(Task={task}, QID={question_id})")
            plt.xlabel("Token Position")
            plt.ylabel("Head Index")

            # Save the figure
            vis_dir = os.path.join(output_path, save_dir)
            os.makedirs(vis_dir, exist_ok=True)
            filename = f"attention_{task}_q_{question_id}_L_{layer_idx}.png"
            vis_path = os.path.join(vis_dir, filename)
            plt.savefig(vis_path, dpi=dpi, bbox_inches='tight')
            plt.close()

            vis_paths.append(vis_path)

        return vis_paths

    except Exception as e:
        logging.error(f"Visualization failed: {str(e)}")
        return None


def attention_rollout(attentions, method="weighted", residual_weight=0.1):
    """Handles variable sequence lengths and GQA architecture"""
    # Convert nested structures to tensors with padding
    processed = []
    max_seq = max(layer.shape[-1] for layer in attentions)  # Find max sequence length
    
    for layer in attentions:
        # Convert numpy arrays and handle GQA tuples [Search Result 3,9]
        if isinstance(layer, (list, tuple)):
            layer = [torch.from_numpy(l) if isinstance(l, np.ndarray) else l for l in layer]
            layer = torch.stack(layer)
        elif isinstance(layer, np.ndarray):
            layer = torch.from_numpy(layer).float()
            
        # Pad sequence dimensions to match max length [Search Result 4,8]
        pad_size = max_seq - layer.shape[-1]
        if pad_size > 0:
            layer = F.pad(layer, (0, pad_size, 0, pad_size))  # (left, right, top, bottom)
            
        processed.append(layer)

    try:
        # Merge GQA heads/groups if needed [Search Result 3]
        if processed[0].dim() == 5:  # (layers, heads, groups, seq, seq)
            processed = [layer.view(layer.size(0), -1, *layer.shape[3:]) for layer in processed]
            
        attentions = torch.stack(processed)
    except RuntimeError as e:
        logging.error(f"Stacking failed after padding: {str(e)}")
        return None

    # Continue with rollout logic
    num_layers, num_heads, seq_len, _ = attentions.shape
    rollout = torch.eye(seq_len, device=attentions.device)

    for layer_idx in range(num_layers):
        layer_attn = attentions[layer_idx]
        
        # Head aggregation [Search Result 10]
        if method == "mean":
            layer_attn = layer_attn.mean(dim=0)
        elif method == "max":
            layer_attn = layer_attn.max(dim=0)[0]
        elif method == "weighted":
            weights = layer_attn.sum(dim=-1, keepdims=True)
            weights /= weights.sum(dim=1, keepdims=True) + 1e-10
            layer_attn = (layer_attn * weights).sum(dim=0)

        # Add residual connection and normalize [Search Result 6]
        layer_attn += residual_weight * torch.eye(seq_len, device=attentions.device)
        layer_attn /= layer_attn.sum(dim=-1, keepdims=True) + 1e-10
        
        # Stable attention propagation
        rollout = 0.7 * (rollout @ layer_attn) + 0.3 * rollout

    return rollout.cpu().numpy()


def rollout_visualization(
    rollout,  # Required
    output_path,  # Required
    tokens,  # Required
    task,  # Required
    question_id,  # Required
    save_dir='attention_rollouts',  # Optional
    heatmap_size=(40, 40),  # Optional
    dpi=1000,  # Optional
    save_format='png'  # Optional
):
    # Implementation

    """
    Visualize the attention rollout as a heatmap with token labels.
    """
    fig, ax = plt.subplots(figsize=(40, 40))
    import matplotlib.colors as colors
    # Create heatmap with token labels
    def shorten(token, max_len=8):
        return token[:max_len] + "..." if len(token) > max_len else token

    short_tokens = [shorten(t) for t in tokens]
    rollout_norm = rollout / (rollout.sum(axis=1, keepdims=True) + 1e-10)
    nonzero_vals = rollout_norm[rollout_norm > 0]
    sns.heatmap(
        rollout_norm,
        ax=ax, 
        cmap="viridis",
        norm=colors.LogNorm(
            vmin=nonzero_vals.min() if nonzero_vals.size else 1e-10, 
            vmax=rollout_norm.max()
        ),
        xticklabels=short_tokens,
        yticklabels=short_tokens,
        square=True
    )
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90, fontsize=3)
    plt.yticks(rotation=0, fontsize=3)
    
    ax.set_title("Attention Rollout")
    ax.tick_params(axis='both', labelsize=6)
    plt.tight_layout()
    
    # Save the visualization
    unique_filename = f"attention_{task}_q_{question_id}_rollout.{save_format}"
    vis_dir = os.path.join(output_path, save_dir)
    os.makedirs(vis_dir, exist_ok=True)
    vis_path = os.path.join(vis_dir, unique_filename)
    plt.savefig(vis_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return vis_path


def process_questions(
    task_data, 
    formatted_prompt, 
    formatted_demos, 
    structure, 
    chat_model,
    output_path,
    max_length_threshold=8192,
    attention_analysis=False,
    attention_config=None,
    **input_kwargs
):
    """
    Process each question: build prompt, check length, run the model, and (if enabled)
    analyze and visualize attention weights.
    """
    questions_output = []
    default_attention_config = {
        'layers': 'all',  # Analyze last layer by default
        'heads': 'all',  # Can be list of head indices or 'all'
        'aggregation': 'max',  # 'mean', 'max', 'weighted', or 'none'

    }
    attention_config = {**default_attention_config, **(attention_config or {})}

    for q_idx, quest in enumerate(task_data["questions"]):
        if task_data["task_type"] == "cls":
            formatted_options = "\n".join(
                f"{letter}. {option}"
                for letter, option in zip(string.ascii_uppercase, quest["options"])
            )

            qprompt = (
                f"Which class from the Options does the Premise belong to?\nYour answer can only be from the options provided.\n"
                f"Premise: {quest['question']}\n"
                f"Options: \n{formatted_options}\n\n"
            )
        elif task_data["task_type"] == "mcq":
            formatted_options = "\n".join(
                f"{letter}. {option}"
                for letter, option in zip(string.ascii_uppercase, quest["options"])
            )
            qprompt = (
                f"Your answer can only be from the options provided.\n"
                f"Question: {quest['question']}\n\n"
                f"Options\n{formatted_options}"
            )
        elif task_data["task_type"] in ["gen", "gen_summ"]:
            qprompt = f"Prompt: {quest['prompt']} \n Context:\n{quest['context']}"

        elif task_data["task_type"] == "qac":
            qprompt = (
                f"Context:\n{quest['context']}\n\n"
                f"Question:\n{quest['question']}\n\n"
                f"Answer the question based on the context."
            )
        elif task_data["task_type"] == "res":
            formatted_options = "\n".join(
                f"{letter}. {option}"
                for letter, option in zip(string.ascii_uppercase, quest["options"])
            )
            qprompt = (
                f"Your answer can only be from the options provided.\n"
                f"Question: {quest['question']}\n"
                f"Options\n{formatted_options}"
            )
        else:
            logging.warning(f"Task type '{task_data['task_type']}' is not recognized. Skipping.")
            continue

        if structure in ["icld_ssp", "icld_esp"]:
            # """```json{{ "Answer": "{letter}" }}```"""
            formatted_question = f"""{qprompt}\n\nAnswer: """
        elif structure == "icld_sum":
            formatted_question = f"Use the demos below as examples on how to answer the question{formatted_demos}\n\n{qprompt}\n\nAnswer: "
        elif structure == "icld_eum":
            formatted_question = f"Answer this question\n{qprompt}\n\nUse the demos below as examples on how to answer the question\n{formatted_demos}\nAnswer: "
        else:
            formatted_question = qprompt  # fallback

        messages = [{"role": "user", "content": formatted_question}]
        logging.debug(f"messages: {messages}")
        total_prompt_text = formatted_prompt + "\n" + formatted_question
        print(50 * "-")
        print(total_prompt_text)
        print(50 * "-")
        approx_tokens = approximate_token_length(total_prompt_text)
        if approx_tokens > max_length_threshold:
            logging.warning(
                f"Skipping question due to length ({approx_tokens} tokens > {max_length_threshold})."
            )
            continue

        # --- Generate response and optionally collect attention ---
        response = ""
        question_entry = quest.copy()
        if attention_analysis:
            # input_kwargs["stop"] = "===\n"
            response, attentions, tokens = chat_model.chat_with_attentions(
                messages, 
                system=formatted_prompt,
                **input_kwargs
            )

            
            # Process attention with config
            analysis_results = analyze_attention(
                attentions=attentions,
                tokens=tokens,
                config=attention_config,
                formatted_demos=formatted_demos,
                demonstrations=task_data.get("demonstrations", [])
            )

            # logging.info(f"Analysis results: {analysis_results}")
            # logging.info(f"Attention shape: {attentions.shape}")
            logging.info(f"layer_data keys: {analysis_results['layer_data'].keys()}")
            logging.info(f"aggregations keys: {analysis_results['aggregations'].keys()}")
            logging.info(f"layer_data shapes: {[layer.shape for layer in analysis_results['layer_data'].values()]}")
            logging.info(f"aggregations shapes: {[layer.shape for layer in analysis_results['aggregations'].values()]}")
            
            # Visualization with config
            if attention_config.get('visualize'):
                v2_paths = visualize_heatmap(
                    analysis_results=analysis_results,
                    output_path=output_path,
                    task=structure,
                    question_id=q_idx,
                    **attention_config['visualize']
                )
                logging.info(f"Saved attention visualization at: {v2_paths}")
                question_entry["attention_heatmap"] = v2_paths
            # analysis_results = None


            if attention_config.get('rollout'):
                # Pad attention matrices to common shape before stacking
                ordered_layers = sorted(analysis_results['layer_data'].keys())
                attentions_list = [
                    analysis_results['layer_data'][layer] 
                    for layer in ordered_layers
                ]
                rollout = attention_rollout(
                    attentions_list,
                    method=attention_config['rollout'].get('method', 'weighted'),
                    residual_weight=0.1
                )

                logging.info(f"Rollout shape: {rollout.shape}")
                logging.info(f"Rollout sum: {np.sum(rollout)}")
                logging.info(f"Rollout max: {np.max(rollout)}")
                logging.info(f"Rollout min: {np.min(rollout)}")
                logging.info(f"Rollout mean: {np.mean(rollout)}")
                logging.info(f"Rollout std: {np.std(rollout)}")
                logging.info(f"Rollout: {rollout}")

                vis_config = attention_config['rollout'].copy()
                vis_path = rollout_visualization(
                    rollout=rollout,
                    output_path=output_path,
                    tokens=tokens,
                    task=structure,
                    question_id=q_idx,
                    **vis_config
                )
                question_entry["attention_rollout"] = vis_path

            
        else:
            # input_kwargs["stop"] = "===\n"
            for new_text in chat_model.stream_chat(messages, system=formatted_prompt, **input_kwargs):
                response += new_text
            analysis_results = None

        question_entry["model_answer"] = response
        messages = []

        questions_output.append(question_entry)
        logging.debug(f"Processed question with formatted demos:\n{formatted_demos}\n\nPrompt: {formatted_prompt}")

    return questions_output


def randomize_demonstrations(demonstrations):
    """
    Shuffle the demonstration set in-place.
    """
    random.shuffle(demonstrations)


def ablation_splits(demonstrations):
    """
    Generate subsets of demonstrations for ablation.
    """
    for i in range(len(demonstrations)):
        yield i, demonstrations[:i] + demonstrations[i+1:]


# -------------------------------------------------------------------
# Main Runner
# -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation with models, tasks, prompts, and perform attention analysis."
    )
    parser.add_argument(
        '--models', 
        nargs='+', 
        default=['LLAMA3_8B', 'LLAMA3_70B'], 
        help="List of models."
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['cnn_dailymail'],
        choices=[
            'mmlu','arc','glue','squad','gov_rep_sum','commonsenseqa','sst2',
            'ag_news','trec','rte','mnli','snli','natural_questions','triviaqa',
            'gsm8k','human_eval','mbpp','cnn_dailymail','xsum','bbh','helm'
        ],
        help="List of tasks/datasets to evaluate."
    )
    parser.add_argument(
        '--prompts', 
        nargs='+', 
        default=['seppt'], 
        choices=['seppm', 'seppt', 'seppmt', 'samppt', 'samppm'], 
        help="List of prompts."
    )
    parser.add_argument(
        '--prompt_structure', 
        nargs='+', 
        default=['icld_ssp', 'icld_esp', 'icld_sum'], 
        choices=['icld_ssp', 'icld_esp', 'icld_sum', 'icld_eum'], 
        help="List of prompt structures."
    )
    parser.add_argument(
        "--prompt_file", 
        type=str, 
        required=True, 
        help="Path to the file containing different prompts"
    )
    parser.add_argument(
        "--model_template_file", 
        type=str, 
        required=True, 
        help="Path to the file containing different model templates"
    )
    parser.add_argument(
        '--input_path', 
        type=str, 
        required=True, 
        help="Path to the JSON file with tasks data."
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        required=True, 
        help="Path to output folder."
    )
    parser.add_argument(
        '--size', 
        type=int, 
        default=1,
        help="Number of questions to sample per task."
    )
    parser.add_argument(
        '--randomize_demos', 
        action='store_true', 
        help="If set, randomly shuffle demonstration order before formatting."
    )
    parser.add_argument(
        '--ablation', 
        action='store_true', 
        help="If set, systematically remove one demonstration at a time and rerun."
    )
    # New flags for attention analysis and visualization
    parser.add_argument(
        '--attention_analysis', 
        action='store_true', 
        help="If set, perform attention analysis and capture attention weights from the model."
    )
    parser.add_argument(
        '--gen_config_file',
        type=str,
        required=True,
        help="Path to the file containing generation configuration."
    )
    parser.add_argument(
        '--visualize_attention', 
        action='store_true', 
        help="If set, generate attention heatmap visualizations for a selected head."
    )
    return parser.parse_args()



def extract_answer(pred: str, question: dict) -> str:
    """Robust answer extraction from messy model outputs.
    
    This function first attempts to extract valid JSON blocks (using optional repair).
    If that fails, it uses multiple-choice heuristics (if options are provided),
    then searches for common answer markers (e.g., "Answer:" or "Solution:"), 
    checks for yes/no answers, and finally returns a cleaned version of the output.
    """
    # Clean up whitespace.
    cleaned_pred = re.sub(r"\s+", " ", pred.strip())
    
    def extract_json_blocks(text: str):
        """Extracts JSON-like substrings by matching balanced braces."""
        blocks = []
        brace_stack = []
        start_idx = None
        for i, ch in enumerate(text):
            if ch == '{':
                if not brace_stack:
                    start_idx = i
                brace_stack.append(ch)
            elif ch == '}':
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack and start_idx is not None:
                        blocks.append(text[start_idx:i+1])
                        start_idx = None
        return blocks

    # Step 1: Attempt JSON extraction and parsing.
    json_blocks = extract_json_blocks(cleaned_pred)
    for jb in json_blocks:
        parsed = None
        try:
            parsed = json.loads(jb)
        except json.JSONDecodeError:
            # If json_repair is available, try to repair and parse the JSON.
            if json_repair:
                try:
                    repaired = json_repair.repair_json(jb)
                    parsed = json.loads(repaired)
                except Exception:
                    parsed = None
        if parsed is not None:
            # If the JSON is a dict, try to return the value under "answer"/"Answer".
            if isinstance(parsed, dict):
                if "answer" in parsed:
                    return str(parsed["answer"]).strip()
                elif "Answer" in parsed:
                    return str(parsed["Answer"]).strip()
                else:
                    # If no answer key exists, return the dict as a string.
                    return str(parsed).strip()
            # If it is a list, assume the first element holds the answer.
            elif isinstance(parsed, list):
                if parsed:
                    first = parsed[0]
                    if isinstance(first, dict):
                        if "answer" in first:
                            return str(first["answer"]).strip()
                        elif "Answer" in first:
                            return str(first["Answer"]).strip()
                    else:
                        return str(first).strip()
            else:
                # For primitive types (string, number, etc.), just return the value.
                return str(parsed).strip()

    # Step 2: Handle multiple-choice answers if options are provided.
    options = question.get("options", [])
    if options:
        # Look for a letter match (e.g., A, B, C, or D) in the last 100 characters.
        letter_match = re.search(r"\b([A-D])\b", cleaned_pred[-100:], re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()
        
        # If no direct letter is found, try matching the option texts.
        option_texts = [opt.strip().lower() for opt in options]
        for idx, opt_text in enumerate(option_texts):
            if re.search(rf"\b{re.escape(opt_text)}\b", cleaned_pred.lower()):
                return string.ascii_uppercase[idx]

    # Step 3: Look for explicitly marked answers in JSON-like key/value format.
    json_match = re.search(
        r'"Answer"\s*:\s*"((?:\\"|[^"])+)"',
        cleaned_pred,
        re.IGNORECASE | re.DOTALL
    )
    if json_match:
        return json_match.group(1).replace('\\"', '"').strip()

    # Step 4: Search for an "Answer:" or "Solution:" prefix.
    answer_line = re.search(
        r"(?:Answer|Solution)[:\s\-]+(.+?)(?=\n\n|\Z)",
        cleaned_pred,
        re.IGNORECASE | re.DOTALL
    )
    if answer_line:
        candidate = answer_line.group(1).strip()
        # If options exist, verify if candidate matches one.
        if options:
            lower_opts = [o.lower() for o in options]
            if candidate.lower() in lower_opts:
                return string.ascii_uppercase[lower_opts.index(candidate.lower())]
        return candidate

    # Step 5: Detect yes/no answers.
    if (yn_match := re.search(r"\b(Yes|No)\b", cleaned_pred, re.IGNORECASE)):
        return yn_match.group(1).capitalize()

    # Step 6: Fallback to return the cleaned text.
    return cleaned_pred

def clean_sentence(text: str) -> str:
    """Extracts complete sentence segments with smart truncation"""
    # Find natural breaks in the text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if sentences:
        # Prefer last complete sentence
        last_complete = next(
            (s for s in reversed(sentences) if len(s.split()) >= 3),
            None
        )
        if last_complete:
            return last_complete.strip()
    
    # Fallback to first meaningful segment
    meaningful = re.search(r"(.{10,}?)(?:\s|$)", text)
    return meaningful.group(1).strip() + "..." if meaningful else text.strip()


def get_reference(q):
    # Prioritize keys that might contain indices
    if "answer_index" in q:
        answer = chr(ord("A") + q["answer_index"])
        return answer
    elif "answer" in q:
        answer = q["answer"]
        return answer.strip()
    else:
        raise KeyError(f"No answer key found in question: {q.keys()}")

def get_prediction(q):
    pred = q["model_answer"]
    if isinstance(pred, int) and "options" in q:
        return q["options"][pred].strip()  # Return option text for classification
    return str(pred).strip()


def main():
    logging.info("PyTorch Version: %s", torch.__version__)
    logging.info("CUDA Version: %s", torch.version.cuda)
    logging.info("Is CUDA Available: %s", torch.cuda.is_available())

    args = parse_args()
    logging.info(f"Running with arguments: {args}")

    # 1. Load data, prompts, and model templates
    eval_set = load_eval_set(args.input_path, args.tasks, size=args.size)
    model_templates = load_model_templates(args.model_template_file)
    prompts = load_prompts(args.prompt_file)
    task_config = load_model_templates(args.gen_config_file)

    # 2. Prepare top-level results dict: results[model][task][structure] = ...
    results_root = {}
    chat_model = None

    for model in args.models:
        # Get the base arguments for this model.
        base_cm_args = model_templates.get(model, {})
        if not base_cm_args:
            logging.warning(f"Template for model '{model}' not found. Using default arguments.")
            base_cm_args = {}
        cm_args = base_cm_args.copy()
        # generation_args = cm_args.get("generation_args", {})
        if args.attention_analysis:
            cm_args.update({"flash_attn": "disabled"})
            cm_args.update({"infer_dtype": "float32"})
        logging.info(f"Initializing ChatModel for '{model}' with arguments: {cm_args}")
        chat_model = ChatModel(args=cm_args)

        results_root[model] = {}

        for task, task_data in eval_set.items():
            task_type = task_data.get("task_type", None)

            cf = f'{model}_{task}'
            if cf in task_config:
                logging.info(f"Found task-specific config for '{cf}'.")
                # Update the base arguments with task-specific settings.
                generation_args = task_config[cf]
                logging.info(f"Running generation for '{model}' for task '{task}' with generation configs: {generation_args}")

            else:
                # Default modifications or leave as is.
                logging.info(f"No task-specific modifications for task type '{cf}'.")

            results_root[model][task] = {}

            demonstrations = task_data.get("demonstrations", [])
            if not demonstrations:
                logging.warning(f"Task '{task}' has no demonstrations. Skipping.")
                continue

            if args.randomize_demos:
                randomize_demonstrations(demonstrations)

            results_root[model][task] = {}

            # Ablation: use full demos or remove one demonstration at a time.
            ablation_sets = [(None, demonstrations)]
            if args.ablation:
                ablation_sets = list(ablation_splits(demonstrations))

            for ablation_idx, demo_subset in ablation_sets:
                ablation_label = f"abl_{ablation_idx}" if ablation_idx is not None else "full"
                results_root[model][task][ablation_label] = {}

                formatted_demos = format_demonstrations(demo_subset, task_type)

                for prompt_type in args.prompts:
                    prompt_key = get_prompt_key(prompt_type, model, task)
                    if not prompt_key:
                        logging.warning(f"Skipping prompt_type '{prompt_type}' for '{model}/{task}'.")
                        continue

                    prompt_template = prompts.get(prompt_key)
                    if not prompt_template:
                        logging.warning(f"Prompt key '{prompt_key}' not found. Using a default prompt.")
                        prompt_template = "Answer this question."

                    for structure in args.prompt_structure:
                        if structure not in ["icld_ssp", "icld_esp", "icld_sum", "icld_eum"]:
                            logging.warning(f"Unknown structure '{structure}'. Skipping.")
                            continue

                        if structure not in results_root[model][task][ablation_label]:
                            results_root[model][task][ablation_label][structure] = {
                                "demonstrations": demo_subset,
                                "questions": [],
                                "metric_scores": {}
                            }

                        formatted_prompt = get_formatted_prompt(prompt_template, formatted_demos, structure)

                        if args.attention_analysis:
                            generation_args["flash_attn"] = "disabled"
                            # cm_args["infer_dtype"] = "float32"
                        
                        model_task_path = os.path.join(args.output_path, model, task, ablation_label, structure)
                        os.makedirs(model_task_path, exist_ok=True)
                        questions_output = process_questions(
                            task_data=task_data,
                            formatted_prompt=formatted_prompt,
                            formatted_demos=formatted_demos,
                            structure=structure,
                            output_path=model_task_path,
                            chat_model=chat_model,
                            attention_analysis=args.attention_analysis,
                            visualize_attention=args.visualize_attention,
                            attention_config={
                                'layers': [-2,-1],  # Last two layers
                                # 'heads': [3, 5, 7],  # Specific heads
                                'aggregation': 'max',
                            },
                            **generation_args
                        )
                        predictions = [get_prediction(q) for q in questions_output]
                        cleaned_predictions = []
                        for question, pred in zip(questions_output, predictions):
                            answer = extract_answer(pred, question)
                            question["cleaned_answer"] = answer
                            question["predicted_answer"] = pred
                            cleaned_predictions.append(answer if answer else pred)
                        references = [get_reference(q) for q in questions_output]
    
                        evaluation_results = evaluate_task(task, cleaned_predictions, references)

                        print("predictions", predictions)
                        print("cleaned_predictions", cleaned_predictions)
                        print("references", references)

                        results_root[model][task][ablation_label][structure]["questions"] = questions_output
                        results_root[model][task][ablation_label][structure]["metric_scores"] = evaluation_results

                    save_generations(
                        outputs=results_root[model][task][ablation_label],
                        save_path=args.output_path,
                        model=model,
                        prompt_type=f"{task}_{ablation_label}_{prompt_type}",
                        save_as=["json"],
                        ablation_idx=ablation_idx
                    )

        del chat_model
        torch_gc()


    for model in results_root:
        model_save_path = os.path.join(args.output_path, model)
        os.makedirs(model_save_path, exist_ok=True)
        combined_json_path = os.path.join(model_save_path, f"{model}_all_results.json")
        with open(combined_json_path, mode="w") as file:
            json.dump(results_root[model], file, indent=4)
        logging.info(f"Saved combined results for model '{model}' at: {combined_json_path}")


if __name__ == "__main__":
    main()
