import os
import json
import glob
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi

# 1. PARSE RESULTS INTO A DATAFRAME

def load_metrics_from_json(json_path):
    """
    Reads one JSON file and returns a list of records:
    [
      {
        'model': ...,
        'dataset': ...,
        'sub_task': ...,
        'metric': ...,
        'value': ...
      },
      ...
    ]
    """
    records = []
    # Example: Mistral_7B_ag_news_full_seppmt.json
    filename = os.path.basename(json_path)
    # We'll guess the 'dataset' from the filename (everything after the 2nd underscore)
    # e.g. 'Mistral_7B_ag_news_full_seppmt.json' -> 'ag_news_full_seppmt'
    parts = filename.split('_')
    model_name = '_'.join(parts[:2])  # e.g. Mistral_7B
    dataset_name = '_'.join(parts[2:])[:-5]  # remove '.json'

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # The file has top-level keys like "icld_ssp", "icld_esp", etc.
    # Each has "metric_scores" sub-object. We'll gather them all.
    for sub_task, content in data.items():
        if not isinstance(content, dict):
            continue
        if 'metric_scores' not in content:
            continue

        metric_scores = content['metric_scores']
        # metric_scores is typically a dict of {metric_name: value or list}
        for metric_name, metric_value in metric_scores.items():
            # Some metrics might be lists. We'll store the average if so.
            if isinstance(metric_value, list):
                # Or store them individually. Here, let's just store the mean.
                if len(metric_value) > 0:
                    val = sum(metric_value) / len(metric_value)
                else:
                    val = None
            else:
                val = metric_value

            record = {
                'model': model_name,
                'dataset': dataset_name,
                'sub_task': sub_task,     # e.g. 'icld_ssp'
                'metric': metric_name,    # e.g. 'accuracy'
                'value': val              # e.g. 0.8
            }
            records.append(record)

    return records


def build_dataframe(base_dir='model_results'):
    """
    Walks through each model folder in base_dir, loads JSON files,
    and returns a pandas DataFrame of all metric scores.
    """
    all_records = []
    # e.g. base_dir/Mistral_7B/*.json
    # We'll do a two-level loop: subdirs (each is a model), then *.json
    for model_folder in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_folder)
        if not os.path.isdir(model_path):
            continue

        json_files = glob.glob(os.path.join(model_path, '*.json'))
        for jf in json_files:
            all_records.extend(load_metrics_from_json(jf))

    df = pd.DataFrame(all_records)
    return df


# 2. EXAMPLE PLOTS

def plot_grouped_bar(df, metric='accuracy'):
    """
    Shows how each model performs on a single metric across all datasets.
    """
    # Filter only rows matching the chosen metric
    subset = df[df['metric'] == metric]
    # We'll group by model & dataset, then average in case multiple subtasks
    grouped = subset.groupby(['model','dataset'])['value'].mean().reset_index()

    # Pivot so each dataset is on x-axis, each model is a series
    pivoted = grouped.pivot(index='dataset', columns='model', values='value')
    pivoted.plot(kind='bar', figsize=(10,6))
    plt.title(f"Grouped Bar Chart: {metric}")
    plt.ylabel(metric)
    plt.xlabel('Dataset')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # plt.show()
    vis_dir = os.path.join('result_plots', "grouped_bar")
    os.makedirs(vis_dir, exist_ok=True)
    filename = f"{metric}_grouped_bar.png"
    vis_path = os.path.join(vis_dir, filename)
    plt.savefig(vis_path, dpi=600, bbox_inches='tight')
    plt.close()


def plot_heatmap(df, metric='accuracy'):
    """
    Creates a heatmap of model x dataset for a single metric.
    """
    subset = df[df['metric'] == metric]
    grouped = subset.groupby(['model','dataset'])['value'].mean().reset_index()
    pivoted = grouped.pivot(index='model', columns='dataset', values='value')

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivoted, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f"Heatmap of {metric}")
    plt.xlabel('Dataset')
    plt.ylabel('Model')
    plt.tight_layout()
    vis_dir = os.path.join('result_plots', "heatmaps")
    os.makedirs(vis_dir, exist_ok=True)
    filename = f"{metric}_heatmap.png"
    vis_path = os.path.join(vis_dir, filename)
    plt.savefig(vis_path, dpi=600, bbox_inches='tight')
    plt.close()


def plot_boxplot(df, metric='accuracy'):
    """
    Shows a boxplot for a single metric, grouped by model (across all datasets/subtasks).
    """
    subset = df[df['metric'] == metric]
    plt.figure(figsize=(8,6))
    sns.boxplot(x='model', y='value', data=subset)
    plt.title(f"Boxplot of {metric} by Model")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    vis_dir = os.path.join('result_plots', "boxplot")
    os.makedirs(vis_dir, exist_ok=True)
    filename = f"{metric}_boxplot.png"
    vis_path = os.path.join(vis_dir, filename)
    plt.savefig(vis_path, dpi=600, bbox_inches='tight')
    plt.close()


def plot_scatter(df, metric_x='accuracy', metric_y='rouge_rouge1'):
    """
    Shows how two metrics correlate. Each point is a (model, dataset, sub_task).
    """
    # We'll do a left-join on (model, dataset, sub_task) to get both metrics side by side.
    df_x = df[df['metric'] == metric_x][['model','dataset','sub_task','value']]
    df_x.rename(columns={'value': metric_x}, inplace=True)
    df_y = df[df['metric'] == metric_y][['model','dataset','sub_task','value']]
    df_y.rename(columns={'value': metric_y}, inplace=True)

    merged = pd.merge(df_x, df_y, on=['model','dataset','sub_task'], how='inner')

    plt.figure(figsize=(6,6))
    sns.scatterplot(data=merged, x=metric_x, y=metric_y, hue='model')
    plt.title(f"Scatter Plot: {metric_x} vs. {metric_y}")
    plt.tight_layout()
    vis_dir = os.path.join('result_plots', "scatter")
    os.makedirs(vis_dir, exist_ok=True)
    filename = f"{metric_x}_{metric_y}_scatter.png"
    vis_path = os.path.join(vis_dir, filename)
    plt.savefig(vis_path, dpi=600, bbox_inches='tight')
    plt.close()


def plot_radar_chart(df, model='Mistral_7B', dataset='ag_news_full_seppmt'):
    """
    Radar chart for a single model/dataset across multiple metrics.
    We'll pick the sub_task with the highest 'accuracy' or just the first sub_task we find.
    """
    # Filter
    sub = df[(df['model'] == model) & (df['dataset'] == dataset)]
    # Suppose we want the sub_task with the highest accuracy:
    best_subtask = (sub[sub['metric'] == 'accuracy']
                    .sort_values('value', ascending=False)
                    .head(1)['sub_task']
                    .values)
    if len(best_subtask) == 0:
        print("No accuracy found, picking the first sub_task.")
        sub_task = sub['sub_task'].unique()[0]
    else:
        sub_task = best_subtask[0]

    # Now filter to just that sub_task
    sub = sub[sub['sub_task'] == sub_task]

    # We'll gather a few metrics to plot. Let's pick a set of metrics to display:
    metrics_of_interest = ['accuracy','rouge_rouge1','bertscore_f1','mean_f1','mean_precision','mean_recall']
    sub = sub[sub['metric'].isin(metrics_of_interest)]

    # If no data, skip
    if len(sub) == 0:
        print(f"No data found for {model}, {dataset}, {sub_task}. Radar chart skipped.")
        return

    # For radar, we need angles for each metric
    sub = sub.groupby('metric')['value'].mean().reset_index()  # in case duplicates
    sub = sub.sort_values('metric')  # just to keep consistent order
    labels = sub['metric'].values
    values = sub['value'].values

    # Close the polygon by repeating first value
    values = list(values) + [values[0]]

    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += [angles[0]]

    fig, ax = plt.subplots(subplot_kw={'polar': True}, figsize=(6,6))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axis per variable + add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Y-labels
    ax.set_rlabel_position(0)
    # We can set the radial limits if we want: ax.set_ylim(0, 1) for example
    # For auto-scaling, we can just skip it.

    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, alpha=0.3)

    title = f"Radar Chart: {model} - {dataset} - {sub_task}"
    plt.title(title, y=1.1)
    plt.tight_layout()
    vis_dir = os.path.join('result_plots', "radars")
    os.makedirs(vis_dir, exist_ok=True)
    filename = f"{model}_boxplot.png"
    vis_path = os.path.join(vis_dir, filename)
    plt.savefig(vis_path, dpi=600, bbox_inches='tight')
    plt.close()


def plot_radar_chart_multi(df, model='Mistral_7B', datasets=None, metrics_of_interest=None):
    """
    Plots a radar chart for a given model with separate lines for each (dataset, sub_task) combination.
    
    Each line corresponds to a unique combination of dataset and sub_task (e.g., mnli_full_seppt_icld_ssp).
    
    Parameters:
      - df: pandas DataFrame containing the metrics with columns ['model','dataset','sub_task','metric','value'].
      - model: the model name to filter on.
      - datasets: an optional list of dataset names to include. If None, all datasets for the model are used.
      - metrics_of_interest: list of metric names to plot. If None, a default list is used.
    """
    # Default list of metrics if not provided
    if metrics_of_interest is None:
        # Include accuracy if present; adjust as needed.
        metrics_of_interest = ['accuracy','rouge_rouge1','bertscore_f1','mean_f1','mean_precision','mean_recall']
    
    # Filter data for the specified model
    df_model = df[df['model'] == model]
    
    # If no specific datasets provided, use all unique datasets for this model
    if datasets is None:
        datasets = df_model['dataset'].unique()
    
    # Prepare a dictionary: key = "dataset_sub_task", value = list of metric values
    combination_metrics = {}
    
    for ds in datasets:
        df_ds = df_model[df_model['dataset'] == ds]
        sub_tasks = df_ds['sub_task'].unique()
        for sub_t in sub_tasks:
            df_combo = df_ds[(df_ds['sub_task'] == sub_t) & (df_ds['metric'].isin(metrics_of_interest))]
            # Group by metric in case of duplicates (take mean)
            grouped = df_combo.groupby('metric')['value'].mean()
            values = [grouped.get(m, np.nan) for m in metrics_of_interest]
            key = f"{ds}_{sub_t}"
            combination_metrics[key] = values
    
    N = len(metrics_of_interest)
    # Calculate angles for each axis.
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += [angles[0]]  # Close the polygon
    
    # Set up the polar plot.
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], metrics_of_interest)
    
    # Plot each combination as a separate line.
    for combo, vals in combination_metrics.items():
        # Skip if any metric is missing.
        if any(np.isnan(vals)):
            print(f"Skipping {combo} due to missing values.")
            continue
        vals_plot = list(vals) + [vals[0]]
        ax.plot(angles, vals_plot, linewidth=2, label=combo)
        ax.fill(angles, vals_plot, alpha=0.1)
    
    plt.title(f"Radar Chart for Model: {model}", size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    # Save the plot
    vis_dir = os.path.join('result_plots', "radars")
    os.makedirs(vis_dir, exist_ok=True)
    filename = f"{model}_radar_chart_separate.png"
    vis_path = os.path.join(vis_dir, filename)
    plt.savefig(vis_path, dpi=600, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Exploratory plots of the per-position scores saved by runner.py."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Folder that was passed to runner.py as --output_path (one sub-folder per model)."
    )
    parser.add_argument(
        "--radar_models",
        nargs="+",
        default=[],
        help="Model keys to draw radar charts for (off by default)."
    )
    args = parser.parse_args()

    # 1) Build the big DataFrame
    df = build_dataframe(args.results_dir)
    print("DataFrame shape:", df.shape)
    print(df.head(10))
    if df.empty:
        raise SystemExit(f"No metric scores found under {args.results_dir}.")
    metrics = set(df['metric'].unique())

    # 2) Plots are written to ./result_plots/
    # a) Grouped bar chart and heatmap (accuracy)
    if 'accuracy' in metrics:
        plot_grouped_bar(df, metric='accuracy')
        plot_heatmap(df, metric='accuracy')

    # b) Box plot (rouge_rouge1)
    if 'rouge_rouge1' in metrics:
        plot_boxplot(df, metric='rouge_rouge1')

    # c) Scatter plot (accuracy vs. rouge_rouge1)
    if {'accuracy', 'rouge_rouge1'} <= metrics:
        plot_scatter(df, metric_x='accuracy', metric_y='rouge_rouge1')

    # d) Radar chart per requested model, one line per (task file, demo position).
    #    A line is drawn only when that task file has every metric on the radar.
    for model in args.radar_models:
        plot_radar_chart_multi(df, model=model)

if __name__ == "__main__":
    main()
