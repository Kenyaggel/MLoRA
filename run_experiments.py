import argparse
import os
import json
import copy
import csv
import random
import time
from itertools import product
from datetime import datetime
import run
import run_moe

def run_experiments(datasets, models, param_changes=None, csv_filename="results.csv", temp_dir = "temp_configs"):
    """
    Runs experiments over each combination of dataset, model, and (optionally)
    hyperparameters, then writes metrics to a CSV file.

    :param datasets: list of dataset names, e.g. ["Movielens", "Amazon"]
    :param models: list of model names, e.g. ["autoint", "autoint_mlora", "wdl"]
    :param param_changes: dict of parameter changes or sweeps, e.g.
                         {"model.num_expart": [8,16], "training.learning_rate": [0.001, 0.0001]}
    :param csv_filename: name of the CSV file to write results into
    :param temp_dir: directory to write temporary config files into
    """
    # Create the folder if it does not exist.
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        print(f"Created temporary directory: {temp_dir}")

    if param_changes is not None and len(param_changes) > 0:
        param_keys = sorted(param_changes.keys())
    else:
        param_keys = []

    # CSV columns:
    header = ["dataset", "model", "config_path"]
    # For each parameter key (like "model.num_expart"), turn it into something like "model_num_expart"
    header += [k.replace('.', '_') for k in param_keys]
    header += ["avg_loss", "avg_auc", "domain_loss", "domain_auc", "w_auc"]

    # Open the CSV file in write mode
    print(f"Writing results to: {csv_filename}")
    with open(csv_filename, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # Write header row
        writer.writerow(header)

        # Outer loops over dataset and model
        for dataset in datasets:
            for model_name in models:
                # Build path to the base config
                config_path = f"config/{dataset}/{model_name}.json"

                if not os.path.isfile(config_path):
                    print(f"[Warning] Could not find config: {config_path}. Skipping.")
                    continue

                # Load the base config
                with open(config_path, "r") as f:
                    base_config = json.load(f)

                # If no param_changes specified, just run once with the base config
                if not param_keys:
                    print(f"\n=== Running: dataset={dataset}, model={model_name} ===")
                    if config['model']['num_experts'] == -1:
                        avg_loss, avg_auc, domain_loss, domain_auc, w_auc = run.main(config)
                    else:
                        avg_loss, avg_auc, domain_loss, domain_auc, w_auc = main(config)

                    # Write one row to the CSV
                    row_data = [dataset, model_name, config_path]
                    row_data += [None] * len(param_keys)
                    # No param keys here
                    row_data += [avg_loss, avg_auc, domain_loss, domain_auc, w_auc]
                    writer.writerow(row_data)
                    csv_file.flush()
                else:

                    # Prepare lists of values for each parameter key.
                    values_lists = []

                    for key in param_keys:
                        value = param_changes[key]
                        if not isinstance(value, list):
                            value = [value]
                        values_lists.append(value)

                    # Iterate over all combinations of parameter values.
                    for combo in product(*values_lists):
                        # Create a deep copy of the base config.
                        config = copy.deepcopy(base_config)
                        # Update the config based on the current combination.
                        for k, v in zip(param_keys, combo):
                            parts = k.split('.')
                            d = config
                            for part in parts[:-1]:
                                d = d[part]
                            d[parts[-1]] = v

                        # Create a unique temporary config filename.
                        combo_str = "_".join(f"{k.split('.')[-1]}_{v}" for k, v in zip(param_keys, combo))
                        temp_config_name = f"{dataset}_{model_name}_{combo_str}.json"
                        temp_config_path = os.path.join(temp_dir, temp_config_name)

                        # Write the updated config to the temporary file.
                        with open(temp_config_path, "w") as f:
                            json.dump(config, f, indent=2)

                        print(f"\n=== Running: dataset={dataset}, model={model_name}, params: {combo_str} ===")

                        if config['model']['num_experts'] == -1:
                            print("Run MLoRA original")
                            avg_loss, avg_auc, domain_loss, domain_auc, w_auc = run.main(config)
                        else:
                            avg_loss, avg_auc, domain_loss, domain_auc, w_auc = run_moe.main(config)

                        # Write the run's results into the CSV.
                        row_data = [dataset, model_name, temp_config_path] + list(combo)
                        row_data += [avg_loss, avg_auc, domain_loss, domain_auc, w_auc]
                        writer.writerow(row_data)
                        csv_file.flush()  # update file after each run

    print(f"\nAll runs complete! Results written to '{csv_filename}'.\n")


def create_experiment(datasets, models, param_changes):
    # include a timestamp for the CSV filename.
    suffix = datasets[0] if len(datasets) == 1 else "multi"
    if param_changes.get("dataset.domain_split_path") is not None and len(param_changes["dataset.domain_split_path"]) == 1:
        suffix += "_" + param_changes["dataset.domain_split_path"][0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return run_experiments(datasets, models, param_changes, csv_filename=f"result/results_{suffix}_{timestamp}.csv", temp_dir="config/temp_configs")


def main(jobid=None):
    datasets = ["Movielens"] #, "Amazon_6"]
    models = ["autoint", "wdl", "deepfm", "nfm", "pnn", "dcn", "xdeepfm", "fibinet"]
    mlora_models = [m + "_mlora" for m in models]
    import sys

    jobid_str = os.getenv('SLURM_ARRAY_TASK_ID')
    # Convert the jobid to an integer if it exists, otherwise default to 1.
    if jobid is None:
        jobid = int(jobid_str) if jobid_str is not None else 1
    print(f"Running job {jobid}")

    # Introduce a random sleep between 0 and 10 seconds, scaled by jobid
    sleep_time = random.uniform(0, 5) + (jobid % 3)  # Random delay with a slight job-based offset
    print(f"Sleeping for {sleep_time:.2f} seconds before starting...")
    time.sleep(sleep_time)

    if jobid == 1:
        param_changes = {
            "model.num_experts": [-1, 2],
            "dataset.domain_split_path": ["split_by_gender"] #, "split_by_age", "split_by_occupation"],
        }
        create_experiment(datasets, mlora_models, param_changes)
    if jobid == 2:
        param_changes = {
            "model.num_experts": [-1, 7],
            "dataset.domain_split_path": ["split_by_age"]
        }
        create_experiment(datasets, mlora_models, param_changes)

    if jobid == 3:
        param_changes = {
            "model.num_experts": [-1, 21],
            "dataset.domain_split_path": ["split_by_occupation"]
        }
        create_experiment(datasets, mlora_models, param_changes)

    if jobid == 4:
        param_changes = {
            "model.num_experts": [-1, 10],
            "dataset.domain_split_path": ["split_by_theme_10"]
        }
        datasets = ["Taobao_10"]
        create_experiment(datasets, mlora_models, param_changes)
    if jobid == 5:
        param_changes = {
            "model.num_experts": [4, 6, 8],
            "dataset.domain_split_path": ["split_by_gender"]
        }
        datasets = ["Movielens"]
        create_experiment(datasets, mlora_models, param_changes)

    if jobid == 6:
        param_changes = {
            "model.num_experts": [20],
            "dataset.domain_split_path": ["split_by_theme_20"]
        }
        datasets = ["Taobao_10"]
        create_experiment(datasets, mlora_models + models, param_changes)

    if jobid == 7:
        param_changes = {
            "model.num_experts": [30],
            "dataset.domain_split_path": ["split_by_theme_30"]
        }
        datasets = ["Taobao_10"]
        create_experiment(datasets, mlora_models + models, param_changes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobid", type=int, default=None, help="Job ID for SLURM array job")
    args = parser.parse_args()
    main(args.jobid)
