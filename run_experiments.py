import os
import json
import shutil
import subprocess
import time
from datetime import datetime

PARAMS_DIR = "agents/params"
OUTPUT_DIR = "evaluation/output"

def train_all():
    for param_file in sorted(os.listdir(PARAMS_DIR)):
        if not param_file.endswith(".json"):
            continue

        with open(os.path.join(PARAMS_DIR, param_file)) as f:
            params = json.load(f)
        exp_id = params["experiment_id"]
        exp_dir = os.path.join(OUTPUT_DIR, exp_id)
        os.makedirs(exp_dir, exist_ok=True)

        # Save current params
        with open("current_params.json", "w") as f:
            json.dump(params, f, indent=2)

        print(f"\n Training experiment: {exp_id}")
        start = time.time()
        start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Train PPO
        subprocess.run(["python", "agents/train_ppo.py"])
        if not os.path.exists("ppo_model.zip"):
            print("PPO model not saved! Skipping this experiment.")
            continue

        # Move files
        for fname in ["ppo_model.zip", "current_params.json"]:
            if os.path.exists(fname):
                shutil.move(fname, os.path.join(exp_dir, fname))

        # Log metadata
        with open(os.path.join(exp_dir, "meta.txt"), "w") as f:
            f.write(f"Experiment ID: {exp_id}\n")
            f.write(f"Started: {start_str}\n")
            f.write(f"Duration: {time.time() - start:.2f} seconds\n")

        print(f"Training completed for {exp_id}")

def evaluate_all():
    for exp_id in sorted(os.listdir(OUTPUT_DIR)):
        exp_path = os.path.join(OUTPUT_DIR, exp_id)
        if not os.path.isdir(exp_path) or exp_id.startswith("."):
            continue

        summary_path = os.path.join(exp_path, "summary_rewards.csv")
        if os.path.exists(summary_path):
            print(f"Skipping {exp_id} (already evaluated)")
            continue

        print(f"\n Evaluating experiment: {exp_id}")
        try:
            subprocess.run(
                ["python", "evaluation/soc_eval.py", exp_id],
                check=True,
                capture_output=False
            )
        except subprocess.CalledProcessError as e:
            print(f"Evaluation failed for {exp_id}")


if __name__ == "__main__":
    train_all()
    evaluate_all()
