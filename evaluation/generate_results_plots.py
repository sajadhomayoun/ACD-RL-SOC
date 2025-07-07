import os
import pandas as pd
import matplotlib.pyplot as plt
import json


def format_step_labels(steps):
    return [f"{int(s/1000)}K" for s in steps]


# Paths
OUTPUT_DIR = "evaluation/output-all-experiments"
PLOTS_DIR = "evaluation/output-all-experiments"
os.makedirs(PLOTS_DIR, exist_ok=True)

records = []

env_labels = {
    "flat": "Flat Reward",
    "standard": "Standard Reward",
    "high_risk": "High-Risk Reward"
}

# Walk through nested directories like output-all-experiments/output-50k/flat_env ...
# Note that this file is NOT being run as a step 'run_experiments.py',
# the input to this file is a directory with all experiments saved in 'output-Xk' where X is 50, 100, 200,... showing the number of steps
# the directory should be named 'output-all-experiments'
for subfolder in sorted(os.listdir(OUTPUT_DIR)):
    subpath = os.path.join(OUTPUT_DIR, subfolder)
    if not os.path.isdir(subpath) or not subfolder.startswith("output-"):
        continue

    # Extract timestep
    try:
        steps = int(subfolder.replace("output-", "").replace("k", "000"))
    except ValueError:
        continue

    for exp_name in os.listdir(subpath):
        exp_path = os.path.join(subpath, exp_name)
        summary_path = os.path.join(exp_path, "summary_rewards.csv")
        if not os.path.exists(summary_path):
            continue

        df = pd.read_csv(summary_path)
        df["env"] = exp_name.replace("_env", "")
        df["timesteps"] = steps
        df["experiment_id"] = f"{exp_name}_{steps}"
        records.append(df)

if not records:
    raise ValueError("No summary files found. Check directory structure and names.")

# Combine all summaries
combined = pd.concat(records)
steps_set = sorted(combined["timesteps"].unique())


# Learning Curve
plt.figure(figsize=(8, 5))
for env_name in combined["env"].unique():
    subset = combined[combined["env"] == env_name].sort_values("timesteps")
    label = env_labels.get(env_name, env_name.capitalize())
    plt.plot(subset["timesteps"], subset["mean_reward"], marker='o', label=label)

plt.xlabel("Training Steps")
plt.xticks(ticks=steps_set, labels=format_step_labels(steps_set))

plt.ylabel("Mean Episode Reward")
# plt.title("Learning Curve (Reward vs. Timesteps)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "learning_curve.pdf"))


# Reward Std Dev Line Plot
plt.figure(figsize=(8, 5))
for env_name in combined["env"].unique():
    subset = combined[combined["env"] == env_name].sort_values("timesteps")
    label = env_labels.get(env_name, env_name.capitalize())
    plt.plot(subset["timesteps"], subset["std_reward"], marker='o', label=label)

plt.xlabel("Training Steps")
plt.xticks(ticks=steps_set, labels=format_step_labels(steps_set))

plt.ylabel("Reward Standard Deviation")
# plt.title("Reward Variability vs. Training Steps")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "reward_variation_linechart.pdf"))


# FP / FN Curve
plt.figure(figsize=(8, 5))
for env_name in combined["env"].unique():
    subset = combined[combined["env"] == env_name].sort_values("timesteps")
    label_fp = env_labels.get(env_name, env_name.capitalize()) + " FP"
    label_fn = env_labels.get(env_name, env_name.capitalize()) + " FN"
    plt.plot(subset["timesteps"], subset["false_positives"], marker='s', linestyle='--', label=label_fp)
    plt.plot(subset["timesteps"], subset["false_negatives"], marker='^', linestyle='-', label=label_fn)

plt.xlabel("Training Steps")
plt.xticks(ticks=steps_set, labels=format_step_labels(steps_set))

plt.ylabel("Count")
# plt.title("False Positives and False Negatives vs. Timesteps")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fp_fn_curve.pdf"))


# Uncertain Region Action Distribution per step
steps_set = sorted(combined["timesteps"].unique())

for step in steps_set:
    plt.figure(figsize=(6, 4))
    width = 0.25
    x = [0, 1, 2]
    for idx, env in enumerate(env_labels.keys()):
        exp_path = os.path.join(OUTPUT_DIR, f"output-{int(step/1000)}k", f"{env}_env")
        eval_file = os.path.join(exp_path, "evaluation.json")
        if not os.path.exists(eval_file):
            continue
        with open(eval_file, "r") as f:
            eval_data = json.load(f)
        ua = eval_data.get("uncertain_actions", {})
        values = [ua.get(str(i), 0) for i in range(3)]
        plt.bar([i + idx * width for i in x], values, width=width, label=env_labels[env])
    plt.xticks([i + width for i in x], ["Ignore", "Alert", "Isolate"])
    plt.ylabel("Count")
    # plt.title(f"Uncertain Region Actions at {step} steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"uncertain_actions_{step}.pdf"))


# Uncertain Region Action Distribution vs. Timesteps
action_labels = ["Ignore", "Alert", "Isolate"]
env_map = {
    "flat": "Flat Reward",
    "standard": "Standard Reward",
    "high_risk": "High-Risk Reward"
}

# Prepare data
data = {env: {step: [0, 0, 0] for step in steps_set} for env in env_map.keys()}

for step in steps_set:
    for env in env_map.keys():
        exp_path = os.path.join(OUTPUT_DIR, f"output-{int(step/1000)}k", f"{env}_env")
        eval_file = os.path.join(exp_path, "evaluation.json")
        if not os.path.exists(eval_file):
            continue
        with open(eval_file, "r") as f:
            eval_data = json.load(f)
        ua = eval_data.get("uncertain_actions", {})
        data[env][step] = [ua.get(str(i), 0) for i in range(3)]

# Uncertain Region Action Usage Over Time
for i, action in enumerate(action_labels):
    plt.figure(figsize=(8, 5))
    for env in env_map.keys():
        y_values = [data[env][step][i] for step in steps_set]
        label = env_map[env]
        plt.plot(steps_set, y_values, marker='o', label=label)
    plt.xlabel("Training Steps")
    plt.xticks(ticks=steps_set, labels=format_step_labels(steps_set))

    plt.ylabel("Action Count")
    # plt.title(f"{action} in Uncertain Region vs. Training Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"uncertain_{action.lower()}_linechart.pdf"))


print("All plots saved in", PLOTS_DIR)
