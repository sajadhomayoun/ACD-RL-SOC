import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.soc_env import SocAlertEnv


def format_label(label):
    value = int(label)
    if value >= 1000:
        return f"{value // 1000}K"
    return str(value)


def evaluate_agent(model, env, n_episodes, agent_name=""):
    rewards, episode_data = [], []
    fn, fp = 0, 0
    actions = {0: 0, 1: 0, 2: 0}
    confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    total_samples = 0

    for i in range(n_episodes):
        obs, _ = env.reset()
        done, total = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs_array = obs[0] if len(obs.shape) > 1 else obs
            threat_score, confidence, criticality = obs_array[0], obs_array[1], obs_array[2]

            # Label: High-risk = True if threat * criticality > 0.6
            label = 1 if threat_score * criticality > 0.6 else 0

            if 0.3 <= confidence <= 0.7:
                actions[int(action)] += 1

            # Pseudo-confusion matrix
            if label == 1 and action == 0:
                confusion["FN"] += 1
            elif label == 0 and action == 2:
                confusion["FP"] += 1
            elif label == 1 and action in [1, 2]:
                confusion["TP"] += 1
            elif label == 0 and action in [0, 1]:
                confusion["TN"] += 1

            if action == 0 and label == 1:
                fn += 1
            if action == 2 and label == 0 and confidence < 0.5:
                fp += 1

            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            total_samples += 1
            done = terminated or truncated

        rewards.append(total)
        episode_data.append({"episode": i+1, "reward": total})

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "min_reward": np.min(rewards),
        "false_negatives": fn,
        "false_positives": fp,
        "total_samples": total_samples,
        "uncertain_actions": actions,
        "confusion_matrix": confusion,
        "rewards": rewards,
        "episode_data": episode_data
    }

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj


def plot_per_experiment(exp_dir, results):
    def save(name): 
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, name))
        plt.close()

    # Rewards over episodes
    plt.figure(figsize=(8, 4))
    plt.plot(results["rewards"])
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    # plt.title("Reward over Episodes")
    save("reward_over_episodes.pdf")

    # Action distribution (uncertain)
    plt.figure(figsize=(6, 4))
    labels = ["Ignore", "Alert", "Isolate"]
    values = [results["uncertain_actions"][i] for i in range(3)]
    plt.bar(labels, values)
    plt.ylabel("Count")
    # plt.title("Uncertain Region Action Distribution")
    save("uncertain_action_distribution.pdf")

    # Pseudo-confusion matrix
    cm = results["confusion_matrix"]
    plt.figure(figsize=(6, 4))
    plt.bar(cm.keys(), cm.values())
    # plt.title("Confusion Matrix (based on risk threshold)")
    save("confusion_matrix.pdf")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python soc_eval.py <experiment_id>")
        sys.exit(1)

    experiment_id = sys.argv[1]
    exp_dir = os.path.join("evaluation/output", experiment_id)
    with open(os.path.join(exp_dir, "current_params.json")) as f:
        params = json.load(f)

    env = SocAlertEnv(params)
    model = PPO.load(os.path.join(exp_dir, "ppo_model"))

    results = evaluate_agent(model, env, n_episodes=100, agent_name="PPO")

    # Save logs
    pd.DataFrame(results["episode_data"]).to_csv(os.path.join(exp_dir, "ppo_episode_data.csv"), index=False)

    pd.DataFrame({
        "agent": ["PPO"],
        "mean_reward": [results["mean_reward"]],
        "std_reward": [results["std_reward"]],
        "min_reward": [results["min_reward"]],
        "false_negatives": [results["false_negatives"]],
        "false_positives": [results["false_positives"]],
        "total_samples": [results["total_samples"]]
    }).to_csv(os.path.join(exp_dir, "summary_rewards.csv"), index=False)


    pd.DataFrame([results["confusion_matrix"]]).to_csv(os.path.join(exp_dir, "confusion_matrix.csv"), index=False)
    plot_per_experiment(exp_dir, results)
    with open(os.path.join(exp_dir, "evaluation.json"), "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"Evaluation complete for {experiment_id}. Plots and logs saved.")
