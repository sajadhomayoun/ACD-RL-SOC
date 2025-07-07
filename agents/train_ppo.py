import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import both environments
from environments.soc_env import SocAlertEnv
from environments.soc_env import SocFlatEnv

def load_params():
    with open("current_params.json", "r") as f:
        return json.load(f)

def make_env(params):
    def _init():
        env_type = params.get("environment", "shaped")
        if env_type == "flat":
            return SocFlatEnv(params=params)
        else:
            return SocAlertEnv(params=params)
    return _init

if __name__ == "__main__":
    params = load_params()

    total_timesteps = params.get("total_timesteps", 10000) # default is 10K
    print(f"Training PPO with {total_timesteps} timesteps using environment: {params.get('environment', 'shaped')}")

    env = SubprocVecEnv([make_env(params) for _ in range(4)])
    model = PPO(
        "MlpPolicy",
        env,
        seed=42,
        verbose=1,
        learning_rate=1e-4,
        gamma=0.99,
        batch_size=64,
        n_steps=2048,
        ent_coef=0.02,
        tensorboard_log=f"./logs/tb_logs/ppo/{params['experiment_id']}",
    )

    model.learn(total_timesteps=total_timesteps, log_interval=10)
    model.save("ppo_model")
    print("PPO training complete and model saved.")
