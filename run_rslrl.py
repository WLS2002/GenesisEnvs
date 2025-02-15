import argparse
import os
import pickle
import shutil

from env import *
from adapter.rslrl_adapter import RslrlAdapter
from rsl_rl.runners import OnPolicyRunner
import wandb
import genesis as gs


gs.init(backend=gs.gpu, precision="32", logging_level="error")

task_to_class = {
    'GraspFixedBlock': GraspFixedBlockEnv,
    'GraspFixedRod': GraspFixedRodEnv,
    'GraspRandomBlock': GraspRandomBlockEnv,
    'GraspRandomRod': GraspRandomRodEnv,
    'WaterFranka': WaterFrankaEnv,
    'ShadowHandBase': ShadowHandBaseEnv,
}

def create_environment(task_name):
    if task_name in task_to_class:
        return task_to_class[task_name]  
    else:
        raise ValueError(f"Task '{task_name}' is not recognized.")

def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.004,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1e-3,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "tanh",
            "actor_hidden_dims": [64, 64],
            "critic_hidden_dims": [64, 64],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 100,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=8192)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("-t", "--task", type=str, default="GraspFixedBlock", help="Task to train on")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="device: cpu or cuda:x or mps for macos")
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    wandb.init(project='genesis', name=args.exp_name, dir=log_dir, mode='offline')

    env = create_environment(args.task)(vis=args.vis, device=args.device, num_envs=args.num_envs)
    # key
    rslrl_adapter = RslrlAdapter(env, max_episode_length=999999999)
    runner = OnPolicyRunner(rslrl_adapter, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(
        num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
    )


if __name__ == "__main__":
    main()

"""
# training
python examples/drone/hover_train.py
"""

