#!/usr/bin/env python3
"""
Train a PPO agent for tree packing using Stable-Baselines3.

Usage (local):
    python python/rl/train_ppo.py --n-trees 5 --timesteps 100000

Usage (lightning.ai with GPU):
    python python/rl/train_ppo.py --n-trees 5 --timesteps 1000000 --device cuda
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np

# Import environment
from tree_packing_env import TreePackingEnv

# Check for stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed. Install with:")
    print("  pip install stable-baselines3[extra]")


def evaluate_agent(env, model, n_episodes: int = 10):
    """Evaluate trained agent."""
    bboxes = []
    successes = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Check if all trees were placed (success)
        success = info['trees_placed'] == env.n_trees
        successes.append(success)
        if success:
            bboxes.append(info['current_bbox'])

    success_rate = np.mean(successes)
    avg_bbox = np.mean(bboxes) if bboxes else float('inf')

    return {
        'success_rate': success_rate,
        'avg_bbox': avg_bbox,
        'min_bbox': min(bboxes) if bboxes else float('inf'),
        'n_episodes': n_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent for tree packing')
    parser.add_argument('--n-trees', type=int, default=5, help='Number of trees')
    parser.add_argument('--timesteps', type=int, default=100000, help='Training timesteps')
    parser.add_argument('--n-envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu, cuda, auto)')
    parser.add_argument('--output-dir', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--eval-freq', type=int, default=10000, help='Evaluation frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    if not SB3_AVAILABLE:
        print("Error: stable-baselines3 required for training")
        return

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"ppo_n{args.n_trees}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training PPO for n={args.n_trees} trees")
    print(f"Timesteps: {args.timesteps}")
    print(f"Output: {output_dir}")
    print()

    # Create vectorized environment
    env = make_vec_env(
        lambda: TreePackingEnv(n_trees=args.n_trees),
        n_envs=args.n_envs,
        seed=args.seed
    )

    # Create evaluation environment
    eval_env = Monitor(TreePackingEnv(n_trees=args.n_trees))

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.eval_freq // args.n_envs, 1),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True
    )

    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=args.device,
        seed=args.seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        tensorboard_log=str(output_dir / "tensorboard")
    )

    print(f"Model device: {model.device}")
    print()

    # Train
    print("Starting training...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    # Save final model
    model.save(str(output_dir / "final_model"))
    print(f"\nFinal model saved to {output_dir / 'final_model'}")

    # Final evaluation
    print("\nFinal evaluation...")
    results = evaluate_agent(eval_env, model, n_episodes=100)

    print(f"Success rate: {results['success_rate']*100:.1f}%")
    print(f"Avg bbox: {results['avg_bbox']:.4f}")
    print(f"Min bbox: {results['min_bbox']:.4f}")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            'config': vars(args),
            'results': results,
        }, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
