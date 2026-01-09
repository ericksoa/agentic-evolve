# Reinforcement Learning for Tree Packing

RL-based approach to learn optimal tree placement strategies.

## Quick Start (Lightning.ai)

### 1. Setup Environment
```bash
pip install stable-baselines3[extra] gymnasium tensorboard
```

### 2. Train PPO Agent
```bash
# Small n (fast, good for testing approach)
python python/rl/train_ppo.py --n-trees 5 --timesteps 500000 --device cuda

# Medium n
python python/rl/train_ppo.py --n-trees 10 --timesteps 1000000 --device cuda

# Large n (needs more timesteps)
python python/rl/train_ppo.py --n-trees 20 --timesteps 2000000 --device cuda
```

### 3. Monitor Training
```bash
tensorboard --logdir checkpoints/
```

## Environment Details

### State Space
- For each tree: (x, y, angle, placed_flag) - normalized
- Current bounding box side (normalized)
- Trees remaining (normalized)
- Dimension: `n_trees * 4 + 2`

### Action Space
- Continuous: (x, y, angle) in [-1, 1]
- Denormalized to actual coordinates and angles

### Rewards
- `-delta_bbox`: Negative change in bounding box (encourage compact packing)
- `-10`: Overlap penalty (invalid placement)
- `+1/(1+bbox)`: Completion bonus (inversely proportional to final bbox)

## Expected Results

With sufficient training:
- n=5: Should beat random (bbox ~2.0 vs ~7.0 random)
- n=10: Should learn reasonably compact placements
- n=20+: May need curriculum learning or transfer from smaller n

## Files

- `tree_packing_env.py` - Gymnasium environment
- `train_ppo.py` - PPO training script
- `evaluate_agent.py` - Evaluation utilities (TODO)

## Notes

- Training on GPU (L40S) is ~10x faster than CPU
- Use `--n-envs 8` or higher for better sample efficiency on GPU
- Monitor tensorboard for learning curves
