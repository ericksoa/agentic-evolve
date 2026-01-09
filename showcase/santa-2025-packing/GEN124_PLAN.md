# Gen124 Plan: ML-Based and Compute-Heavy Optimization

## Current State
- **Score**: 85.10 (Gen120)
- **Target**: ~69.02 (top leaderboard)
- **Gap**: 23.3% (16 points)
- **Gen121-123 Finding**: Incremental optimization cannot close this gap

## Lightning.ai Infrastructure

We'll use lightning.ai studios with L40S GPUs for all GPU-accelerated work.

**Available GPUs** (from [Lightning.ai pricing](https://lightning.ai/pricing/)):
| GPU | VRAM | Rate | Best For |
|-----|------|------|----------|
| T4 | 16GB | $0.68/hr | Light training, inference |
| L4 | 24GB | $0.70/hr | Medium training |
| A10G | 24GB | $1.80/hr | Training, larger models |
| **L40S** | **48GB** | ~$2-3/hr | **Heavy training, large batches** |

**Plan**: Use L40S for all training (48GB VRAM is excellent for RL/GNN)

## Research Summary

Based on recent literature (2024-2025):
- **RL for irregular packing** shows promise - trains agents to make sequential placement decisions
- **GNN-based methods** can model spatial relationships between shapes
- **Neural-driven heuristics** trained with CMA-ES outperform hand-crafted rules
- **Massive parallel SA** on GPU can achieve 100x speedup over CPU
- **Multiple-chain SA** explores solution space more thoroughly

Key papers:
- [Hybrid RL for 2D Irregular Packing](https://www.mdpi.com/2227-7390/11/2/327)
- [Learning-Based 2D Irregular Shape Packing](https://arxiv.org/html/2309.10329)
- [GPU-accelerated Parallel SA](https://arxiv.org/abs/2408.00018)
- [Neural-Driven Constructive Heuristic](https://www.mdpi.com/2079-9292/14/10/1956)

## Approach Categories

### Category A: Machine Learning (High Impact, High Effort)

#### A1: Reinforcement Learning for Sequential Packing
**Concept**: Train an RL agent to place trees one at a time, learning optimal placement policies.

**Implementation**:
```python
# MDP Formulation
State: Current packing (tree positions, angles, bounding box)
Action: (x, y, angle) for next tree placement
Reward: -Δbbox_side (negative increase in bounding box)
Terminal: All n trees placed

# Network Architecture
- Encoder: Process current tree positions with attention/transformer
- Decoder: Output continuous (x, y, angle) via policy network
- Value network: Estimate expected future reward
```

**Framework Options**:
- Stable-Baselines3 (PPO, SAC) - easiest to start
- RLlib (Ray) - scales to cloud
- Custom PyTorch - most flexible

**Expected Compute**:
- Training: 10-50 GPU-hours per n value
- Can train on smaller n, transfer to larger

**Potential Impact**: High (could learn non-obvious placement strategies)

#### A2: Graph Neural Network for Packing Refinement
**Concept**: Represent packing as a graph, use GNN to predict position adjustments.

**Implementation**:
```python
# Graph Structure
Nodes: Trees (features: x, y, angle, local density)
Edges: Connect nearby trees (distance < threshold)
Output: Position/angle deltas for each tree

# Architecture
- Message passing layers to aggregate neighbor info
- Global readout for bounding box prediction
- Train to minimize bbox while avoiding overlaps
```

**Expected Compute**: 5-20 GPU-hours
**Potential Impact**: Medium (refines existing solutions)

#### A3: Neural Placement Heuristic (Supervised)
**Concept**: Train a network to predict good positions given current packing state.

**Implementation**:
1. Generate training data: Many SA runs, save intermediate states + best next placements
2. Train network: state → (x, y, angle) for next tree
3. Use trained network to guide search (beam search / MCTS)

**Expected Compute**: 2-10 GPU-hours for training, then fast inference
**Potential Impact**: Medium (data quality dependent)

### Category B: Massive Compute (Medium Impact, Medium Effort)

#### B1: Cloud-Scale Random Restart SA
**Concept**: Run millions of SA restarts across cloud instances.

**Implementation**:
```bash
# AWS/GCP Spot Instances
- 100 c5.xlarge instances @ $0.03/hr spot = $3/hr
- Each instance runs 4 parallel SA chains
- 400 chains × 1000 restarts/hr = 400,000 restarts/hr
- Budget: $50 = 16 hours = 6.4M restarts

# Expected variance exploitation
- Gen121 Best-of-100: 85.39 → 85.10 gain from variance
- Best-of-6.4M could find significantly better configurations
```

**Infrastructure**:
- AWS Batch or GCP Cloud Run for auto-scaling
- S3/GCS for result aggregation
- Could use existing lightning.ai credits

**Expected Compute**: $50-200 for thorough search
**Potential Impact**: Medium (brute force variance exploitation)

#### B2: GPU-Accelerated Multi-Chain SA
**Concept**: Port SA to CUDA, run thousands of parallel chains on GPU.

**Implementation**:
```python
# CUDA Kernel Design
- Each thread: one SA chain
- Shared memory: tree positions (fast read/write)
- Global memory: best solutions per n

# Expected speedup
- CPU SA: ~1000 iterations/second
- GPU SA: ~100,000+ iterations/second (100x)
- Can explore more temperature schedules
```

**Framework Options**:
- CuPy (easiest)
- Numba CUDA
- Raw CUDA/C++

**Expected Compute**: 1-5 GPU-hours
**Potential Impact**: Medium-High (much deeper search)

#### B3: Population-Based Training (PBT)
**Concept**: Evolve hyperparameters of the solver alongside solutions.

**Implementation**:
- Population of 20-50 solver configurations
- Each has different: temperature schedule, placement order strategy, angle discretization
- Evolve configurations based on solution quality
- Combines evolutionary optimization with hyperparameter tuning

**Expected Compute**: 10-50 GPU-hours
**Potential Impact**: Medium (finds better solver configurations)

### Category C: Hybrid Approaches (Variable Impact)

#### C1: MCTS with Neural Value Function
**Concept**: Use Monte Carlo Tree Search for placement decisions, neural network for position evaluation.

**Implementation**:
- Each MCTS node: partial packing state
- Actions: discretized (x, y, angle) for next tree
- Value function: trained network predicts final bbox
- Rollout policy: fast greedy placement

**Expected Compute**: 5-20 GPU-hours (training + search)
**Potential Impact**: High (systematic exploration)

#### C2: Transfer Learning from Small n
**Concept**: Train on small n (faster), transfer knowledge to large n.

**Implementation**:
1. Train RL agent thoroughly on n=1-20
2. Use learned representations/policies for n>20
3. Fine-tune on larger n if needed

**Expected Compute**: 10-30 GPU-hours
**Potential Impact**: Medium (depends on transferability)

## Recommended Execution Order

### Phase 1: GPU Foundation (1-2 days, ~$15-30 on L40S)

**1.1 GPU SA Prototype (B2)**
- Port core SA loop to PyTorch/CuPy on L40S
- Run thousands of parallel SA chains
- Benchmark: expect 50-100x speedup over CPU

**1.2 GPU SA Full Run**
- Run 100K+ restarts per n value
- Compare with current best (85.10)
- This alone could find improvements through variance

*Note: Skipping cloud CPU SA since Gen121 showed lightning.ai CPU is 4x slower than local M3.*

### Phase 2: ML Foundation (3-5 days, ~$50 compute)

**2.1 RL Environment Setup (A1)**
```python
# Create gym-compatible environment
class TreePackingEnv(gym.Env):
    def __init__(self, n):
        self.n = n
        self.action_space = Box(low=[-2,-2,0], high=[2,2,360])
        self.observation_space = ...

    def step(self, action):
        # Place tree, return reward
        ...

    def reset(self):
        # Start new packing
        ...
```

**2.2 Train RL Agent on Small n**
- Start with n=5-10 (fast iteration)
- Use PPO or SAC from Stable-Baselines3
- Train until convergence, evaluate vs greedy baseline

**2.3 Evaluate and Scale**
- If RL beats baseline on small n, scale to larger n
- If not, pivot to other approaches

### Phase 3: Advanced ML (1-2 weeks, ~$100-200 compute)

**3.1 GNN Refinement (A2)**
- Implement if RL shows promise
- Train on current best solutions
- Use to post-process RL outputs

**3.2 MCTS with Neural Value (C1)**
- Implement if RL representations are useful
- Could combine with trained RL policy

### Phase 4: Full-Scale Optimization (Ongoing)

**4.1 Combine Best Approaches**
- ML-guided placement + GPU SA refinement
- Best-of-many with cloud compute
- Ensemble of different methods

## Cost Estimates (Lightning.ai)

| Approach | GPU | Hours | Est. Cost | Expected Improvement |
|----------|-----|-------|-----------|---------------------|
| GPU SA Prototype | L40S | 2-5 hrs | $5-15 | 0-3 points |
| RL Training (small n) | L40S | 10-20 hrs | $25-60 | 0-5 points |
| RL Training (all n) | L40S | 50-100 hrs | $125-300 | 0-10 points |
| GNN Refinement | L40S | 10-20 hrs | $25-60 | 0-5 points |
| MCTS + Neural | L40S | 20-30 hrs | $50-90 | 0-5 points |
| **Total Budget** | **L40S** | **~100-175 hrs** | **~$250-500** | **2-15 points** |

*Note: L40S at ~$2.50/hr on lightning.ai. Free tier includes some GPU hours.*

**Pro Plan Consideration**: $50/month gets 40 credits + 1 free 24/7 studio - good value if we need extended training.

## Implementation Files Needed

```
python/
├── rl/
│   ├── tree_packing_env.py     # Gym environment
│   ├── train_ppo.py            # PPO training script
│   ├── train_sac.py            # SAC training script
│   └── evaluate_agent.py       # Evaluation utilities
├── gpu/
│   ├── cupy_sa.py              # CuPy SA implementation
│   ├── cuda_kernels.py         # Custom CUDA kernels
│   └── benchmark_gpu.py        # GPU vs CPU benchmarks
├── cloud/
│   ├── aws_batch_config.json   # AWS Batch configuration
│   ├── submit_jobs.py          # Job submission script
│   └── aggregate_results.py    # Result aggregation
└── gnn/
    ├── packing_graph.py        # Graph construction
    ├── gnn_model.py            # GNN architecture
    └── train_gnn.py            # Training script
```

## Success Criteria

| Phase | Target | Success |
|-------|--------|---------|
| Phase 1 | Better than 85.10 | Score < 84 |
| Phase 2 | RL beats greedy on small n | >5% improvement |
| Phase 3 | Score improvement | Score < 82 |
| Phase 4 | Significant progress | Score < 78 |
| **Stretch** | Near top | Score < 72 |

## Risk Assessment

| Approach | Complexity | Time | Risk | Reward |
|----------|------------|------|------|--------|
| Cloud SA | Low | 1 day | Low | Low-Med |
| GPU SA | Medium | 2 days | Low | Medium |
| RL (small n) | High | 3-5 days | Medium | High |
| RL (full) | Very High | 1-2 weeks | High | Very High |
| GNN | High | 3-5 days | Medium | Medium |
| MCTS | High | 1 week | Medium | High |

## Recommended Starting Point

**Start with Phase 1.1 (Cloud SA)** because:
1. Uses existing infrastructure (Rust solver)
2. Low risk, could find improvements from variance alone
3. Results inform whether more compute helps

**Then proceed to Phase 2.1 (RL Environment)** because:
1. Most promising ML approach for this problem class
2. Well-documented in recent literature
3. Could learn non-obvious placement strategies

## Recovery Notes

If session restarts:
1. Read this file for context
2. Check which phase was in progress
3. Look for checkpoint files in `python/rl/checkpoints/`
4. Check cloud job status with `python cloud/check_jobs.py`

## Critical Constraints

- **Budget**: Keep total under $500 unless explicitly approved
- **Local compute**: Max 4 parallel processes locally
- **Cloud**: Use spot/preemptible instances to minimize cost
- **GPU**: T4 or V100 are cost-effective; avoid A100 unless needed

## Lightning.ai Setup

### Studio Configuration
```bash
# Create new studio with L40S GPU
# Settings:
#   - GPU: L40S (48GB)
#   - CPU: 8 cores recommended
#   - Storage: 50GB minimum
#   - Python: 3.10 or 3.11

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install stable-baselines3[extra] gymnasium
pip install torch-geometric  # For GNN
pip install cupy-cuda11x  # For GPU SA
pip install wandb  # For experiment tracking

# Clone repo and setup
git clone https://github.com/ericksoa/agentic-evolve.git
cd agentic-evolve/showcase/santa-2025-packing
```

### Quick GPU Test
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## Progress Log
- [x] Phase 1.1: GPU SA prototype - COMPLETE
  - Created `python/gpu/gpu_sa.py` with parallel SA chains
  - Added `initialize_from_best()` for refinement mode
  - Added final validation pass to filter invalid solutions
  - Supports both "refine" and "global" search modes
- [x] Phase 1.2: Local testing - COMPLETE
  - Tested on CPU (M3 Mac) - works but slow (~60-80 iter/s)
  - Refinement mode: No improvements found (current best is already well-optimized)
  - Global search: Works but needs GPU for meaningful results
  - Ready for lightning.ai testing with L40S GPU
- [x] Phase 2.1: RL environment setup - COMPLETE
  - Created `python/rl/tree_packing_env.py` (Gymnasium compatible)
  - State: tree positions/angles + bbox + remaining
  - Action: continuous (x, y, angle)
  - Rewards: -delta_bbox, overlap penalty, completion bonus
- [x] Phase 2.2: Training script - COMPLETE
  - Created `python/rl/train_ppo.py` with Stable-Baselines3
  - Supports multi-env training, checkpointing, tensorboard
  - Ready for lightning.ai GPU training
- [ ] Phase 2.3: Train and evaluate on small n (L40S)
- [ ] Phase 3+: Advanced approaches (GNN, MCTS)
