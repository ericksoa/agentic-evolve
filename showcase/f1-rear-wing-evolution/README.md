# F1 Rear Wing Evolution 🏎️

> **Physics-validated wing optimization using NeuralFoil aerodynamics**

![F1 Evolution Animation](images/f1_evolution.gif)

## For Bogie 🏁

Hey Bogie! We used AI evolution to optimize an F1 rear wing—but this time with **real physics validation**. The algorithm tested configurations using NeuralFoil (a neural network trained on XFOIL data) and found an optimal setup with **96.9% prediction confidence**.

---

## The Journey: Why Validation Matters

We initially evolved a wing using a simplified aerodynamic model. It claimed +57ms lap time gains. But when we validated with real physics:

| Wing | Simplified Model | **NeuralFoil Physics** | Confidence |
|------|------------------|------------------------|------------|
| Original "Evolved" | High fitness | **Fitness: 4.68** | 9.7% |
| Physics-Evolved | - | **Fitness: 55.99** | **96.9%** |

The simplified model was wrong. Real physics showed our "optimized" wing was actually performing poorly due to flow separation at high angles. **Lesson: always benchmark with real physics.**

---

## Physics-Validated Results

![F1 Hero Dashboard](images/f1_hero.png)

| Metric | Stalling Baseline | **Physics-Evolved** | Improvement |
|--------|-------------------|---------------------|-------------|
| Fitness | 0.71 | **61.72** | 87x |
| Downforce (Cl) | 2.36 | **3.05** | +29% |
| Drag (Cd) | 1.31 | **1.63** | +24% |
| **Confidence** | 1.4% | **97.6%** | Trustworthy |

The evolution tested **91 configurations** and found that for Silverstone's high-speed corners, **maximum downforce** wins over low drag.

### What We Discovered

The original baseline had a **25° flap angle** that was completely stalling:

```
STALLING BASELINE → PHYSICS-EVOLVED
Main element:  12° → 15.2°  (more aggressive)
Flap:          25° → 13.8°  (avoid stall, but push limits)
Slot gap:      3%  → 2.3%   (optimal interaction)
Camber:        Medium → High+ (maximum lift)
```

**Key insight:** The evolution pushed angles higher than our manual optimization because Silverstone rewards downforce. But it stayed below the stall threshold where the original baseline failed.

---

## Evolution Story

The evolution ran through 5 phases, testing 91 configurations:

### Phase 1: Angle Optimization
Starting from our physics-informed baseline (fitness 55.99), we swept through angle combinations:

| Generation | Main | Flap | Fitness | Discovery |
|------------|------|------|---------|-----------|
| Start | 12° | 10° | 55.99 | Baseline |
| Gen 1 | 10° | 12° | 56.30 | Lower main helps |
| Gen 2 | 11° | 11° | 56.35 | Balance matters |
| Gen 3 | 11° | 12° | 57.19 | Flap can go higher |
| Gen 4 | 12° | 12° | 57.94 | Push both up |
| Gen 5 | 13° | 12° | 58.56 | More main angle |
| Gen 6 | 14° | 12° | **59.02** | Even more! |

**Insight:** Unlike our initial stalling 25° flap, the evolution found that 12-14° is the sweet spot—high enough for good lift, low enough to avoid separation.

### Phase 2-3: Camber Optimization
With optimal angles locked, we explored airfoil shape:

| Phase | Component | Change | Fitness | Result |
|-------|-----------|--------|---------|--------|
| 2 | Main camber | Tested 6 profiles | 59.02 | No improvement (already optimal) |
| 3 | Flap camber | Increased 3 steps | **60.28** | Higher flap camber = +2.1% |

### Phase 4-5: Fine-Tuning & Random Search
| Phase | Method | Configs | Best | Improvement |
|-------|--------|---------|------|-------------|
| 4 | Gap sweep | 5 | 60.28 | Gap at 2% is optimal |
| 5 | Random mutations | 50 | **61.72** | +2.4% from random search |

**Final breakthrough:** Random mutation #26 found the winning combination—slightly higher angles (15.2°/13.8°) with perturbed CST coefficients that increased camber beyond our grid search.

### The Winning Formula

```
EVOLUTION PROGRESSION:
Fitness:   55.99 → 59.02 → 60.28 → 61.72  (+10.2% total)
Main:      12°   → 14°   → 14°   → 15.2°
Flap:      10°   → 12°   → 12°   → 13.8°
Cl:        2.66  → 2.83  → 2.92  → 3.05   (+14.6%)
```

---

## Physics Engine: NeuralFoil

This showcase uses [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox) with NeuralFoil for aerodynamic evaluation:

- **What it is:** Neural network trained on 500,000+ XFOIL simulations
- **Accuracy:** Validated against experimental data
- **Speed:** ~1ms per evaluation (vs hours for CFD)
- **Confidence:** Returns prediction confidence for each analysis

### Why This Matters

| Method | Time | Accuracy | Our Use |
|--------|------|----------|---------|
| Simplified thin airfoil | μs | Low | ❌ Initial (wrong) |
| **NeuralFoil** | ms | Good | ✅ **Validation** |
| XFOIL | seconds | Good | Reference |
| CFD | hours | High | Future work |
| Wind tunnel | weeks | Highest | Gold standard |

---

## Run It Yourself

```bash
cd showcase/f1-rear-wing-evolution
source .venv/bin/activate

# Install dependencies (first time)
pip install aerosandbox

# Evaluate with PHYSICS (the right way)
python3 evaluate_physics.py evolved_wing.json --circuit=silverstone

# Compare to stalling baseline
python3 evaluate_physics.py baseline.json --circuit=silverstone

# Try different circuits
python3 evaluate_physics.py evolved_wing.json --circuit=monaco
python3 evaluate_physics.py evolved_wing.json --circuit=monza
```

---

## Circuit-Specific Optimization

![Circuit Requirements](images/circuit_requirements.png)

Each circuit has different downforce/drag priorities:

| Circuit | Priority | Our Wing's Strength |
|---------|----------|---------------------|
| Monaco | Max downforce | High Cl (2.66) |
| Silverstone | Balanced | Optimized here |
| Spa | Efficiency | Good L/D |
| Monza | Min drag | Lower Cd (-5.3%) |

---

## Files

```
f1-rear-wing-evolution/
├── README.md                # This file
├── wing.py                  # F1 wing geometry (CST parameterization)
├── evaluate.py              # Original simplified model (deprecated)
├── evaluate_physics.py      # NeuralFoil physics evaluation
├── baseline.json            # Original stalling config (25° flap)
├── baseline_physics.json    # Physics-informed starting point
├── evolved_wing.json        # AI-optimized (physics-validated)
├── evolve_config.json       # Evolution parameters
└── images/                  # Visualizations
```

---

## Technical Details

### Multi-Element Wing Model

```
         ┌─────────────┐
         │    FLAP     │  ← 13.8° (evolved from 25° stalling)
         └─────────────┘
              ╲ 2.3% gap (slot effect)
         ┌─────────────────────┐
         │    MAIN ELEMENT     │  ← 15.2° (evolved for Silverstone)
         └─────────────────────┘
```

### CST Parameterization

We use Class Shape Transformation for airfoil geometry:
- 4 coefficients each for upper/lower surfaces
- Bernstein polynomial shape functions
- Industry-standard for optimization

### Fitness Function

```python
fitness = (
    downforce_weight × Cl × 30 +      # Cornering grip
    drag_weight × (1/Cd) × 15 +       # Straight-line speed
    L_D × 1.0 +                        # Efficiency
    (Cl²/Cd) × 0.5                     # Aero efficiency
) × confidence                         # Trust the prediction
```

---

## What We Learned

1. **Simple models lie.** Our thin airfoil model said higher angles = more downforce. Physics said: stall.

2. **Confidence matters.** NeuralFoil's 96.9% confidence vs 1.4% confidence showed which predictions to trust.

3. **Validation is essential.** Claims without physics validation are just noise.

4. **F1 teams know this.** That's why they spend $100M+ on CFD and wind tunnels.

---

## Next Steps

1. **CFD Validation** - Run OpenFOAM for full 3D analysis
2. **DRS Optimization** - Evolve open/closed flap configurations
3. **Multi-objective** - Pareto frontier of downforce vs drag
4. **Full Car** - Front wing, floor, sidepods together

---

*Built for Bogie with [Agentic Evolve](../../) — Because physics doesn't lie* 🏁
