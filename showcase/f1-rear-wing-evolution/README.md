# F1 Rear Wing Evolution 🏎️

> **AI evolution finds optimal rear wing configuration through 2,880 parameter combinations**

![F1 Evolution Animation](images/f1_evolution.gif)

## For Bogie 🏁

Hey Bogie! We used AI evolution to optimize an F1 rear wing. The algorithm tested **2,880 configurations** and found what our model predicts is the optimal balance of downforce and drag.

⚠️ **Important caveat:** This uses a simplified aerodynamic model based on thin airfoil theory, not CFD or wind tunnel data. The numbers show *relative improvement within our model*, not validated real-world performance. Real F1 teams spend $100M+ on CFD and wind tunnels for a reason!

---

## The Results (Model Predictions)

![F1 Hero Dashboard](images/f1_hero.png)

| Metric | Baseline | **AI Evolved** | Delta |
|--------|----------|----------------|-------|
| Lap Time | Reference | **-57ms** | Model estimate |
| Downforce (Cl) | 3.00 | **3.00** | Maintained |
| Drag (Cd) | 1.64 | **1.62** | -1.2% |
| Efficiency (L/D) | 1.83 | **1.86** | +1.6% |

*Note: Cl/Cd values are from our simplified model. Real F1 rear wings typically have Cl ~1.5-2.5 and Cd ~0.3-0.7.*

### What Changed?

The AI discovered an **unconventional low-flap-angle configuration**:

```
BASELINE → EVOLVED
Main element:  12° → 16°  (more aggressive)
Flap:          25° → 12°  (surprisingly flat!)
Slot gap:      3.0% → 5.0% (wider for flow attachment)
```

**The insight:** Instead of a steep flap (traditional approach), the AI found that a flatter flap with a wider slot maintains downforce while reducing drag. Whether this holds up in real physics would need CFD validation—but it's an interesting direction to explore.

---

## Circuit-Specific Optimization

![Circuit Requirements](images/circuit_requirements.png)

| Circuit | Model Prediction | Setup Focus |
|---------|------------------|-------------|
| Monaco | Lower drag | Max downforce for hairpins |
| Silverstone | -1.2% Cd | Balance for Maggots-Becketts |
| Spa | Better L/D | Efficiency for Kemmel straight |
| Monza | Min drag config | Low drag for Temple of Speed |

*Lap time gains would require CFD/wind tunnel validation.*

---

## How It Works

### F1 Rear Wing Anatomy

```
         ┌─────────────┐
         │    FLAP     │  ← Adjustable (DRS)
         └─────────────┘
              ╲ slot gap
         ┌─────────────────────┐
         │    MAIN ELEMENT     │  ← Fixed profile
         └─────────────────────┘
```

**Multi-element wings work because:**
1. Slot between elements energizes boundary layer
2. Each element can operate near stall point
3. Combined Cl exceeds single-element maximum

### What We Optimized

- **CST Coefficients** - Shape of upper/lower surfaces (8 parameters)
- **Angles of Attack** - Main element and flap incidence (2 parameters)
- **Slot Gap** - Space between elements (1 parameter)

### Fitness Function

```python
fitness = (
    downforce_weight × Cl × 40 +      # Grip in corners
    drag_weight × (1/Cd) × 20 +       # Speed on straights
    efficiency × 5 +                   # Overall balance
    lap_time_bonus × 10               # Bottom line metric
)
```

---

## F1 Aero 101

### Why Rear Wings Matter

- **30%** of total car drag comes from rear wing
- **500kg+** of downforce at 300 km/h
- **0.1s** lap time per 10 points of downforce

### The Trade-off

```
MORE DOWNFORCE → Faster corners, slower straights
LESS DRAG      → Faster straights, slower corners
```

Teams spend **$100M+** annually on CFD and wind tunnels to find the perfect balance. This demo shows how AI can quickly explore a parameter space—but would need real validation before making performance claims.

---

## Run It Yourself

```bash
cd showcase/f1-rear-wing-evolution
source .venv/bin/activate

# Evaluate baseline
python3 evaluate.py baseline.json --circuit=silverstone

# Evaluate evolved wing
python3 evaluate.py evolved_wing.json --circuit=silverstone

# Try different circuits
python3 evaluate.py evolved_wing.json --circuit=monaco
python3 evaluate.py evolved_wing.json --circuit=monza
```

---

## The Tech Stack

- **CST Parameterization** - Industry-standard airfoil shape representation
- **Multi-element Aero Model** - Simplified thin airfoil theory (not CFD-validated)
- **Circuit Profiles** - Downforce/drag weightings per track
- **Lap Time Model** - Simplified corner/straight speed estimation

---

## Files

```
f1-rear-wing-evolution/
├── README.md              # This file
├── wing.py               # F1 wing geometry
├── evaluate.py           # Aerodynamic evaluation
├── baseline.json         # Starting configuration
├── evolved_wing.json     # AI-optimized wing
├── evolve_config.json    # Evolution parameters
└── images/               # Visualizations
    ├── f1_hero.png
    ├── f1_evolution.gif
    ├── f1_wing_configs.png
    └── circuit_requirements.png
```

---

## What Real Teams Do

| Method | Time | Cost | Our Approach |
|--------|------|------|--------------|
| Wind Tunnel | Weeks | $$$$ | ✗ |
| CFD Simulation | Days | $$$ | ✗ |
| AI Evolution | Minutes | $ | ✓ |

*Note: Real F1 teams use all methods. This showcase demonstrates the potential of AI-guided optimization.*

---

## How to Actually Validate This

To make real performance claims, you'd need:

1. **XFOIL** - Panel method for 2D airfoil analysis (free, quick)
2. **OpenFOAM CFD** - Full 3D simulation with turbulence modeling
3. **Wind Tunnel** - Scale model testing (gold standard)
4. **Track Testing** - Real car, real driver, real data

This showcase demonstrates **the optimization technique**, not validated aerodynamics. The interesting part is how AI can efficiently search a large parameter space—the physics model would need to be upgraded for real predictions.

## Next Steps

1. **Validate with XFOIL** - Get real 2D lift/drag coefficients
2. **CFD in OpenFOAM** - 3D simulation with proper boundary conditions
3. **DRS Optimization** - Evolve for open/closed configurations
4. **Full Car** - Optimize front wing, floor, and sidepods together

---

*Built for Bogie with [Agentic Evolve](../../) — Because finding optimal parameters matters* 🏁
