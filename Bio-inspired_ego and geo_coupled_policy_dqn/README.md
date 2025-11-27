# Coupled Geocentric–Egocentric Deep Q-Network (Coupled DQN)

This project implements a **single unified Deep Q-Network (DQN)** that fuses:

- **Geocentric navigation** (global direction to the goal)
- **Egocentric perception** (local 3×3 obstacle awareness)

into **one end-to-end neural policy** for maze navigation.  
The agent learns to navigate a complex 15×15 maze using **both global goal direction and local obstacle sensing**.

---

## 🧠 Core Concept

Unlike the hierarchical coupled policy (separate global & local controllers), this model:

- Uses **one neural network**
- Learns directly from:
  - Global goal vector `(dx, dy)`
  - Local occupancy grid `(3×3)`
- Outputs **absolute movement actions**:
  - `0 = Up`
  - `1 = Right`
  - `2 = Down`
  - `3 = Left`

This represents a **fully end-to-end biologically inspired navigation policy**.

---

## 🗺 Environment Description

- **Grid Size:** 15 × 15
- **Start Position:** `(14, 0)`
- **Goal Position:** `(0, 14)`
- **Cell Values:**
  - `0` → Free
  - `1` → Wall
- **Movement:** 4-connected grid
- **Collision Handling:** Agent stays in place with penalty

---

## 🧩 State Representation (11-D Vector)

Each state is composed of:

| Component | Size | Description |
|-----------|------|-------------|
| Geocentric vector | 2 | `(goal_x - x, goal_y - y)` |
| Local patch | 9 | Flattened 3×3 neighborhood |
| **Total** | **11** | Input to DQN |

---

## 🎯 Action Space

| Action ID | Movement |
|-----------|----------|
| 0 | Up |
| 1 | Right |
| 2 | Down |
| 3 | Left |

---

## 🏆 Reward Function

The reward is **shaped using distance-to-goal**:

| Event | Reward |
|-------|--------|
| Reach goal | `+200` |
| Collision | `-20` |
| Step penalty | `-0.01` |
| Distance improvement | `+(d_prev − d_new)` |

This encourages:
- Shortest paths
- Smooth progress
- Strong obstacle avoidance

---

## 🧠 Deep Q-Network Architecture

```

Input: 11
→ Linear(128)
→ ReLU
→ Linear(128)
→ ReLU
→ Linear(4)
→ Q-values for 4 actions

````

- **Optimizer:** Adam
- **Loss:** Mean Squared Error (MSE)
- **Discount Factor (`γ`):** 0.99
- **Target Network:** Yes
- **Replay Buffer:** Yes (50,000 capacity)

---

## ⚙️ Training Hyperparameters

| Parameter | Value |
|----------|-------|
| Episodes | 2000 |
| Max steps per episode | 300 |
| Batch size | 128 |
| Learning rate | 1e-4 |
| Epsilon start | 1.0 |
| Epsilon min | 0.05 |
| Epsilon decay | 0.995 |
| Target update | Every 10 episodes |

---

## ▶️ How to Run

### 1️⃣ Install Requirements
```bash
pip install numpy torch matplotlib
````

### 2️⃣ Train or Load Model

```bash
python coupled_dqn.py
```

If a trained model already exists:

```
Coupled_DQN_Output/coupled_dqn.pth
```

it will be loaded automatically.

---

## 📂 Generated Output Files

All outputs are saved in:

```
Coupled_DQN_Output/
```

| File                            | Description              |
| ------------------------------- | ------------------------ |
| `coupled_dqn.pth`               | Trained model            |
| `coupled_learning_curve.png`    | Training reward curve    |
| `coupled_final_path_eval_*.png` | Final agent trajectories |
| `coupled_policy_map.png`        | Argmax policy arrows     |

---

## 🧪 Evaluation Phase

After training:

* The model is evaluated for **5 independent trials**
* For each trial:

  * Final path is plotted
  * Total reward and steps are printed
* A **global policy map** is generated showing the best action at every free cell

---

## 🗺 Policy Map Visualization

The policy map shows:

* **Arrows** = chosen action at each cell
* **Blue cell** = Start
* **Green cell** = Goal
* **Gray cells** = Walls

This helps visually verify:

* Global consistency
* Local reactivity
* Absence of collision-seeking behavior

---

## ✅ Key Features

* End-to-end **sensor fusion** (global + local)
* Distance-shaped reward
* Fully learned obstacle avoidance
* Goal-directed behavior without hand-coded rules
* Target network stabilization
* Experience replay

---

## 🔬 Scientific Significance

This implementation demonstrates:

* **Coupled global–local representation learning**
* A biologically inspired fusion of:

  * **Compass-based navigation**
  * **Local tactile sensing**
* Comparison baseline for **hierarchical vs end-to-end control**
* A clean testbed for **neuro-navigation research**

---

## 🚀 Possible Extensions

* Add sensor noise
* Dynamic obstacles
* Multi-goal learning
* Curriculum learning
* Partial observability
* Multi-agent cooperation

---
