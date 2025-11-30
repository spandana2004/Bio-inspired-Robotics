# Dual-Policy Reinforcement Learning Gating Architecture for Bio-Inspired Autonomous Navigation

This repository implements a **bio-inspired dual-policy navigation framework** for autonomous agents, combining:

* An **Egocentric policy** (local, sensory-based navigation)
* A **Geocentric policy** (global, map-based navigation)
* A **Gating Policy Network (GPN)** that adaptively selects between the two using reinforcement learning

The architecture is designed to emulate biological navigation systems by dynamically switching between local reactive behavior and global goal-directed planning.

---

## 📁 Project Structure

```
Dual-Policy Reinforcement Learning Gating Architecture for Bio-Inspired Autonomous Navigation Agent/
│
├── Egocentric_policy_framework/
│   ├── all_pattern_learn.py
│   ├── mapping_vectors_for_test_maze.py
│   ├── ego_vector_table.csv
│   ├── Policy_map.png
│   └── Discretized_actions_and_final_trajectory.png
│
├── Geocentric_policy_framework/
│   ├── geocentric_grid.py
│   ├── geocentric_final_trajectory_generator.py
│   ├── angles_to_goal.csv
│   ├── Policy_map.png
│   └── Final_trajectory.png
│
├── Gating_Policy_Network/
│   ├── gating_policy.py
│   ├── gating_model_pytorch.pth
│   ├── learning_curve_reward.png
│   ├── training_loss_curve.png
│   ├── performance_vs_density.png
│   ├── policy_dist_bar.png
│   ├── policy_dist_curve.png
│   ├── policy_map_test.png
│   ├── trajectory_test.png
│   └── policy_map_train_maze_*.png
│
└── (visualization and experimental result files)
```

---

## 🧠 System Overview

### 1. Egocentric Policy

* Learns navigation behavior based purely on **local sensory vectors**.
* Generates discretized movement vectors for obstacle avoidance.
* Key files:

  * `all_pattern_learn.py`
  * `mapping_vectors_for_test_maze.py`
  * `ego_vector_table.csv`

### 2. Geocentric Policy

* Performs **global, goal-oriented planning** based on a grid-world map.
* Computes angles and trajectories to the target.
* Key files:

  * `geocentric_grid.py`
  * `geocentric_final_trajectory_generator.py`
  * `angles_to_goal.csv`

### 3. Gating Policy Network (DQN)

* A **Deep Q-Network (PyTorch)** that decides whether to use:

  * Egocentric policy **or**
  * Geocentric policy
* Learns from environment density and navigation performance.
* Key files:

  * `gating_policy.py`
  * `gating_model_pytorch.pth`

---

## ⚙️ Installation

### 1. Create a Python environment (recommended)

```bash
conda create -n dual_policy_nav python=3.9
conda activate dual_policy_nav
```

### 2. Install dependencies

```bash
pip install numpy matplotlib pandas torch
```

> If you encounter missing packages, install them using `pip install <package>`.

---

## ▶️ How to Run

### 1. Egocentric Policy Learning

```bash
cd Egocentric_policy_framework
python all_pattern_learn.py
```

For testing on a predefined maze:

```bash
python mapping_vectors_for_test_maze.py
```

Outputs:

* Discretized action maps
* Final trajectory plots
* Policy visualization

---

### 2. Geocentric Navigation

```bash
cd Geocentric_policy_framework
python geocentric_final_trajectory_generator.py
```

Outputs:

* Global policy maps
* Final geocentric trajectory plots
* Angle-to-goal tables

---

### 3. Gating Policy Network (Main Controller)

```bash
cd Gating_Policy_Network
python gating_policy.py
```

This will:

* Load the pretrained DQN (`gating_model_pytorch.pth`)
* Run policy selection on different mazes
* Evaluate performance vs obstacle density
* Generate:

  * Learning curves
  * Training loss curves
  * Policy distribution plots
  * Test trajectories
  * Policy heatmaps

---

## 📊 Output Visualizations

The framework automatically saves:

* **Agent trajectories**
* **Policy maps (train & test)**
* **Reward learning curves**
* **Training loss curves**
* **Policy selection distributions**
* **Performance vs maze density plots**

All outputs are saved inside the respective framework folders.

---

## 🧪 Experimental Setup

* Environment: Discrete grid-based maze
* Actions: Discretized compass movements
* Controller: Deep Q-Network (DQN)
* Inputs to Gating Network:

  * Local egocentric vectors
  * Global goal direction
  * Environmental density indicators

---

## 📌 Key Features

* Hybrid bio-inspired navigation system
* Adaptive switching between local and global policies
* PyTorch-based deep reinforcement learning
* Interpretable policy visualizations
* Robust evaluation across multiple maze densities

---

## 🧾 Model Files

| File                       | Description                    |
| -------------------------- | ------------------------------ |
| `gating_model_pytorch.pth` | Trained gating policy DQN      |
| `ego_vector_table.csv`     | Learned egocentric vectors     |
| `angles_to_goal.csv`       | Geocentric angle look-up table |

---

## 🚧 Known Limitations

* Designed for **2D grid environments only**
* Real-time robotics deployment is **not implemented**
* Hyperparameters are fixed inside the scripts
