# Dual-Policy Reinforcement Learning Gating Architecture for Bio-Inspired Autonomous Navigation

This repository implements a **bio-inspired dual-policy navigation framework** combining:

* **Egocentric Policy** (local reactive navigation)
* **Geocentric Policy** (global map-based navigation)
* **Gating Policy Network (GPN)** — a Deep Q-Network that dynamically switches between the two

The architecture draws inspiration from biological navigation systems that integrate **localized sensing** with **global orientation**, especially under varying environmental hazard densities.

---

# 📁 Project Structure

```
Dual-Policy Reinforcement Learning Gating Architecture for Bio-Inspired Autonomous Navigation/
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
│   ├── policy_map_train_maze_1.png
│   ├── policy_map_train_maze_2.png
│   ├── policy_map_train_maze_3.png
│   ├── ... (up to maze_10)
│
│── Hazard_density_based_simulation/          
│   │── gating_final_hazards.py
│   │── angles_to_goal.csv
|   │── ego_vector_table.csv
|   |
│   │── hazard_comparison_run1/
│   │   ├── density_0.10/
│   │   │   ├── learning_curve_reward.png
│   │   │   ├── training_loss_curve.png
│   │   │   ├── performance_vs_density.png
│   │   │   ├── policy_dist_bar.png
│   │   │   ├── policy_dist_curve.png
│   │   │   ├── policy_map_train_maze_1.png ... maze_10.png
│   │   ├── density_0.20/
|   |   ├── density_0.30/
│   │   ├── density_0.40/
|   |   ├── density_0.50/
│   │   └── density_0.60/
│   │
│   │── hazard_comparison_run2/
│   │   └── (same density folders and files)
│   │
│   │── hazard_comparison_run3/
│       └── (same density folders and files)
│
└── README.md
```

This **fully integrated structure** includes all policy maps, learning curves, distribution plots, trajectories, and density-wise evaluation results.

---

# 🧠 System Overview

## 1. Egocentric Policy

Local, reactive navigation learned from **sensory vectors**.

* Learns obstacle-aware motion patterns
* Produces discretized action vectors
* Generates egocentric trajectory plots

**Key Files:**
`all_pattern_learn.py`, `mapping_vectors_for_test_maze.py`, `ego_vector_table.csv`

---

## 2. Geocentric Policy

Global, map-based navigation using **grid-world geometry**.

* Calculates target angles
* Generates optimal or near-optimal global paths

**Key Files:**
`geocentric_grid.py`, `geocentric_final_trajectory_generator.py`, `angles_to_goal.csv`

---

## 3. Gating Policy Network (Deep Q-Network)

The core controller that **chooses** between Egocentric and Geocentric actions based on learned Q-values.

* Learns switching behavior using reinforcement learning
* Responds to hazard density and environment configuration
* Provides combined navigation performance

**Outputs include:**

* Reward learning curve
* Training loss curve
* Policy distribution analysis
* Trajectories & heatmaps
* Performance vs hazard density
* Maze-wise policy maps (1–10)
* Multi-run (1–3) & multi-density (0.10–0.60) comparisons

All extended results from your ZIP are stored in:

```
Gating_Policy_Network/experimental_results/
```

---

# ⚙️ Installation

### Create environment

```bash
conda create -n dual_policy_nav python=3.9
conda activate dual_policy_nav
```

### Install dependencies

```bash
pip install numpy matplotlib pandas torch
```

---

# ▶️ How to Run

## Egocentric Policy

```bash
cd Egocentric_policy_framework
python all_pattern_learn.py
python mapping_vectors_for_test_maze.py
```

## Geocentric Policy

```bash
cd Geocentric_policy_framework
python geocentric_final_trajectory_generator.py
```

## Gating Policy Network

```bash
cd Gating_Policy_Network
python gating_policy.py
```
This loads the trained DQN (`gating_model_pytorch.pth`) and executes all evaluation routines.

## Hazard_density_based_simulation

```bash
cd Hazard_density_based_simulation
python gating_final_hazards.py
```

---

# 📊 Output Visualizations

Automatically generated:

* Agent trajectories
* Policy maps (train & test)
* Reward learning curves
* Training loss curves
* Policy selection distributions
* Performance vs hazard density
* Run-wise (1–3) and density-wise (0.10–0.60) comparisons

---

# 🧪 Experimental Setup

* Grid-based maze environment
* Discrete compass movement actions
* DQN-based gating mechanism
* Evaluation across densities: **0.10, 0.20, 0.40, 0.60**
* 10 distinct maze configurations per density
* 3 independent runs for robustness

---

# 📌 Key Features

* Bio-inspired navigation intelligence
* Dual-policy switching (local↔global) via reinforcement learning
* Rich visualization suite
* Generalization tests across many maze densities
* Fully interpretable behavior analysis

---

# 🚧 Known Limitations

* 2D environments only
* No real-time robotics deployment
* Hyperparameters fixed inside scripts

---

# 📬 Contact

Feel free to submit an issue or contribute to improve the framework.
