# **Bio-inspired-Robotics**

This repository is a **comprehensive research workspace for bio-inspired navigation and reinforcement learning**, containing **multiple independent but related projects** that explore:

* Classical and deep reinforcement learning
* Egocentric (local) and geocentric (global) navigation
* Hybrid and gated control architectures
* Biomimetic agent simulations
* Benchmarking and performance evaluation

Each major folder is a **standalone research module with its own detailed `README.md`** explaining implementation, usage, and results.

---

## 📁 **Repository Modules (High-Level Overview)**

```
Benchmarking_Bio_Inspired_Navigation/
│
├── Benchmarking Bio-inspired navigation/
├── Bio-inspired_ego and geo_coupled_policy_dqn/
├── Dual-Policy Reinforcement Learning Gating Architecture for Bio-Inspired Autonomous Navigation/
├── Simple Ant Simulation/
├── grid_world_game/
└── README.md  ← (This file: Project Cover)
```

---

## 🔹 **1. Benchmarking Bio-Inspired Navigation**

**Purpose:**
Centralized **evaluation and comparison framework** used to benchmark multiple navigation strategies under varying environment densities and complexities.

**Focus:**

* Performance comparison across controllers
* Success rate, reward, and path efficiency analysis
* Cross-model robustness evaluation

📄 *Refer to this folder’s `README.md` for full experimental protocols and results.*

---

## 🔹 **2. Bio-Inspired Egocentric & Geocentric Coupled Policy DQN**

**Purpose:**
Implements a **single end-to-end Deep Q-Network (DQN)** that **jointly learns egocentric and geocentric cues** without explicit policy switching.

**Focus:**

* Fused local–global state representation
* Unified deep reinforcement learning controller
* Comparison against modular and gated systems

📄 *Refer to this folder’s `README.md` for model architecture, training procedure, and results.*

---

## 🔹 **3. Dual-Policy Reinforcement Learning Gating Architecture for Bio-Inspired Autonomous Navigation**

**Purpose:**
A **modular biologically inspired navigation system** consisting of:

* Egocentric policy (local sensory navigation)
* Geocentric policy (global map-based planning)
* A Deep RL **Gating Policy Network (GPN)** to adaptively switch between them

**Focus:**

* Strategy switching inspired by biological navigation
* Interpretable hybrid control
* Policy selection under varying environmental conditions

📄 *Refer to this folder’s `README.md` for full system design, training, and evaluation details.*

---

## 🔹 **4. Simple Ant Simulation**

**Purpose:**
A **biomimetic motion and navigation simulator** inspired by ant-like movement and path-following behavior.

**Focus:**

* Bio-locomotion primitives
* Goal-directed ant navigation
* Trajectory logging and visualization

📄 *Refer to this folder’s `README.md` for simulation setup and execution.*

---

## 🔹 **5. Grid World Game**

**Purpose:**
A **classical reinforcement learning baseline environment** implemented in a 2D grid world with hazards and goals.

**Focus:**

* Tabular Q-Learning
* Discrete state–action spaces
* MuJoCo-based visualization
* Baseline comparison for bio-inspired methods

📄 *Refer to this folder’s `README.md` for environment design and learning setup.*

---

## 🧠 **Research Scope**

This repository collectively studies:

* Bio-inspired navigation principles
* Local vs global spatial representations
* Modular vs end-to-end deep RL architectures
* Strategy switching via reinforcement learning
* Comparative benchmarking across controllers

It is suitable for:

* **Academic research**
* **Thesis and dissertation work**
* **Algorithm benchmarking**
* **Bio-inspired robotics studies**

---

## ⚙️ **Common Dependencies**

Most modules rely on:

```bash
numpy
matplotlib
pandas
torch
mujoco
```

Each folder may require additional packages — refer to the **individual `README.md` files** for exact dependency lists.

---

## NOTE
Please make sure the file path is given correctly according to how you saved them on your system or directory.
