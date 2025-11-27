# Ant Foraging Simulation with Path Integration (MuJoCo)

This project simulates an **ant performing a random correlated walk to search for food** inside a bounded environment using **MuJoCo**. Once the ant reaches the food, it **returns to the start position using path integration** by reversing the sum of its movement vectors.

The repository also includes:
- ✅ A **final path image** (matplotlib plot)
- ✅ A **simulation video** recorded from the MuJoCo viewer

---

## 📌 Features

- Random **correlated walk** for outbound food search  
- **Square food region** as a target
- **Wall-bounce logic** to keep the ant inside the arena
- **Vector-sum-based return navigation** (path integration)
- **Real-time MuJoCo visualization**
- **Trajectory plotting** using Matplotlib

---

## 🧠 Concept Overview

1. The ant starts at the origin.
2. It performs a **biased random walk** with angular noise.
3. Upon entering the food square:
   - The walk stops
   - All movement vectors are summed
4. The ant **returns directly to the start** using the **negative of the summed vector**.
5. The full trajectory is plotted:
   - Blue = outbound path
   - Red = return path

This mimics **biologically inspired navigation using path integration**.

---

## 📂 Files in This Project

```text
.
├── ant_foraing_path_integration.py        # Main simulation script
├── final_ant_simulation.png           # Generated matplotlib plot
├── Simulation on mujoco.mp4     # Recorded MuJoCo simulation
└── README.md                # Project documentation
