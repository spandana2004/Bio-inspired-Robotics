# Biologically Inspired Navigation Benchmark (EGO, GEO, HYBRID)

This project implements and compares three **biologically inspired navigation strategies** in a **15×15 maze** under **sensor noise and environmental perturbations**. It evaluates their performance using **path length, success rate, straightness, and A* optimality**.

Unlike learning-based approaches, all policies here are **rule-based and reactive**, inspired by:
- Egocentric navigation (local decision-making)
- Geocentric navigation (compass-based guidance)
- Hybrid strategies (geocentric + egocentric fallback)

---

## 🧭 Implemented Navigation Policies

| Policy | Description |
|--------|------------|
| **EGO** | Pure egocentric greedy navigation using local Manhattan distance |
| **GEO** | Search → Dance → Roll using noisy compass to estimate goal direction |
| **HYB** | Geocentric navigation with **egocentric fallback when blocked** |

---

## 🗺 Environment

- **Grid size:** 15 × 15
- **Cell types:**
  - `0` → Free cell
  - `1` → Obstacle / Wall
- **Start:** `(14, 0)` – bottom-left
- **Goal:** `(0, 14)` – top-right
- **World mapping:** Grid → Continuous 2D plane
- **Obstacle density variations:** `0.00`, `0.03`, `0.07`
- **Compass noise levels:** `0.0`, `0.05`, `0.1`

---

## 🧪 Experimental Design

For each:
- Obstacle density
- Compass noise level
- Navigation policy

The script runs:

```text
30 episodes × 3 policies × 3 densities × 3 noise levels
````

Each episode records:

* Whether the agent reached the goal
* Total path length
* Path efficiency vs **A*** optimal
* Straightness index during geocentric roll phase

---

## 📊 Evaluation Metrics

| Metric                 | Description                            |
| ---------------------- | -------------------------------------- |
| **Success Rate**       | Fraction of runs reaching the goal     |
| **Path Length**        | Total number of movement steps         |
| **Path Efficiency**    | A* shortest path / agent path          |
| **Straightness Index** | Net displacement ÷ roll-phase distance |
| **A* Length**          | Optimal grid path for comparison       |

---

## 🧠 Algorithmic Components

### A* Oracle

Used as a **ground-truth optimal path length** reference for efficiency scoring.

### Egocentric Policy (EGO)

* Chooses locally optimal move
* Uses Manhattan distance to goal
* No global direction estimate

### Geocentric Policy (GEO)

1. **SEARCH** – Random + noisy compass exploration
2. **DANCE** – Multiple compass samples to estimate heading
3. **ROLL** – Commit to averaged heading toward goal

### Hybrid Policy (HYB)

* Same as GEO
* **Automatically switches to EGO when movement is blocked**
* Returns to GEO when free again

---

## 📂 Output Files

All results are saved in:

```text
simple_bio_nav_outputs/
```

| File              | Contents                |
| ----------------- | ----------------------- |
| `results.csv`     | Per-episode raw results |
| `summary.json`    | Statistical summaries   |
| `succ_rate_*.png` | Success rate bar plots  |
| `path_eff_*.png`  | Path efficiency plots   |
| `traj_*.png`      | Example trajectories    |

---

## 📈 Plots Generated

* **Success rate vs policy**
* **Path efficiency vs policy**
* **Example trajectories for each policy**
* Comparison across:

  * Obstacle density
  * Compass noise

---

## 🔧 Requirements

```bash
pip install numpy matplotlib
```

Python version: **3.8+**

---

## ▶️ How to Run

```bash
python simple_bio_navigation.py
```

The script will:

1. Generate perturbed mazes
2. Run experiments for all policies
3. Compute all metrics
4. Save CSV + JSON summaries
5. Generate plots automatically

---

## 🧬 Biological Inspiration

This framework is inspired by:

* Ant and insect navigation
* Compass-based homing
* Path integration via heading commitment
* Local obstacle avoidance

It demonstrates how **simple biological heuristics scale under sensor noise and environmental uncertainty**.

---

## ✅ Expected Results

* **EGO** performs well in dense mazes but inefficiently
* **GEO** produces straighter paths but degrades under high noise
* **HYB** is the most robust under perturbations
* Path efficiency drops as obstacle density increases
* Success rate decreases with compass noise

---

## 🚀 Possible Extensions

* Add multiple goals
* Dynamic maze changes
* Energy-based cost model
* Continuous control instead of grid
* Multi-agent cooperation
* Deep-RL comparison baseline

