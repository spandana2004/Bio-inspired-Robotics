# Q-Learning Maze Solver using MuJoCo (10×10 Grid World)

This project implements a **Q-Learning reinforcement learning agent** that learns to navigate a **10×10 maze with hazards** using **MuJoCo for visualization**. The agent starts at the **top-left corner** and must reach the **top-right goal** while avoiding red hazard cells.

A trained policy is demonstrated in real time using the MuJoCo viewer.

---

## 🎯 Project Objectives

- Model a **discrete grid-world maze** using MuJoCo geometry
- Train a **Q-Learning agent** from scratch
- Penalize collisions with hazards
- Reward reaching the goal
- Visualize the learned policy in **real-time 3D simulation**

---

## 🧠 Environment Description

- **Grid size:** 10 × 10  
- **Start state:** (0, 0) → top-left corner  
- **Goal state:** (0, 9) → top-right corner  
- **Red cells:** Hazards (terminal negative reward)
- **Green cell:** Goal (terminal positive reward)
- **Blue cube:** Agent  

### Rewards
| Event | Reward |
|------|--------|
| Goal reached | +10 |
| Hazard hit | −10 |
| Each step | −0.1 |

---

## 📂 Project Structure

```text
.
├── maze_q_learning.py     # Main training + simulation script
├── simulation_video.mp4  # (Optional) MuJoCo demo recording
└── README.md             # Project documentation
````

---

## 🛠 Requirements

Install the required dependencies:

```bash
pip install mujoco mujoco-python-viewer numpy
```

System Requirements:

* Python 3.8+
* MuJoCo 2.3+
* Working OpenGL support

---

## ▶️ How to Run

```bash
python maze_q_learning.py
```

What happens when you run the script:

1. The agent is **trained for 2000 episodes**
2. Q-table is learned using **epsilon-greedy exploration**
3. After training, a **slow real-time demonstration begins**
4. The learned policy is visualized using the MuJoCo viewer

---

## ⚙️ Q-Learning Parameters

```python
training_episodes = 2000
max_steps_per_episode = 100
learning_rate = 0.1
discount_rate = 0.99
epsilon = 0.2
```

---

## 🧱 Maze Encoding

The maze is defined as a **10×10 binary grid**:

* `0` → Free cell
* `1` → Hazard cell

Hazards are automatically converted into **MuJoCo red box geometries**.

Start: `(0,0)`
Goal: `(0,9)`

---

## 🧪 Learning Algorithm

The update rule used is standard **Q-Learning**:

```
Q(s,a) ← (1 − α) Q(s,a) + α [r + γ max Q(s',a')]
```

Where:

* α = learning rate
* γ = discount factor
* r = reward
* s, a = current state and action
* s′ = next state

---

## 🎥 Simulation Output

During the demo:

* The agent follows the **greedy policy**
* Movements are shown step-by-step
* Console logs show:

  * Step number
  * Agent state
  * Chosen action
  * Reward received

If included, the recorded video shows the learned navigation visually.

---

## ✅ Expected Behavior

* The agent **avoids hazards after training**
* The agent **moves directly toward the goal**
* The demonstration **terminates when the goal is reached**
* Total reward becomes positive after successful learning

---

## 🔬 Applications

This project is useful for:

* Reinforcement learning education
* Grid-world MDP visualization
* Robotics path planning intuition
* MuJoCo + discrete RL integration
* AI navigation research

---

## 🚀 Possible Extensions

* Add stochastic movement
* Add multiple goals
* Use SARSA instead of Q-Learning
* Add dynamic obstacles
* Use Deep Q-Learning (DQN)
* Add reward shaping or curriculum learning

---
