# coupled_dqn.py
"""
Coupled Geocentric+Egocentric DQN
- Single DQN that takes [geocentric_delta (2), local 3x3 occupancy (9)] -> absolute actions (Up,Right,Down,Left)
- Reward: shaped with distance delta, large goal reward, collision penalty, small step penalty
- Stabilized with Target network + replay buffer
- Saves model and learning curve; plots final path & policy map
"""

import os
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Config / Paths
# ----------------------------
OUTPUT_DIR = "Coupled_DQN_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTPUT_DIR, "coupled_dqn.pth")
LEARNING_CURVE = os.path.join(OUTPUT_DIR, "coupled_learning_curve.png")
FINAL_PATH_IMG = os.path.join(OUTPUT_DIR, "coupled_final_path.png")
POLICY_MAP_IMG = os.path.join(OUTPUT_DIR, "coupled_policy_map.png")
SEED = 42

# Reproducibility
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ----------------------------
# Maze / Grid (same as your egocentric.py)
# ----------------------------
GRID_SIZE = 15
MAZE = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [1,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
    [1,0,1,1,1,0,1,0,1,1,1,0,0,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,1,1,1,1,1,0,0,1,0,1],
    [1,0,0,0,1,0,0,0,0,0,0,0,1,0,1],
    [1,0,0,1,1,0,0,0,0,0,0,0,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,0,0,0,1,1,1,0,1,1,1,0,1],
    [1,0,1,1,1,0,0,0,0,0,1,0,0,0,1],
    [1,0,0,0,0,0,0,0,1,1,1,0,0,0,1],
    [1,1,0,0,1,0,1,0,0,0,0,0,1,0,1],
    [0,0,0,0,0,0,0,0,0,1,1,1,1,0,1],
    [0,0,1,0,1,1,1,1,1,1,1,1,1,1,1],
])
START = (14, 0)
GOAL = (0, 14)

# ----------------------------
# Coordinate utilities (kept consistent)
# ----------------------------
CELL_SIZE = 1.0
WORLD_OFFSET_X = -7.0
WORLD_OFFSET_Y = 7.0

def grid_to_world(grid_pos):
    r, c = grid_pos
    x = WORLD_OFFSET_X + c * CELL_SIZE
    y = WORLD_OFFSET_Y - r * CELL_SIZE
    return np.array([x, y])

START_WORLD = grid_to_world(START)
GOAL_WORLD = grid_to_world(GOAL)

# ----------------------------
# Environment wrapper (coupled env)
# ----------------------------
def is_valid(r, c):
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and MAZE[r, c] == 0

def get_local_patch(pos, radius=1):
    r, c = pos
    patch = np.ones((2*radius+1, 2*radius+1))
    for dr in range(-radius, radius+1):
        for dc in range(-radius, radius+1):
            rr, cc = r+dr, c+dc
            if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE:
                patch[dr+radius, dc+radius] = MAZE[rr, cc]
    return patch.flatten().astype(np.float32)  # 9-length

def step_env(pos, action):
    # action 0=up,1=right,2=down,3=left
    r, c = pos
    if action == 0: nr, nc = r-1, c
    elif action == 1: nr, nc = r, c+1
    elif action == 2: nr, nc = r+1, c
    else: nr, nc = r, c-1
    collided = False
    if not is_valid(nr, nc):
        # collision: stay in place
        nr, nc = r, c
        collided = True
    new_pos = (nr, nc)
    return new_pos, collided

# ----------------------------
# DQN: network, buffer
# ----------------------------
STATE_SIZE = 11  # 2 (geo vector) + 9 (local 3x3)
ACTION_SIZE = 4

class CoupledDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SIZE)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)
    def push(self, s,a,r,s2,d):
        self.buf.append((s,a,r,s2,d))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s,a,r,s2,d = map(np.array, zip(*batch))
        return s, a, r.astype(np.float32), s2, d.astype(np.uint8)
    def __len__(self):
        return len(self.buf)

# ----------------------------
# Hyperparameters
# ----------------------------
EPISODES = 2000            # adjust upwards if needed
MAX_STEPS = 300
BATCH_SIZE = 128
GAMMA = 0.99
LR = 1e-4
EPS_START = 1.0
EPS_MIN = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE_EP = 10

# Reward params
GOAL_REWARD = 200.0
COLLISION_PENALTY = -20.0
STEP_PENALTY = -0.01  # small penalty to avoid dithering

# ----------------------------
# Training loop
# ----------------------------
def compute_state(pos):
    # geocentric vector (goal_world - current_world)
    current_world = grid_to_world(pos)
    geo = (GOAL_WORLD - current_world).astype(np.float32)
    patch = get_local_patch(pos)  # 9-length
    state = np.concatenate([geo, patch])
    return state

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = CoupledDQN().to(device)
    target = CoupledDQN().to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(50000)

    eps = EPS_START
    episode_rewards = []

    for ep in range(1, EPISODES+1):
        pos = START
        state = compute_state(pos)
        total_reward = 0.0
        done = False

        for step in range(MAX_STEPS):
            # epsilon-greedy
            if random.random() < eps:
                action = random.randint(0, ACTION_SIZE-1)
            else:
                with torch.no_grad():
                    q = policy(torch.FloatTensor(state).unsqueeze(0).to(device))
                    action = int(q.argmax(1).item())

            # step
            dist_before = np.linalg.norm(grid_to_world(pos) - GOAL_WORLD)
            next_pos, collided = step_env(pos, action)
            dist_after = np.linalg.norm(grid_to_world(next_pos) - GOAL_WORLD)
            # reward composition
            if next_pos == GOAL:
                reward = GOAL_REWARD
                done = True
            else:
                if collided:
                    reward = COLLISION_PENALTY + STEP_PENALTY + (dist_before - dist_after)
                else:
                    reward = STEP_PENALTY + (dist_before - dist_after)

            total_reward += reward
            next_state = compute_state(next_pos)
            buffer.push(state, action, reward, next_state, done)

            state = next_state
            pos = next_pos

            # learning step
            if len(buffer) >= BATCH_SIZE:
                s, a, r, s2, d = buffer.sample(BATCH_SIZE)
                s_tensor = torch.FloatTensor(s).to(device)
                s2_tensor = torch.FloatTensor(s2).to(device)
                a_tensor = torch.LongTensor(a).to(device)
                r_tensor = torch.FloatTensor(r).to(device)
                d_tensor = torch.FloatTensor(d).to(device)

                q_pred = policy(s_tensor).gather(1, a_tensor.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target(s2_tensor).max(1)[0]
                q_target = r_tensor + GAMMA * q_next * (1 - d_tensor)

                loss = loss_fn(q_pred, q_target)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            if done:
                break

        episode_rewards.append(total_reward)
        eps = max(EPS_MIN, eps * EPS_DECAY)

        if ep % TARGET_UPDATE_EP == 0:
            target.load_state_dict(policy.state_dict())

        if ep % 50 == 0:
            avg = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            print(f"Ep {ep}/{EPISODES} | Avg (50): {avg:.2f} | eps: {eps:.3f}")

    torch.save(policy.state_dict(), MODEL_PATH)
    print(f"Training completed. Model saved to {MODEL_PATH}")
    # save learning curve
    plt.figure(figsize=(10,5)); plt.plot(episode_rewards); plt.title("Coupled DQN Learning Curve"); plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.grid(True)
    plt.savefig(LEARNING_CURVE); plt.close()
    print(f"Saved learning curve to {LEARNING_CURVE}")
    return policy

# ----------------------------
# Evaluation / Visualization
# ----------------------------
def evaluate(policy, greedy=True, max_steps=1000):
    policy.eval()
    pos = START
    path = [pos]
    total_reward = 0.0
    reached = False

    for step in range(max_steps):
        state = compute_state(pos)
        with torch.no_grad():
            q = policy(torch.FloatTensor(state).unsqueeze(0))
            if greedy:
                action = int(q.argmax(1).item())
            else:
                probs = torch.softmax(q, dim=1).cpu().numpy().flatten()
                action = int(np.random.choice(4, p=probs))

        dist_before = np.linalg.norm(grid_to_world(pos) - GOAL_WORLD)
        next_pos, collided = step_env(pos, action)
        dist_after = np.linalg.norm(grid_to_world(next_pos) - GOAL_WORLD)

        if next_pos == GOAL:
            reward = GOAL_REWARD
            total_reward += reward
            path.append(next_pos)
            reached = True
            break
        else:
            reward = STEP_PENALTY + (dist_before - dist_after)
            if collided:
                reward += COLLISION_PENALTY
            total_reward += reward

        pos = next_pos
        path.append(pos)

    return path, total_reward, reached

def plot_final_path(path, filename=FINAL_PATH_IMG):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_title("Coupled DQN Final Path")
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if MAZE[r, c] == 1:
                ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor='lightgray'))
    ax.add_patch(patches.Rectangle((START[1], START[0]), 1, 1, facecolor='blue', label='Start'))
    ax.add_patch(patches.Rectangle((GOAL[1], GOAL[0]), 1, 1, facecolor='green', label='Goal'))
    if path:
        px = [c + 0.5 for r,c in path]
        py = [r + 0.5 for r,c in path]
        ax.plot(px, py, marker='o', color='red')
    ax.set_xlim(0, GRID_SIZE); ax.set_ylim(0, GRID_SIZE); ax.set_aspect('equal'); ax.invert_yaxis()
    ax.grid(True); plt.savefig(filename); plt.close(fig)
    print(f"Saved final path to {filename}")

def plot_policy_map(policy, filename=POLICY_MAP_IMG):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_title("Coupled Policy Map (argmax action per cell)")
    ax.set_xlim(0, GRID_SIZE); ax.set_ylim(0, GRID_SIZE); ax.set_aspect('equal'); ax.invert_yaxis()
    ax.add_patch(patches.Rectangle((START[1], START[0]), 1, 1, facecolor='blue'))
    ax.add_patch(patches.Rectangle((GOAL[1], GOAL[0]), 1, 1, facecolor='green'))

    action_arrows = {0:(0,-0.35),1:(0.35,0),2:(0,0.35),3:(-0.35,0)}
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if MAZE[r, c] == 1:
                continue
            state = compute_state((r,c))
            with torch.no_grad():
                q = policy(torch.FloatTensor(state).unsqueeze(0))
                act = int(q.argmax(1).item())
            dx, dy = action_arrows[act]
            ax.arrow(c+0.5, r+0.5, dx, dy, head_width=0.18, head_length=0.18, fc='black', ec='black')

    ax.grid(True); plt.savefig(filename); plt.close(fig)
    print(f"Saved policy map to {filename}")

# ----------------------------
# Run training & evaluation
# ----------------------------
if __name__ == "__main__":
    # Train (or load if you prefer)
    if os.path.exists(MODEL_PATH):
        model = CoupledDQN()
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        print("Loaded existing coupled model.")
    else:
        print("Training coupled DQN...")
        model = train()

    # Evaluate multiple times
    successes = 0
    total_reward_list = []
    for i in range(5):
        path, tot, reached = evaluate(model, greedy=True)
        print(f"Eval {i+1}: reward={tot:.1f} reached={reached} steps={len(path)-1}")
        total_reward_list.append(tot)
        if reached: successes += 1
        plot_final_path(path, filename=os.path.join(OUTPUT_DIR, f"coupled_final_path_eval_{i+1}.png"))
    print(f"Successes: {successes}/5 | Avg reward: {np.mean(total_reward_list):.2f}")
    # policy map
    plot_policy_map(model, filename=os.path.join(OUTPUT_DIR, "coupled_policy_map.png"))
