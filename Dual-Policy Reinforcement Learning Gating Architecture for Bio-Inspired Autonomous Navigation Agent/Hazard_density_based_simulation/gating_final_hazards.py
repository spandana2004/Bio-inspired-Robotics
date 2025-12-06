import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os
import csv
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import make_interp_spline

# --- 0. PyTorch Device Setup ---
print("--- PyTorch Hardware Check ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("--------------------------\n")

# --- 1. User-Configurable Section ---
GEO_Q_CSV = "/home/interface-lab/Spandana/IITM_Internship/NEW/final iitm simulation/angles_to_goal.csv"
EGO_VECTOR_TABLE_CSV = "/home/interface-lab/Spandana/IITM_Internship/NEW/final iitm simulation/ego_vector_table.csv"

# This is the base directory. Subfolders for each density will be created here.
BASE_OUTPUT_DIR = "/home/interface-lab/Spandana/IITM_Internship/NEW/final iitm simulation/hazard_comparison_run3"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Maze & start/goal configuration
MAZE_SIZE = (16, 16)
START_POS = (1, 1)
GOAL_POS  = (14, 14)

# Training & Evaluation Parameters
NUM_TRAINING_MAZES = 10
EPISODES = 3000
MAX_STEPS_PER_EPISODE = 250
TARGET_UPDATE_FREQUENCY = 10
FIXED_SEED = 64  # The seed used to lock all randomness

# --- 2. Seeding Function (CRITICAL FOR COMPARISON) ---
def set_seed(seed=64):
    """
    Sets the seed for all random number generators to ensure reproducibility.
    Resets weights and random choices for every training run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"--- GLOBAL SEED RESET TO {seed} ---")

# --- 3. Egocentric Map Generation ---
def load_vector_table_from_csv(filename):
    vector_map = {}
    try:
        with open(filename, "r") as f:
            reader = csv.reader(f); next(reader)
            for row in reader: vector_map[row[0]] = (float(row[1]), float(row[2]))
        print(f"Successfully loaded {len(vector_map)} egocentric patterns from '{filename}'")
        return vector_map
    except FileNotFoundError: print(f"FATAL ERROR: The file '{filename}' was not found."); exit()

def map_vectors_to_maze(maze, vector_map):
    rows, cols = maze.shape; maze_vector_field = np.full((rows, cols), None, dtype=object)
    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == 1: maze_vector_field[r, c] = (np.nan, np.nan)
            else:
                patch = np.ones((3, 3), dtype=int)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols: patch[dr + 1, dc + 1] = maze[nr, nc]
                pattern_key = " ".join(map(str, patch.flatten()))
                maze_vector_field[r, c] = vector_map.get(pattern_key, (0, 0))
    return maze_vector_field

# --- 4. Maze Generation and Pathfinding ---
def bfs_shortest_path(maze, start, goal):
    queue = deque([([start], start)]); visited = {start}
    while queue:
        path, current_pos = queue.popleft()
        if current_pos == goal: return path
        r, c = current_pos
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1] and
                maze[nr, nc] == 0 and (nr, nc) not in visited):
                visited.add((nr, nc)); new_path = list(path); new_path.append((nr, nc))
                queue.append((new_path, (nr, nc)))
    return None

def generate_maze(size, hazard_density, start_pos, goal_pos):
    while True:
        maze = np.zeros(size, dtype=int)
        maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
        free_cells = []
        for r in range(1, size[0] - 1):
            for c in range(1, size[1] - 1):
                if (r,c) == start_pos or (r,c) == goal_pos: continue
                if abs(r - start_pos[0]) <=1 and abs(c - start_pos[1]) <= 1: continue
                if abs(r - goal_pos[0]) <=1 and abs(c - goal_pos[1]) <= 1: continue
                free_cells.append((r,c))
        num_hazards = int(len(free_cells) * hazard_density)
        if num_hazards > len(free_cells): num_hazards = len(free_cells)
        hazard_locations = random.sample(free_cells, num_hazards)
        for r, c in hazard_locations: maze[r, c] = 1
        if bfs_shortest_path(maze, start_pos, goal_pos) is not None: return maze

# --- 5. Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity=20000): self.buffer = deque(maxlen=capacity)
    def remember(self, state, action, reward, next_state, done): self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size): return random.sample(self.buffer, batch_size)
    def __len__(self): return len(self.buffer)

# --- 6. The DQN Agent (PyTorch Implementation) ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__(); self.layer1 = nn.Linear(state_size, 32)
        self.layer2 = nn.Linear(32, 32); self.layer3 = nn.Linear(32, action_size)
    def forward(self, x):
        x = F.relu(self.layer1(x)); x = F.relu(self.layer2(x)); return self.layer3(x)

class DungBeetleAgent_PyTorch:
    def __init__(self, state_size, action_size, geo_data):
        self.geo_data = geo_data; self.state_size, self.action_size = state_size, action_size
        self.gamma, self.epsilon, self.epsilon_min, self.epsilon_decay = 0.95, 1.0, 0.01, 0.999
        self.learning_rate, self.batch_size = 0.0005, 64; self.memory = ReplayBuffer()
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()

    def get_state(self, pos, maze, ego_data):
        row, col = pos; rows, cols = maze.shape; local_view = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = row + dr, col + dc
                local_view.append(maze[nr, nc] if 0 <= nr < rows and 0 <= nc < cols else 1)
        geo_angle = self.geo_data[row, col]; geo_is_valid = 1.0 if not np.isnan(geo_angle) else 0.0
        geo_angle_norm = 0.0 if np.isnan(geo_angle) else (geo_angle + 180) / 360.0
        ego_vector = ego_data[row, col]; ego_vx, ego_vy = ego_vector[0], ego_vector[1]
        ego_vx_norm = 0.0 if np.isnan(ego_vx) else ego_vx / 740.0
        ego_vy_norm = 0.0 if np.isnan(ego_vy) else ego_vy / 740.0
        state_vector = np.array(local_view + [geo_angle_norm, geo_is_valid, ego_vx_norm, ego_vy_norm])
        return torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(device)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon: return random.randrange(self.action_size)
        with torch.no_grad():
            return self.policy_net(state).argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size: return None
        minibatch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        state_batch = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        action_batch = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_state_batch = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        done_batch = torch.tensor(dones, dtype=torch.float32).to(device)
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad(): next_q_values = self.target_net(next_state_batch).max(1)[0]
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return loss.item()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path); print(f"PyTorch model saved to {path}")

# --- 7. Environment Simulation Functions ---
def get_next_pos_from_angle_STANDARD(row, col, angle, maze):
    rows, cols = maze.shape
    if np.isnan(angle): return row, col
    if 45 <= angle < 135: move = (-1, 0)
    elif 135 <= angle or angle < -135: move = (0, -1)
    elif -135 <= angle < -45: move = (1, 0)
    else: move = (0, 1)
    nr, nc = row + move[0], col + move[1]
    return (nr, nc) if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0 else (row, col)

def get_next_pos_from_vector(row, col, vector, maze):
    rows, cols = maze.shape; vx, vy = vector
    if np.isnan(vx) or (vx == 0 and vy == 0): return row, col
    move = (0, 1) if vx > 0 else (0, -1) if abs(vx) > abs(vy) else (-1, 0) if vy > 0 else (1, 0)
    nr, nc = row + move[0], col + move[1]
    return (nr, nc) if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0 else (row, col)

# --- 8. Visualization and Analysis Functions ---
def plot_learning_curves(rewards, output_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Total Reward per Episode', color='b')
    plt.ylabel('Total Reward')
    plt.xlabel('Episode')
    plt.title('Agent Training Performance: Reward Curve')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    filepath = os.path.join(output_dir, 'learning_curve_reward.png')
    plt.savefig(filepath); plt.close(); print(f"Reward learning curve saved to {filepath}")

def plot_loss_curve(losses, output_dir, window=100):
    if not losses: return
    moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(12, 6))
    plt.plot(moving_avg, label=f'Moving Average (window={window})')
    plt.title('DQN Training Loss'); plt.xlabel('Training Steps'); plt.ylabel('Smooth L1 Loss')
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    filepath = os.path.join(output_dir, 'training_loss_curve.png')
    plt.savefig(filepath); plt.close(); print(f"Loss curve saved to {filepath}")

def plot_trajectory(maze, path, goal_pos, output_dir, filename='trajectory_unseen_test.png'):
    fig, ax = plt.subplots(figsize=(12, 12)); ax.imshow(maze, cmap='Greys', interpolation='none')
    if len(path) > 1: path_rows, path_cols = zip(*path)
    else: path_rows, path_cols = [path[0][0]], [path[0][1]]
    ax.plot(path_cols, path_rows, 'r-', lw=2.5, label='Agent Path'); ax.plot(START_POS[1], START_POS[0], 'bs', ms=12, label='Start')
    ax.plot(goal_pos[1], goal_pos[0], 'g*', ms=18, label='Goal'); ax.legend(); ax.set_title(f'Agent Trajectory on Unseen Test Maze', fontsize=16)
    filepath = os.path.join(output_dir, filename); plt.savefig(filepath); plt.close(); print(f"Trajectory plot saved to {filepath}")

def plot_policy_map(agent, maze, ego_data, goal_pos, output_dir, filename='policy_map_unseen_test.png'):
    fig, ax = plt.subplots(figsize=(12, 12)); ax.imshow(maze, cmap='Greys', interpolation='none')
    policy_colors = {'Geocentric': 'green', 'Egocentric': 'blue', 'Override': 'red'}
    for r in range(maze.shape[0]):
        for c in range(maze.shape[1]):
            if maze[r, c] == 0:
                state = agent.get_state((r, c), maze, ego_data); action = agent.choose_action(state)
                if action == 0: next_pos = get_next_pos_from_angle_STANDARD(r, c, agent.geo_data[(r,c)], maze)
                else: next_pos = get_next_pos_from_vector(r, c, ego_data[(r,c)], maze)
                is_stuck = (next_pos == (r,c)); policy = "Override" if is_stuck else ("Geocentric" if action == 0 else "Egocentric")
                if is_stuck:
                     valid_moves = [(r+dr, c+dc) for dr,dc in [(-1,0), (1,0), (0,-1), (0,1)] if 0<=r+dr<maze.shape[0] and 0<=c+dc<maze.shape[1] and maze[r+dr,c+dc]==0]
                     best_move = min(valid_moves, key=lambda p: abs(goal_pos[0]-p[0])+abs(goal_pos[1]-p[1])) if valid_moves else (r,c)
                     dr, dc = best_move[0]-r, best_move[1]-c
                else: dr, dc = next_pos[0]-r, next_pos[1]-c
                if not (dr == 0 and dc == 0): ax.arrow(c, r, dc*0.4, dr*0.4, head_width=0.3, head_length=0.3, fc=policy_colors[policy], ec=policy_colors[policy])
    ax.plot(START_POS[1], START_POS[0], 'bs', ms=12, label='Start'); ax.plot(goal_pos[1], goal_pos[0], 'g*', ms=18, label='Goal')
    legend_patches = [mpatches.Patch(color=c, label=p) for p, c in policy_colors.items()]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left'); ax.set_title(f"Agent's Policy Map on Unseen Test Maze", fontsize=16)
    filepath = os.path.join(output_dir, filename); plt.savefig(filepath, bbox_inches='tight'); plt.close(); print(f"Policy map saved to {filepath}")

def analyze_performance_vs_density(agent, ego_vector_table, output_dir):
    print("\n--- Analyzing Performance vs. Hazard Density ---")
    old_eps = agent.epsilon
    agent.epsilon = 0
    hazard_densities = np.linspace(0.0, 0.6, 7)
    mazes_per_density = 50
    results = {'densities': hazard_densities, 'success_rates': [], 'path_efficiencies': [], 'geo_freqs': [], 'ego_freqs': [], 'override_freqs': []}
    
    for density in hazard_densities:
        successes, total_agent_steps, total_optimal_steps = 0, 0, 0; policy_counts = {'Geocentric': 0, 'Egocentric': 0, 'Override': 0}
        
        for i in range(mazes_per_density):
            maze = generate_maze(MAZE_SIZE, density, START_POS, GOAL_POS); ego_data = map_vectors_to_maze(maze, ego_vector_table)
            current_pos = START_POS; path = [current_pos]; path_history = deque(maxlen=4); goal_reached = False
            for step in range(MAX_STEPS_PER_EPISODE):
                path_history.append(current_pos); state = agent.get_state(current_pos, maze, ego_data); action = agent.choose_action(state)
                if action == 0: tentative_next_pos = get_next_pos_from_angle_STANDARD(current_pos[0], current_pos[1], agent.geo_data[current_pos], maze)
                else: tentative_next_pos = get_next_pos_from_vector(current_pos[0], current_pos[1], ego_data[current_pos], maze)
                is_stuck = (len(path_history) > 1 and tentative_next_pos == path_history[-2]) or (tentative_next_pos == current_pos)
                policy_counts["Override" if is_stuck else ("Geocentric" if action == 0 else "Egocentric")] += 1
                if is_stuck:
                    valid_moves = [(current_pos[0]+dr, current_pos[1]+dc) for dr,dc in [(-1,0), (1,0), (0,-1), (0,1)] if 0<=current_pos[0]+dr<maze.shape[0] and 0<=current_pos[1]+dc<maze.shape[1] and maze[current_pos[0]+dr, current_pos[1]+dc]==0 and (len(path_history)<=1 or (current_pos[0]+dr, current_pos[1]+dc)!=path_history[-2])]
                    next_pos = min(valid_moves, key=lambda p: abs(GOAL_POS[0]-p[0])+abs(GOAL_POS[1]-p[1])) if valid_moves else current_pos
                else: next_pos = tentative_next_pos
                if current_pos == next_pos: break
                current_pos = next_pos; path.append(current_pos)
                if current_pos == GOAL_POS: goal_reached = True; break
            if goal_reached:
                successes += 1; optimal_path = bfs_shortest_path(maze, START_POS, GOAL_POS)
                if optimal_path: total_agent_steps += len(path)-1; total_optimal_steps += len(optimal_path)-1
        
        results['success_rates'].append((successes/mazes_per_density)*100)
        results['path_efficiencies'].append((total_optimal_steps/total_agent_steps)*100 if total_agent_steps > 0 else 0)
        total_steps = sum(policy_counts.values())
        if total_steps > 0:
            results['geo_freqs'].append((policy_counts['Geocentric']/total_steps)*100); results['ego_freqs'].append((policy_counts['Egocentric']/total_steps)*100); results['override_freqs'].append((policy_counts['Override']/total_steps)*100)
        else: results['geo_freqs'].append(0); results['ego_freqs'].append(0); results['override_freqs'].append(0)
    
    agent.epsilon = old_eps
    plot_performance_vs_density(results, output_dir)
    plot_policy_distribution_charts(results, output_dir)
    return results

def plot_performance_vs_density(results, output_dir):
    densities_pct = [d*100 for d in results['densities']]; fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(densities_pct, results['success_rates'], 'b-o', label='Success Rate'); ax1.set_ylabel('Success Rate (%)'); ax1.set_title('Agent Performance vs. Hazard Density'); ax1.grid(True, linestyle='--')
    ax2.plot(densities_pct, results['path_efficiencies'], 'r-s', label='Path Efficiency'); ax2.set_ylabel('Path Efficiency (%)'); ax2.set_xlabel('Hazard Density (%)'); ax2.grid(True, linestyle='--')
    filepath = os.path.join(output_dir, 'performance_vs_density.png'); plt.savefig(filepath); plt.close(); print(f"Performance vs density plot saved to {filepath}")

def plot_policy_distribution_charts(results, output_dir):
    densities_pct = np.array([d * 100 for d in results['densities']]); geo_freqs = np.array(results['geo_freqs']); ego_freqs = np.array(results['ego_freqs']); override_freqs = np.array(results['override_freqs'])
    plt.figure(figsize=(12, 8)); plt.bar(densities_pct, geo_freqs, color='g', edgecolor='white', width=5, label='Geocentric')
    plt.bar(densities_pct, ego_freqs, bottom=geo_freqs, color='b', edgecolor='white', width=5, label='Egocentric')
    plt.bar(densities_pct, override_freqs, bottom=geo_freqs+ego_freqs, color='r', edgecolor='white', width=5, label='Override')
    plt.xlabel('Hazard Density (%)'); plt.ylabel('Policy Usage Frequency (%)'); plt.title('Policy Distribution vs. Hazard Density (Bar Chart)')
    plt.xticks(densities_pct); plt.legend(); plt.tight_layout(); filepath_bar = os.path.join(output_dir, 'policy_dist_bar.png')
    plt.savefig(filepath_bar); plt.close(); print(f"Policy distribution bar chart saved to {filepath_bar}")
    plt.figure(figsize=(12, 8)); densities_smooth = np.linspace(densities_pct.min(), densities_pct.max(), 300)
    if len(densities_pct) > 3:
        spl_geo = make_interp_spline(densities_pct, geo_freqs, k=3); geo_smooth = spl_geo(densities_smooth)
        spl_ego = make_interp_spline(densities_pct, ego_freqs, k=3); ego_smooth = spl_ego(densities_smooth)
        spl_over = make_interp_spline(densities_pct, override_freqs, k=3); over_smooth = spl_over(densities_smooth)
        plt.plot(densities_smooth, geo_smooth, 'g-', label='Geocentric'); plt.plot(densities_pct, geo_freqs, 'go')
        plt.plot(densities_smooth, ego_smooth, 'b-', label='Egocentric'); plt.plot(densities_pct, ego_freqs, 'bs')
        plt.plot(densities_smooth, over_smooth, 'r-', label='Override'); plt.plot(densities_pct, override_freqs, 'r^')
    else:
        plt.plot(densities_pct, geo_freqs, 'g-o', label='Geocentric'); plt.plot(densities_pct, ego_freqs, 'b-s', label='Egocentric'); plt.plot(densities_pct, override_freqs, 'r-^', label='Override')
    plt.xlabel('Hazard Density (%)'); plt.ylabel('Policy Usage Frequency (%)'); plt.title('Policy Distribution vs. Hazard Density (Smooth Curve)')
    plt.xticks(densities_pct); plt.grid(True, linestyle='--'); plt.legend(); plt.tight_layout(); filepath_curve = os.path.join(output_dir, 'policy_dist_curve.png')
    plt.savefig(filepath_curve); plt.close(); print(f"Policy distribution curve plot saved to {filepath_curve}")

def plot_all_training_policy_maps(agent, mazes, ego_maps, goal_pos, output_dir):
    print("\nGenerating Policy Maps for all 10 Training Mazes...")
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    policy_colors = {'Geocentric': 'green', 'Egocentric': 'blue', 'Override': 'red'}

    for idx, (maze, ego_data) in enumerate(zip(mazes, ego_maps)):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(maze, cmap='Greys', interpolation='none')
        for r in range(maze.shape[0]):
            for c in range(maze.shape[1]):
                if maze[r, c] == 0 and (r, c) != goal_pos:
                    state = agent.get_state((r, c), maze, ego_data)
                    with torch.no_grad(): action = agent.policy_net(state).argmax().item()
                    if action == 0: tentative_next_pos = get_next_pos_from_angle_STANDARD(r, c, agent.geo_data[(r, c)], maze); policy = 'Geocentric'
                    else: tentative_next_pos = get_next_pos_from_vector(r, c, ego_data[(r, c)], maze); policy = 'Egocentric'
                    is_stuck = (tentative_next_pos == (r, c))
                    if is_stuck:
                        policy = 'Override'
                        valid_moves = []
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1] and maze[nr, nc] == 0: valid_moves.append((nr, nc))
                        if valid_moves:
                            best_move = min(valid_moves, key=lambda p: abs(goal_pos[0]-p[0]) + abs(goal_pos[1]-p[1]))
                            dr, dc = best_move[0] - r, best_move[1] - c
                        else: dr, dc = 0, 0 
                    else: dr, dc = tentative_next_pos[0] - r, tentative_next_pos[1] - c
                    if not (dr == 0 and dc == 0): ax.arrow(c, r, dc * 0.4, dr * 0.4, head_width=0.3, head_length=0.3, fc=policy_colors[policy], ec=policy_colors[policy])

        ax.plot(START_POS[1], START_POS[0], 'bs', ms=12, label='Start')
        ax.plot(goal_pos[1], goal_pos[0], 'g*', ms=18, label='Goal')
        legend_patches = [mpatches.Patch(color=c, label=p) for p, c in policy_colors.items()]
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        filename = f"policy_map_train_maze_{idx+1}.png"
        ax.set_title(f"Agent's Policy Map ({filename})", fontsize=16)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight'); plt.close(fig); print(f"Saved: {filepath}")

    agent.epsilon = old_epsilon

def plot_training_self_success(densities, self_success_rates, output_dir):
    """
    Plot 1: Success rate of each agent when tested on the specific density it was trained on.
    """
    densities_pct = [d * 100 for d in densities]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(densities_pct, self_success_rates, color='purple', alpha=0.7, width=5, edgecolor='black')
    
    plt.xlabel('Training Hazard Density (%)')
    plt.ylabel('Success Rate on Native Density (%)')
    plt.title('Self-Consistency: Performance on Training Density')
    plt.xticks(densities_pct)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add text labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')

    filepath = os.path.join(output_dir, 'summary_self_success_rates.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Summary Plot 1 saved to {filepath}")

def plot_cross_density_comparison(all_model_results, output_dir):
    """
    Plot 2: A single graph containing lines for every trained model, 
    showing how they perform across ALL densities.
    """
    plt.figure(figsize=(14, 8))
    
    # Define a color map to distinguish lines easily
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_model_results)))
    
    for idx, (train_density, result_data) in enumerate(all_model_results.items()):
        # result_data contains: {'densities': [...], 'success_rates': [...], ...}
        eval_densities_pct = [d * 100 for d in result_data['densities']]
        success_rates = result_data['success_rates']
        
        plt.plot(eval_densities_pct, success_rates, marker='o', linewidth=2, 
                 label=f'Trained @ {train_density*100:.0f}%', color=colors[idx])

    plt.xlabel('Evaluation Hazard Density (%)')
    plt.ylabel('Success Rate (%)')
    plt.title('Cross-Evaluation: Generalization Capabilities of Each Model')
    plt.legend(title="Model Training Config", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 105)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'summary_cross_evaluation.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Summary Plot 2 saved to {filepath}")

# --- 10. Main Execution Block ---
# --- 10. Main Execution Block ---
if __name__ == "__main__":
    
    # 1. LOAD DATA ONCE
    try:
        geocentric_data = np.loadtxt(GEO_Q_CSV, delimiter=',')
        ego_vector_table = load_vector_table_from_csv(EGO_VECTOR_TABLE_CSV)
    except Exception as e: print(f"Error during initial data loading: {e}"); exit()

    # 2. DEFINE DENSITIES TO LOOP THROUGH
    # (Note: analyze_performance checks 0.0 to 0.6, ensure these match for "Self Success" extraction)
    DENSITIES_TO_TEST = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

    # --- STORAGE FOR FINAL PLOTS ---
    # Dictionary to store full results: { 0.10: results_dict, 0.20: results_dict, ... }
    global_model_results = {} 
    # List to store tuple: (training_density, success_on_that_density)
    self_performance_summary = []

    # 3. LOOP
    for density in DENSITIES_TO_TEST:
        print("\n" + "#"*60)
        print(f"STARTING SIMULATION FOR HAZARD DENSITY: {density*100:.0f}%")
        print("#"*60)

        # A. RESET SEED
        set_seed(FIXED_SEED)

        # B. SETUP FOLDER
        CURRENT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"density_{density:.2f}")
        os.makedirs(CURRENT_OUTPUT_DIR, exist_ok=True)
        MODEL_SAVE_PATH = os.path.join(CURRENT_OUTPUT_DIR, 'gating_model_pytorch.pth')

        # C. GENERATE MAZES
        print(f"Generating {NUM_TRAINING_MAZES} mazes with {density*100:.0f}% density...")
        training_mazes = [generate_maze(MAZE_SIZE, density, START_POS, GOAL_POS) for _ in range(NUM_TRAINING_MAZES)]
        training_ego_maps = [map_vectors_to_maze(m, ego_vector_table) for m in training_mazes]
        
        # D. INITIALIZE NEW AGENT
        STATE_SIZE = 13; ACTION_SIZE = 2
        agent = DungBeetleAgent_PyTorch(STATE_SIZE, ACTION_SIZE, geocentric_data)
        
        rewards_per_episode, steps_per_episode, training_losses = [], [], []

        print(f"--- Starting Training ({density*100:.0f}%) ---")
        for e in range(EPISODES):
            maze_index = e % NUM_TRAINING_MAZES; MAZE = training_mazes[maze_index]
            egocentric_data = training_ego_maps[maze_index]
            current_pos, total_reward = START_POS, 0
            state = agent.get_state(current_pos, MAZE, egocentric_data); path_history = deque(maxlen=4)
            for step in range(MAX_STEPS_PER_EPISODE):
                path_history.append(current_pos); action = agent.choose_action(state)
                if action == 0: tentative_next_pos = get_next_pos_from_angle_STANDARD(current_pos[0], current_pos[1], geocentric_data[current_pos], MAZE)
                else: tentative_next_pos = get_next_pos_from_vector(current_pos[0], current_pos[1], egocentric_data[current_pos], MAZE)
                is_stuck = (len(path_history)>1 and tentative_next_pos==path_history[-2]) or (tentative_next_pos==current_pos)
                if is_stuck:
                    valid_moves = [(current_pos[0]+dr, current_pos[1]+dc) for dr,dc in [(-1,0), (1,0), (0,-1), (0,1)] if 0<=current_pos[0]+dr<MAZE.shape[0] and 0<=current_pos[1]+dc<MAZE.shape[1] and MAZE[current_pos[0]+dr, current_pos[1]+dc]==0 and (len(path_history)<=1 or (current_pos[0]+dr, current_pos[1]+dc)!=path_history[-2])]
                    next_pos = min(valid_moves, key=lambda p:abs(GOAL_POS[0]-p[0])+abs(GOAL_POS[1]-p[1])) if valid_moves else current_pos
                else: next_pos = tentative_next_pos
                done = (next_pos == GOAL_POS) or (step == MAX_STEPS_PER_EPISODE - 1)
                dist_old = abs(GOAL_POS[0]-current_pos[0])+abs(GOAL_POS[1]-current_pos[1]); dist_new = abs(GOAL_POS[0]-next_pos[0])+abs(GOAL_POS[1]-next_pos[1])
                if next_pos == GOAL_POS: reward = 100
                elif is_stuck: reward = -15
                elif step == MAX_STEPS_PER_EPISODE-1: reward = -25
                elif next_pos == current_pos: reward = -5
                else: reward = (dist_old - dist_new) * 2 - 0.2
                total_reward += reward
                next_state = agent.get_state(next_pos, MAZE, egocentric_data)
                agent.memory.remember(state.cpu().numpy().flatten(), action, reward, next_state.cpu().numpy().flatten(), done)
                loss = agent.replay()
                if loss is not None: training_losses.append(loss)
                state, current_pos = next_state, next_pos
                if done: break
            
            rewards_per_episode.append(total_reward); steps_per_episode.append(step + 1)
            if (e+1) % TARGET_UPDATE_FREQUENCY == 0: agent.update_target_model()
            agent.decay_epsilon()
            
            if (e+1) % 100 == 0:
                print(f"Density: {density*100:.0f}% | Ep: {e+1}/{EPISODES} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f}")

        # E. SAVE AND ANALYZE FOR THIS DENSITY
        print(f"--- Training Complete for {density*100:.0f}%, Generating Analysis... ---")
        agent.save_model(MODEL_SAVE_PATH)
        
        plot_learning_curves(rewards_per_episode, CURRENT_OUTPUT_DIR)
        plot_loss_curve(training_losses, CURRENT_OUTPUT_DIR)
        plot_all_training_policy_maps(agent, training_mazes, training_ego_maps, GOAL_POS, CURRENT_OUTPUT_DIR)
        
        # Test on an UNSEEN maze (Just one for trajectory plot)
        print("Running single unseen evaluation for plot...")
        agent.epsilon = 0
        for i in range(50):
            test_maze = generate_maze(MAZE_SIZE, density, START_POS, GOAL_POS)
            test_ego_data = map_vectors_to_maze(test_maze, ego_vector_table)
            current_pos = START_POS; path = [current_pos]; goal_reached = False
            for step in range(MAX_STEPS_PER_EPISODE):
                state = agent.get_state(current_pos, test_maze, test_ego_data); action = agent.choose_action(state)
                if action == 0: next_pos = get_next_pos_from_angle_STANDARD(current_pos[0], current_pos[1], agent.geo_data[current_pos], test_maze)
                else: next_pos = get_next_pos_from_vector(current_pos[0], current_pos[1], test_ego_data[current_pos], test_maze)
                path.append(next_pos); current_pos = next_pos
                if current_pos == GOAL_POS: goal_reached = True; break
                if len(path) > 1 and current_pos == path[-2]: break
            if goal_reached:
                plot_trajectory(test_maze, path, GOAL_POS, CURRENT_OUTPUT_DIR)
                plot_policy_map(agent, test_maze, test_ego_data, GOAL_POS, CURRENT_OUTPUT_DIR)
                break
        
        # --- FULL ANALYSIS VS ALL DENSITIES ---
        # Note: We capture the return value 'results' here
        results = analyze_performance_vs_density(agent, ego_vector_table, CURRENT_OUTPUT_DIR)
        
        # Store for Mega-Plots
        global_model_results[density] = results
        
        # Extract performance on *native* density
        # The 'densities' list in results is [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        # We need to find the index that matches the current 'density'
        try:
            # Find closest index to handle floating point precision
            idx = np.argmin(np.abs(np.array(results['densities']) - density))
            native_success = results['success_rates'][idx]
            self_performance_summary.append(native_success)
        except Exception as e:
            print(f"Warning: Could not extract native success rate for {density}: {e}")
            self_performance_summary.append(0)

    # --- 4. FINAL SUMMARY PLOTS (After all density loops are done) ---
    print("\n" + "="*60)
    print("GENERATING FINAL SUMMARY COMPARISON PLOTS")
    print("="*60)
    
    # Extract just the success rates for the first plot
    plot_training_self_success(DENSITIES_TO_TEST, self_performance_summary, BASE_OUTPUT_DIR)
    
    # Generate the combined line graph
    plot_cross_density_comparison(global_model_results, BASE_OUTPUT_DIR)
    
    print("All simulations and summary plots complete.")