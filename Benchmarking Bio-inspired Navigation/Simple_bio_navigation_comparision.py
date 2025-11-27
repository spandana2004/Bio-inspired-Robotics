import os
import math
import random
import json
from collections import deque
from heapq import heappush, heappop

import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Config & paths
# --------------------------
OUTPUT_DIR = "simple_bio_nav_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(OUTPUT_DIR, "results.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "summary.json")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --------------------------
# Maze (same layout you provided)
# --------------------------
GRID_SIZE = 15
BASE_MAZE = np.array([
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
], dtype=np.int32)

START = (14, 0)
GOAL = (0, 14)

CELL_SIZE = 1.0
WORLD_OFFSET_X = -7.0
WORLD_OFFSET_Y = 7.0

def grid_to_world(grid_pos):
    r, c = grid_pos
    x = WORLD_OFFSET_X + c * CELL_SIZE
    y = WORLD_OFFSET_Y - r * CELL_SIZE
    return np.array([x, y], dtype=np.float32)

START_WORLD = grid_to_world(START)
GOAL_WORLD = grid_to_world(GOAL)

# --------------------------
# Utilities
# --------------------------
def is_valid(maze, r, c):
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and maze[r, c] == 0

def get_neighbors(pos):
    r, c = pos
    return [
        (r-1, c),
        (r, c+1),
        (r+1, c),
        (r, c-1),
    ]

ACTION_TO_DELTA = {0:(-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}

def step_env(maze, pos, action):
    r, c = pos
    dr, dc = ACTION_TO_DELTA[action]
    nr, nc = r+dr, c+dc
    collided = False
    if not is_valid(maze, nr, nc):
        nr, nc = r, c
        collided = True
    return (nr, nc), collided

def angle_to_goal_from_grid(pos):
    pos_w = grid_to_world(pos)
    vec = GOAL_WORLD - pos_w
    ang = math.atan2(vec[1], vec[0])
    return ang

def discrete_action_from_angle(angle):
    # 0=up,1=right,2=down,3=left
    deg = (math.degrees(angle) + 360.0) % 360.0
    if deg <= 45 or deg > 315:
        return 1  # right
    elif deg <= 135:
        return 0  # up
    elif deg <= 225:
        return 3  # left
    else:
        return 2  # down

def sample_noisy_compass(pos, sigma=0.0):
    true_ang = angle_to_goal_from_grid(pos)
    noisy = true_ang + np.random.normal(scale=sigma)
    noisy = (noisy + math.pi) % (2*math.pi) - math.pi
    return noisy

# --------------------------
# A* (grid) for optimal path length
# --------------------------
def astar_shortest_path_length(maze, start, goal):
    if not is_valid(maze, start[0], start[1]) or not is_valid(maze, goal[0], goal[1]):
        return None
    h = lambda p: abs(p[0]-goal[0]) + abs(p[1]-goal[1])
    openpq = []
    heappush(openpq, (h(start), 0, start))
    gscore = {start: 0}
    visited = set()
    while openpq:
        f, g, cur = heappop(openpq)
        if cur == goal:
            return g
        if cur in visited:
            continue
        visited.add(cur)
        for nbr in get_neighbors(cur):
            if not is_valid(maze, nbr[0], nbr[1]): continue
            ng = g + 1
            if nbr not in gscore or ng < gscore[nbr]:
                gscore[nbr] = ng
                heappush(openpq, (ng + h(nbr), ng, nbr))
    return None

# --------------------------
# Policies
# --------------------------

# Egocentric reactive single-step
def ego_policy_step(maze, pos):
    cand = []
    for a in range(4):
        dr, dc = ACTION_TO_DELTA[a]
        nr, nc = pos[0]+dr, pos[1]+dc
        if is_valid(maze, nr, nc):
            # manhattan distance heuristic to goal
            d = abs(nr-GOAL[0]) + abs(nc-GOAL[1])
            cand.append((d,a))
    if not cand:
        return random.randint(0,3)
    cand.sort()
    best_d = cand[0][0]
    best_actions = [a for d,a in cand if d==best_d]
    return random.choice(best_actions)

# Wrapper to run EGO as a full episode
def ego_policy_episode(maze, max_steps=500):
    pos = START
    path = [pos]
    for step in range(max_steps):
        action = ego_policy_step(maze, pos)
        next_pos, collided = step_env(maze, pos, action)
        pos = next_pos
        path.append(pos)
        if pos == GOAL:
            return {"path": path, "reached": True, "roll_start": None}
    return {"path": path, "reached": False, "roll_start": None}

# Geocentric: SEARCH -> DANCE -> ROLL
DETECTION_RADIUS = 1.5
DANCE_STEPS = 6
N_COMPASS_SAMPLES = 8

def geo_policy_episode(maze, max_steps=500, compass_noise=0.08):
    pos = START
    path = [pos]
    phase = "SEARCH"
    steps_in_phase = 0
    target_heading = None
    roll_start_idx = None

    for step in range(max_steps):
        # detection
        if np.linalg.norm(grid_to_world(pos) - GOAL_WORLD) <= DETECTION_RADIUS:
            if phase == "SEARCH":
                phase = "DANCE"
                steps_in_phase = 0

        # decide action
        action = None
        if phase == "SEARCH":
            if random.random() < 0.2:
                ang = sample_noisy_compass(pos, sigma=compass_noise)
                action = discrete_action_from_angle(ang + np.random.normal(scale=0.5))
            else:
                action = random.randint(0,3)
        elif phase == "DANCE":
            # stand still for DANCE_STEPS then set heading
            if steps_in_phase >= DANCE_STEPS:
                readings = [sample_noisy_compass(pos, sigma=compass_noise) for _ in range(N_COMPASS_SAMPLES)]
                ssum = sum(math.sin(a) for a in readings)
                csum = sum(math.cos(a) for a in readings)
                target_heading = math.atan2(ssum, csum)
                phase = "ROLL"
                steps_in_phase = 0
                roll_start_idx = len(path)  # next movement index
            else:
                action = None  # stay/spin
        elif phase == "ROLL":
            cand_action = discrete_action_from_angle(target_heading)
            order = [cand_action, (cand_action+1)%4, (cand_action-1)%4, (cand_action+2)%4]
            for a in order:
                nr, nc = pos[0] + ACTION_TO_DELTA[a][0], pos[1] + ACTION_TO_DELTA[a][1]
                if is_valid(maze, nr, nc):
                    action = a
                    break
            if action is None:
                action = random.randint(0,3)

        # execute
        if action is not None:
            next_pos, collided = step_env(maze, pos, action)
            pos = next_pos
            path.append(pos)
            if pos == GOAL:
                return {"path": path, "reached": True, "roll_start": roll_start_idx}
        steps_in_phase += 1
    return {"path": path, "reached": False, "roll_start": roll_start_idx}

# Hybrid: GEO default, fallback to EGO when blocked for short time
FALLBACK_STEPS = 6

def hybrid_policy_episode(maze, max_steps=500, compass_noise=0.08):
    pos = START
    path = [pos]
    phase = "SEARCH"
    steps_in_phase = 0
    target_heading = None
    fallback_counter = 0
    roll_start_idx = None

    for step in range(max_steps):
        if np.linalg.norm(grid_to_world(pos) - GOAL_WORLD) <= DETECTION_RADIUS:
            if phase == "SEARCH":
                phase = "DANCE"
                steps_in_phase = 0

        action = None
        if phase == "SEARCH":
            if random.random() < 0.2:
                ang = sample_noisy_compass(pos, sigma=compass_noise)
                action = discrete_action_from_angle(ang + np.random.normal(scale=0.5))
            else:
                action = random.randint(0,3)
        elif phase == "DANCE":
            if steps_in_phase >= DANCE_STEPS:
                readings = [sample_noisy_compass(pos, sigma=compass_noise) for _ in range(N_COMPASS_SAMPLES)]
                ssum = sum(math.sin(a) for a in readings)
                csum = sum(math.cos(a) for a in readings)
                target_heading = math.atan2(ssum, csum)
                phase = "ROLL"
                steps_in_phase = 0
                fallback_counter = 0
                roll_start_idx = len(path)
            else:
                action = None
        elif phase == "ROLL":
            cand = discrete_action_from_angle(target_heading)
            nr, nc = pos[0] + ACTION_TO_DELTA[cand][0], pos[1] + ACTION_TO_DELTA[cand][1]
            if not is_valid(maze, nr, nc):
                # trigger fallback
                fallback_counter = FALLBACK_STEPS
            if fallback_counter > 0:
                action = ego_policy_step(maze, pos)
                fallback_counter -= 1
            else:
                order = [cand, (cand+1)%4, (cand-1)%4, (cand+2)%4]
                for a in order:
                    nr2, nc2 = pos[0] + ACTION_TO_DELTA[a][0], pos[1] + ACTION_TO_DELTA[a][1]
                    if is_valid(maze, nr2, nc2):
                        action = a
                        break
                if action is None:
                    action = random.randint(0,3)

        if action is not None:
            next_pos, collided = step_env(maze, pos, action)
            pos = next_pos
            path.append(pos)
            if pos == GOAL:
                return {"path": path, "reached": True, "roll_start": roll_start_idx}
        steps_in_phase += 1
    return {"path": path, "reached": False, "roll_start": roll_start_idx}

# --------------------------
# Maze perturbation utility
# --------------------------
def make_maze_variant(base_maze, rng_seed=None, flip_prob=0.03):
    rng = np.random.RandomState(None if rng_seed is None else rng_seed)
    m = base_maze.copy()
    for r in range(m.shape[0]):
        for c in range(m.shape[1]):
            if (r,c) in [START, GOAL]:
                continue
            if rng.rand() < flip_prob:
                m[r,c] = 1 - m[r,c]
    m[START[0], START[1]] = 0
    m[GOAL[0], GOAL[1]] = 0
    return m

# --------------------------
# Metrics helpers
# --------------------------
def compute_path_length(path):
    return len(path) - 1

def compute_straightness_index(path, roll_start_idx):
    # straightness = net displacement / path length (during roll)
    if roll_start_idx is None or roll_start_idx >= len(path)-1:
        return np.nan
    roll_path = path[roll_start_idx:]
    net_disp = np.linalg.norm(np.array(grid_to_world(roll_path[-1])) - np.array(grid_to_world(roll_path[0])))
    roll_len = len(roll_path) - 1
    if roll_len <= 0:
        return np.nan
    return float(net_disp / roll_len)

# --------------------------
# Visualization helpers
# --------------------------
def plot_maze_with_path(maze, path, filename, title="Trajectory"):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title(title)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if maze[r,c] == 1:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor='lightgray'))
    ax.add_patch(plt.Rectangle((START[1], START[0]), 1, 1, facecolor='blue', label='Start'))
    ax.add_patch(plt.Rectangle((GOAL[1], GOAL[0]), 1, 1, facecolor='green', label='Goal'))
    if path:
        px = [c + 0.5 for r,c in path]
        py = [r + 0.5 for r,c in path]
        ax.plot(px, py, marker='o', color='red', linewidth=2, markersize=3)
    ax.set_xlim(0, GRID_SIZE); ax.set_ylim(0, GRID_SIZE); ax.set_aspect('equal'); ax.invert_yaxis()
    ax.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_bar_with_error(labels, means, stds, ylabel, filename, title=""):
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.savefig(filename)
    plt.close()

# --------------------------
# Experiment orchestration
# --------------------------
def run_experiments():
    # settings
    densities = [0.00, 0.03, 0.07]   # flip_prob for make_maze_variant
    noise_levels = [0.0, 0.05, 0.1]  # compass noise
    episodes_per_setting = 30

    policies = {
        "EGO": lambda m, noise: ego_policy_episode(m),
        "GEO": lambda m, noise: geo_policy_episode(m, compass_noise=noise),
        "HYB": lambda m, noise: hybrid_policy_episode(m, compass_noise=noise),
    }

    results = []

    # precompute A* oracle for each maze variant
    maze_list = []
    for d in densities:
        m = make_maze_variant(BASE_MAZE, rng_seed=SEED + int(d*1000), flip_prob=d)
        maze_list.append((d, m))

    astar_lengths = {}
    for d, m in maze_list:
        L = astar_shortest_path_length(m, START, GOAL)
        astar_lengths[d] = L

    total_runs = len(densities) * len(noise_levels) * episodes_per_setting * len(policies)
    print(f"Starting experiments: {total_runs} episodes total (approx).")

    run_count = 0
    for d, m in maze_list:
        L_opt = astar_lengths[d]
        for noise in noise_levels:
            for policy_name, policy_fn in policies.items():
                for ep in range(episodes_per_setting):
                    run_count += 1
                    if run_count % 50 == 0:
                        print(f"... episode {run_count}/{total_runs}")
                    res = policy_fn(m, noise)
                    path = res.get("path", [])
                    reached = bool(res.get("reached", False))
                    roll_start = res.get("roll_start", None)
                    path_len = compute_path_length(path)
                    straightness = compute_straightness_index(path, roll_start)
                    # path efficiency wrt A*
                    if L_opt is None:
                        path_eff = np.nan
                    else:
                        path_eff = float(L_opt) / float(path_len) if path_len>0 else np.nan
                    results.append({
                        "policy": policy_name,
                        "density": d,
                        "noise": noise,
                        "episode": ep,
                        "reached": int(reached),
                        "path_len": path_len,
                        "path_eff": path_eff,
                        "straightness": straightness,
                        "astar_len": (L_opt if L_opt is not None else -1)
                    })
    # save CSV-like JSON for clarity
    with open(RESULTS_CSV, "w") as fh:
        # write header
        keys = list(results[0].keys())
        fh.write(",".join(keys) + "\n")
        for r in results:
            fh.write(",".join(str(r[k]) if r[k] is not None else "" for k in keys) + "\n")
    print(f"Saved results to {RESULTS_CSV}")

    # Summaries & plots
    summary = {}
    for d in densities:
        for noise in noise_levels:
            for policy_name in policies.keys():
                subset = [r for r in results if r["policy"]==policy_name and r["density"]==d and r["noise"]==noise]
                if not subset:
                    continue
                succs = [s["reached"] for s in subset]
                effs = [s["path_eff"] for s in subset if not math.isnan(s["path_eff"])]
                strs = [s["straightness"] for s in subset if s["straightness"] is not None and not math.isnan(s["straightness"])]
                summary_key = f"{policy_name}_d{d}_n{noise}"
                summary[summary_key] = {
                    "n": len(subset),
                    "succ_mean": float(np.mean(succs)),
                    "succ_std": float(np.std(succs)),
                    "path_eff_mean": float(np.nanmean(effs)) if effs else float('nan'),
                    "path_eff_std": float(np.nanstd(effs)) if effs else float('nan'),
                    "straight_mean": float(np.nanmean(strs)) if strs else float('nan'),
                    "straight_std": float(np.nanstd(strs)) if strs else float('nan')
                }

    # example plots: success rate across policies for each density & noise=0.0
    for d in densities:
        for noise in [0.0, 0.1]:
            labels = []
            succ_means = []
            succ_stds = []
            eff_means = []
            eff_stds = []
            for policy_name in policies.keys():
                key = f"{policy_name}_d{d}_n{noise}"
                if key in summary:
                    s = summary[key]
                    labels.append(policy_name)
                    succ_means.append(s["succ_mean"])
                    succ_stds.append(s["succ_std"])
                    eff_means.append(s["path_eff_mean"])
                    eff_stds.append(s["path_eff_std"])
            if labels:
                plot_bar_with_error(labels, succ_means, succ_stds,
                                    ylabel="Success rate",
                                    filename=os.path.join(OUTPUT_DIR, f"succ_rate_d{d}_n{noise}.png"),
                                    title=f"Success rate (density={d}, noise={noise})")
                plot_bar_with_error(labels, eff_means, eff_stds,
                                    ylabel="Path efficiency (A*/agent)",
                                    filename=os.path.join(OUTPUT_DIR, f"path_eff_d{d}_n{noise}.png"),
                                    title=f"Path efficiency (density={d}, noise={noise})")

    # plot example trajectories: choose first successful episode per policy/density/noise if any
    plot_count = 0
    for d, m in maze_list:
        for noise in [0.0, 0.1]:
            for policy_name in policies.keys():
                subset = [r for r in results if r["policy"]==policy_name and r["density"]==d and r["noise"]==noise and r["reached"]==1]
                if subset:
                    # pick first
                    ep = subset[0]["episode"]
                    # re-run that episode to get path (deterministic due to seed? not strictly deterministic; but good enough)
                    if policy_name == "EGO":
                        res = ego_policy_episode(m)
                    elif policy_name == "GEO":
                        res = geo_policy_episode(m, compass_noise=noise)
                    else:
                        res = hybrid_policy_episode(m, compass_noise=noise)
                    plot_maze_with_path(m, res["path"], os.path.join(OUTPUT_DIR, f"traj_{policy_name}_d{d}_n{noise}.png"),
                                       title=f"{policy_name} traj (d={d}, n={noise})")
                    plot_count += 1
                    if plot_count >= 9:
                        break
            if plot_count >= 9:
                break
        if plot_count >= 9:
            break

    # save summary JSON
    with open(SUMMARY_JSON, "w") as fh:
        json.dump({"summary": summary, "astar_lengths": astar_lengths}, fh, indent=2)
    print(f"Saved summary to {SUMMARY_JSON}")

    print("Done experiments and plotting.")


if __name__ == "__main__":
    run_experiments()
