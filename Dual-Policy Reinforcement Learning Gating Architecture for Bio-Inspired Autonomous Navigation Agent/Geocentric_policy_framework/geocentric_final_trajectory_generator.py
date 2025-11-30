import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

# ------------------------------------------
# MAZE AND POSITIONS
# ------------------------------------------
MAZE = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
    [1,1,0,0,0,0,0,0,1,1,1,1,1,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1],
    [1,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1],
    [1,0,0,0,0,0,0,1,0,1,0,0,0,0,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1],
    [1,0,0,0,0,1,1,0,1,0,0,0,0,0,1,1],
    [1,1,1,0,0,1,0,1,1,0,0,0,0,0,1,1],
    [1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
])

start = (1, 1)   # row, col
goal  = (14, 14)

# ------------------------------------------
# ANGLE CALCULATION
# ------------------------------------------
def calculate_angles_to_goal(maze, goal):
    rows, cols = maze.shape
    angles = np.zeros((rows, cols))

    for r in range(rows):
        for c in range(cols):
            dy = -(goal[0] - r)  # invert y for Cartesian
            dx =  (goal[1] - c)
            angles[r, c] = math.degrees(math.atan2(dy, dx))

    return angles

angles = calculate_angles_to_goal(MAZE, goal)


# ------------------------------------------
# DISCRETIZATION RULES
# ------------------------------------------
def discretize_angle(angle):
    if -45 <= angle <= 45:
        return "R"
    if 45 < angle <= 135:
        return "U"
    if angle > 135 or angle < -135:
        return "L"
    return "D"

def compute_discrete_field(angles):
    rows, cols = angles.shape
    field = np.empty((rows, cols), dtype=str)

    for r in range(rows):
        for c in range(cols):
            field[r, c] = discretize_angle(angles[r, c])

    return field

action_field = compute_discrete_field(angles)


# ------------------------------------------
# PLOT DISCRETIZED ACTION FIELD
# ------------------------------------------
def plot_discretized_actions(maze, action_field, start, goal):
    rows, cols = maze.shape
    fig, ax = plt.subplots(figsize=(10,10))

    for r in range(rows):
        for c in range(cols):
            color = "white" if maze[r,c] == 0 else "black"
            if (r,c) == start: color = "blue"
            if (r,c) == goal:  color = "green"

            ax.add_patch(mpatches.Rectangle((c, r), 1, 1,
                                             facecolor=color, edgecolor="gray"))

            if maze[r,c] == 0 and (r,c) != goal:
                action = action_field[r,c]
                dx = dy = 0

                if action == "R": dx = 0.4
                if action == "L": dx = -0.4
                if action == "U": dy = -0.4
                if action == "D": dy = 0.4

                ax.arrow(c+0.5, r+0.5, dx, dy,
                         head_width=0.2, head_length=0.15, color="gray")

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)

    ax.set_xticks(np.arange(cols)+0.5)
    ax.set_xticklabels(np.arange(cols))
    ax.set_yticks(np.arange(rows)+0.5)
    ax.set_yticklabels(np.arange(rows))

    ax.set_aspect("equal")
    plt.title("Discretized Action Field")
    plt.xlabel("Column Number")
    plt.ylabel("Row Number")
    plt.show()


# ------------------------------------------
# SIMULATE TRAJECTORY
# ------------------------------------------
def next_step(pos, action):
    r, c = pos
    if action == "R": return (r, c+1)
    if action == "L": return (r, c-1)
    if action == "U": return (r-1, c)
    if action == "D": return (r+1, c)
    return pos

def simulate_path(start, goal, maze, action_field, max_steps=500):
    path = [start]
    curr = start

    for _ in range(max_steps):
        if curr == goal:
            break

        action = action_field[curr[0], curr[1]]
        nxt = next_step(curr, action)

        # block movement into walls
        if maze[nxt[0], nxt[1]] == 1:
            break

        curr = nxt
        path.append(curr)

    return path

path = simulate_path(start, goal, MAZE, action_field)


# ------------------------------------------
# PLOT TRAJECTORY
# ------------------------------------------
def plot_trajectory(maze, path, start, goal):
    fig, ax = plt.subplots(figsize=(10,10))
    rows, cols = maze.shape

    for r in range(rows):
        for c in range(cols):
            color = "white" if maze[r,c] == 0 else "black"
            if (r,c) == start: color = "blue"
            if (r,c) == goal:  color = "green"

            ax.add_patch(mpatches.Rectangle((c, r), 1, 1,
                                             facecolor=color, edgecolor="gray"))

    xs = [c+0.5 for (_,c) in path]
    ys = [r+0.5 for (r,_) in path]

    plt.plot(xs, ys, color="red", linewidth=3)

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)

    ax.set_xticks(np.arange(cols)+0.5)
    ax.set_xticklabels(np.arange(cols))
    ax.set_yticks(np.arange(rows)+0.5)
    ax.set_yticklabels(np.arange(rows))

    ax.set_aspect("equal")
    plt.xlabel("Column Number")
    plt.ylabel("Row Number")
    plt.title("Final Trajectory")
    plt.show()


# ------------------------------------------
# RUN BOTH PLOTS
# ------------------------------------------
plot_discretized_actions(MAZE, action_field, start, goal)
plot_trajectory(MAZE, path, start, goal)
