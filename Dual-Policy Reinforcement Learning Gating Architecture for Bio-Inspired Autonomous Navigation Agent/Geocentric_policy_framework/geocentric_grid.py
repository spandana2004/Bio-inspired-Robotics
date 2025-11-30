import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

# Define the maze layout
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

start_pos = (1, 1)
goal_pos = (14, 14)

# --- Function for Angle Calculation ---
def calculate_angles_to_goal(maze, goal):
    """Calculates the angle in degrees from each cell to the goal using a Cartesian standard."""
    rows, cols = maze.shape
    angles = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            # The y-axis for atan2 should be positive when moving "up" (decreasing row index).
            delta_y = -(goal[0] - r)
            delta_x = goal[1] - c
            angle_rad = math.atan2(delta_y, delta_x)
            angles[r, c] = math.degrees(angle_rad)
    return angles

# --- Function for Discretized Arrow Plot (MODIFIED) ---
def plot_discretized_arrows(maze, start, goal, angles, filename="discretized_arrows_grid_labels.png"):
    rows, cols = maze.shape
    fig, ax = plt.subplots(figsize=(12, 12))
    colors = {0: 'white', 1: 'red'}
    for r in range(rows):
        for c in range(cols):
            # Plotting coordinates (x, y)
            x, y = c, rows - 1 - r
            cell_color = colors[maze[r,c]]
            if (r,c) == start: cell_color = 'green'
            if (r,c) == goal: cell_color = 'blue'
            rect = mpatches.Rectangle((x, y), 1, 1, facecolor=cell_color, edgecolor='black')
            ax.add_patch(rect)
            if maze[r,c] == 0 and (r,c) != goal:
                angle_rad = math.radians(angles[r,c])
                u, v = math.cos(angle_rad), math.sin(angle_rad) # u=dx, v=dy
                ax.arrow(x + 0.5, y + 0.5, u * 0.4, v * 0.4, head_width=0.3, head_length=0.3, fc='black', ec='black')

    # --- CHANGE: Customize Ticks and Labels ---
    # Set axis limits
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)

    # Set x-axis ticks to be in the center of each cell and label them with column index
    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_xticklabels(np.arange(cols))

    # Set y-axis ticks to be in the center of each cell and label them with row index
    # The y-axis is inverted in the plot (0 is at the bottom), so we reverse the labels.
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_yticklabels(np.arange(rows - 1, -1, -1))

    ax.set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--')
    plt.title(f"Discretized Arrow Directions to Goal at {goal}", fontsize=16)
    # --- CHANGE: Updated Axis Labels ---
    plt.xlabel("Column Number")
    plt.ylabel("Row Number")
    plt.savefig(filename)
    print(f"Discretized arrow plot saved to '{filename}'")


# --- Main Execution ---
if __name__ == '__main__':
    print("Calculating angles using Cartesian standard...")
    angles = calculate_angles_to_goal(MAZE, goal_pos)

    angle_filename = 'angles_to_goal.csv'
    np.savetxt(angle_filename, angles, delimiter=',', fmt='%.2f')
    print(f"Angle values saved to '{angle_filename}'")

    plot_discretized_arrows(MAZE, start_pos, goal_pos, angles)
    plt.show()