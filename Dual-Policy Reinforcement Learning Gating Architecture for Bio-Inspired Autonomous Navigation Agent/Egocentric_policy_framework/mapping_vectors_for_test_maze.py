import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
import math

# --- Part 1: Load the Vector Table from the existing CSV ---

def load_vector_table_from_csv(filename="ego_vector_table.csv"):
    """
    Loads the pattern-to-vector mappings from the specified CSV file.
    """
    vector_map = {}
    try:
        with open(filename, "r") as f:
            reader = csv.reader(f)
            next(reader) # Skip the header row
            for row in reader:
                pattern_str = row[0]
                x_comp = float(row[1])
                y_comp = float(row[2])
                vector_map[pattern_str] = (x_comp, y_comp)
        print(f"Successfully loaded {len(vector_map)} vectors from '{filename}'")
        return vector_map
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        print("Please make sure the CSV file is in the same directory as the script.")
        return None

# --- Part 2: Map the Vectors onto the Specific Maze ---

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

def map_vectors_to_maze(maze, vector_map):
    """
    Creates a 16x16 grid with vectors from the map corresponding to each cell's pattern.
    """
    rows, cols = maze.shape
    maze_vector_field = np.full((rows, cols), None)

    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == 1: # Wall
                maze_vector_field[r, c] = (np.nan, np.nan)
            else: # Open path
                # Extract 3x3 patch, treating boundaries as walls (1)
                patch = np.ones((3, 3), dtype=int)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            patch[dr + 1, dc + 1] = maze[nr, nc]

                # Flatten patch and create key for lookup
                pattern_key = " ".join(map(str, patch.flatten()))

                # Assign vector from map, default to (0,0) if somehow not found
                maze_vector_field[r, c] = vector_map.get(pattern_key, (0, 0))

    return maze_vector_field

# --- Part 3: Save the Maze Vector Field to a new CSV ---

def save_mapped_vectors_to_csv(vector_field, filename="maze_vector_field.csv"):
    """
    Saves the 16x16 vector grid to a CSV file.
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for row in vector_field:
            # Format each cell as a tuple string (vx, vy) or (nan, nan)
            writer.writerow([f"({v[0]:.4f}, {v[1]:.4f})" if isinstance(v, tuple) else "(nan, nan)" for v in row])
    print(f"Mapped maze vectors saved to '{filename}'")

# --- Part 4: Plot the Final Vector Field on the Maze ---

def plot_vector_field(maze, vector_field, filename="maze_egocentric_plot.png"):
    """
    Generates and saves a plot of the maze with the calculated vector arrows.
    """
    rows, cols = maze.shape
    fig, ax = plt.subplots(figsize=(12, 12))
    colors = {0: 'white', 1: 'red'}

    for r in range(rows):
        for c in range(cols):
            plot_x, plot_y = c, rows - 1 - r # Invert row for correct plotting

            # Draw the maze cell
            rect = mpatches.Rectangle((plot_x, plot_y), 1, 1, facecolor=colors[maze[r, c]], edgecolor='black')
            ax.add_patch(rect)

            # Get the vector for the cell
            vx, vy = vector_field[r, c]

            # Draw arrow if the vector is valid and not (0,0)
            if not np.isnan(vx) and not (vx == 0 and vy == 0):
                norm = math.sqrt(vx**2 + vy**2)
                u, v = (vx / norm, vy / norm) if norm > 0 else (0, 0)

                ax.arrow(
                    plot_x + 0.5, plot_y + 0.5, # Arrow start
                    u * 0.4, v * 0.4,           # Arrow vector (normalized and scaled)
                    head_width=0.18, head_length=0.15, fc='black', ec='black'
                )

    # --- Formatting the plot ---
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_xticklabels(np.arange(cols))
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_yticklabels(np.arange(rows - 1, -1, -1)) # Labels from 15 down to 0
    ax.set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--')
    plt.title("Egocentric Obstacle-Avoidance Vector Field from CSV", fontsize=16)
    plt.xlabel("Column Number")
    plt.ylabel("Row Number")

    plt.savefig(filename)
    print(f"Final plot saved to '{filename}'")
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load the vector table from your CSV
    ego_vector_table = load_vector_table_from_csv("ego_vector_table.csv")

    if ego_vector_table:
        # 2. Map the loaded vectors onto the maze grid
        maze_vectors = map_vectors_to_maze(MAZE, ego_vector_table)

        # 3. Save the resulting 16x16 grid to a new CSV
        save_mapped_vectors_to_csv(maze_vectors)

        # 4. Generate and display the final plot
        plot_vector_field(MAZE, maze_vectors)