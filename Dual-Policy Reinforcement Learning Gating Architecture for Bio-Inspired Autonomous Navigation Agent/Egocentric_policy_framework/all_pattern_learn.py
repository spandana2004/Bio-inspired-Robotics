import numpy as np
from itertools import product
import csv
import math

gamma = 0.9
Q_table = {}
Vector_table = {} # Renamed from Angle_table to Vector_table

# -------------------------------------------------
#  Rewards based on wall presence in 3×3 patch
# -------------------------------------------------
def compute_reward(pattern, action):
    # Pattern indices:
    # 0 1 2
    # 3 4 5
    # 6 7 8
    up    = pattern[1]
    down  = pattern[7]
    left  = pattern[3]
    right = pattern[5]

    # -75 for hitting a wall, -1 for free space
    if action == 0: return -75 if up    == 1 else -1   # up
    if action == 1: return -75 if down  == 1 else -1   # down
    if action == 2: return -75 if left  == 1 else -1   # left
    if action == 3: return -75 if right == 1 else -1   # right
    return 0

# -------------------------------------------------
#  Compute Q-values for ALL 512 patterns
# -------------------------------------------------
def compute_optimal_Q():
    all_patterns = [tuple(p) for p in product([0,1], repeat=9)]

    for pattern in all_patterns:
        q_vals = []
        for action in range(4):
            reward = compute_reward(pattern, action)
            # Simple Q calculation based on immediate reward
            Q = reward / (1 - gamma)
            q_vals.append(Q)
        Q_table[pattern] = q_vals

    print("Computed optimal Q-table for all 512 patterns.")

# -------------------------------------------------
#  Compute Vector Components (X, Y)
# -------------------------------------------------
def compute_vectors():
    for pattern, q_vals in Q_table.items():
        q_up, q_down, q_left, q_right = q_vals

        # Calculate net forces
        # If Right is better than Left, X is positive
        # If Up is better than Down, Y is positive
        x_component = q_right - q_left
        y_component = q_up - q_down

        # We store the raw components.
        # If x=0 and y=0, it means the agent is neutral (safe space or surrounded).
        Vector_table[pattern] = (x_component, y_component)

    print("Computed vector components (X, Y).")

# -------------------------------------------------
#  Save to CSV
# -------------------------------------------------
def save_vector_csv(filename="ego_vector_table.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # Updated Header: pattern, x, y
        writer.writerow(["pattern_9_bits", "ego_x_component", "ego_y_component"])

        for pattern in Vector_table.keys():
            # Convert tuple (0,0,1...) to string "0 0 1..."
            pattern_str = " ".join(str(x) for x in pattern)

            vx, vy = Vector_table[pattern]

            # Write row. Formatting to 4 decimal places for cleanliness.
            writer.writerow([pattern_str, f"{vx:.4f}", f"{vy:.4f}"])

    print(f"Saved: {filename}")

# -------------------------------------------------
#  MAIN
# -------------------------------------------------
if __name__ == "__main__":
    compute_optimal_Q()
    compute_vectors()

    # You can change the path here
    save_vector_csv("ego_vector_table.csv")