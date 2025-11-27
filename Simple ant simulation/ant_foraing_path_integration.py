import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import random
from mujoco import MjModel, MjData

# Constants
FOOD_CENTER = np.array([5.0, 6.0])         # Food box center
FOOD_HALF_SIZE = 1.0                       # Half-side length of food box
INITIAL_POS = np.array([0.0, 0.0])
STEP_LENGTH = 0.5
BOX_LIMIT = 10  

# Load MuJoCo model
model = mujoco.MjModel.from_xml_string(""" 
<mujoco model="ant_with_marker">
    <compiler angle="radian" coordinate="local" />
    <option gravity="0 0 -9.81" timestep="0.01" />

    <default>
        <joint damping="1"/>
        <geom friction="1 0.5 0.5" density="300" margin="0.002" />
    </default>

    <worldbody>
        <!-- Start marker -->
        <geom name="start_marker" type="sphere" pos="0 0 0.05" size="0.2" rgba="0 0 1 1" contype="0" conaffinity="0"/>

        <!-- Food marker -->
        <geom name="food_marker" type="box" pos="5.0 6.0 0.05" size="1.0 1.0 0.05" rgba="0 1 0 1" contype="0" conaffinity="0"/>

        <!-- Ground -->
        <geom name="floor" type="plane" size="10 10 0.1" rgba="0.9 0.9 0.9 1"/>

        <!-- Ant body -->
        <body name="torso" pos="0 0 0.1">
            <joint name="root" type="free"/>
            <geom name="torso_geom" type="sphere" size="0.25" rgba="0.8 0.4 0.4 1"/>
        </body>
    </worldbody>

    <actuator />
</mujoco>
""")

data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, data)

# Helper functions
def quat_from_yaw(yaw):
    return np.array([0, 0, math.sin(yaw / 2), math.cos(yaw / 2)])

def random_turn():
    return random.uniform(0, 360)

# Set initial pose
data.qpos[:3] = [*INITIAL_POS, 0.1]
angle_deg = 0
data.qpos[3:7] = quat_from_yaw(np.radians(angle_deg))

path = [list(INITIAL_POS)]
directions = []  # Stores step vectors (with magnitude)

# ---------- Outbound Search Phase ----------
print("Searching for food...")

while True:
    px, py = data.qpos[0], data.qpos[1]
    if (abs(px - FOOD_CENTER[0]) <= FOOD_HALF_SIZE) and (abs(py - FOOD_CENTER[1]) <= FOOD_HALF_SIZE):
        print("Food reached!")
        dx = FOOD_CENTER[0] - px
        dy = FOOD_CENTER[1] - py
        data.qpos[3:7] = quat_from_yaw(math.atan2(dy, dx))
        path.append([px, py])
        mujoco.mj_step(model, data)
        viewer.sync()
        break

    # Turn and move
    angle_deg = random_turn()
    angle_rad = np.radians(angle_deg)
    step_vec = STEP_LENGTH * np.array([np.cos(angle_rad), np.sin(angle_rad)])
    new_x = data.qpos[0] + step_vec[0]
    new_y = data.qpos[1] + step_vec[1]

    # Wall bounce logic
    if abs(new_x) > BOX_LIMIT or abs(new_y) > BOX_LIMIT:
        print("Bounce: hitting wall")
        angle_deg = (angle_deg + random.uniform(90, 270)) % 360
        continue

    # Apply movement
    directions.append(step_vec)
    data.qpos[0] = new_x
    data.qpos[1] = new_y
    data.qpos[2] = 0.1
    data.qpos[3:7] = quat_from_yaw(angle_rad)
    path.append([data.qpos[0], data.qpos[1]])
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.05)

# ---------- Return Using Step Vector Sum ----------
print("Returning to start using reversed step vector sum...")

if directions:
    total_vector = np.sum(directions, axis=0)  # includes magnitude
    return_vector = -total_vector
    distance = np.linalg.norm(return_vector)

    if distance > 0:
        direction = return_vector / distance
        yaw = math.atan2(direction[1], direction[0])
        quat = quat_from_yaw(yaw)
        num_steps = int(distance / STEP_LENGTH)

        for _ in range(num_steps):
            data.qpos[0] += STEP_LENGTH * direction[0]
            data.qpos[1] += STEP_LENGTH * direction[1]
            data.qpos[2] = 0.1
            data.qpos[3:7] = quat
            path.append([data.qpos[0], data.qpos[1]])
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.05)

        # Final precise step
        final_dx = INITIAL_POS[0] - data.qpos[0]
        final_dy = INITIAL_POS[1] - data.qpos[1]
        final_dist = np.linalg.norm([final_dx, final_dy])

        if final_dist > 1e-4:
            final_dir = np.array([final_dx, final_dy]) / final_dist
            yaw = math.atan2(final_dir[1], final_dir[0])
            quat = quat_from_yaw(yaw)

            data.qpos[0] += final_dx
            data.qpos[1] += final_dy
            data.qpos[2] = 0.1
            data.qpos[3:7] = quat
            path.append([data.qpos[0], data.qpos[1]])
            mujoco.mj_step(model, data)
            viewer.sync()

print("Return complete.")
viewer.close()

# ---------- Plotting ----------
path = np.array(path)
outbound_path = path[:len(directions)+1]  # initial + all outbound moves
return_path = path[len(directions)+1:]    # return moves

fig, ax = plt.subplots()
ax.plot(outbound_path[:, 0], outbound_path[:, 1], 'b-', label="Path to Food")
ax.plot(return_path[:, 0], return_path[:, 1], 'r-', label="Return Path")
ax.plot(*INITIAL_POS, 'bo', markersize=10, label="Start")
ax.plot(*FOOD_CENTER, 'g*', markersize=15, label="Food (Center)")

# Draw food square
square = plt.Rectangle((FOOD_CENTER[0] - FOOD_HALF_SIZE, FOOD_CENTER[1] - FOOD_HALF_SIZE),
                       2 * FOOD_HALF_SIZE, 2 * FOOD_HALF_SIZE,
                       linewidth=1, edgecolor='g', facecolor='g', alpha=0.3)
ax.add_patch(square)

ax.set_title("Ant Movement with Wall Bounce and Return")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.legend()
ax.set_aspect('equal', 'box')
ax.grid(True)
plt.show()
