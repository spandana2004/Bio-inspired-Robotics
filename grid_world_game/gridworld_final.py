import mujoco
import mujoco.viewer
import numpy as np
import time
import random

# --- 10x10 maze from your image ---
# 0 = free cell (walkable), 1 = hazard/wall (red)
# Start = (0,0) [top-left], Goal = (0,9) [top-right]
MAZE = [
 [0,0,1,0,0,0,0,0,0,0],  # row 0
 [1,0,1,0,1,0,1,0,1,0],  # row 1
 [1,0,1,0,1,0,1,0,1,0],  # row 2
 [1,0,0,0,1,1,1,0,0,0],  # row 3
 [1,1,1,0,0,0,0,0,1,0],  # row 4
 [0,0,0,0,1,1,1,0,1,0],  # row 5
 [0,1,1,0,0,0,1,0,1,0],  # row 6
 [0,0,0,0,1,0,1,0,1,0],  # row 7
 [1,1,1,1,1,0,0,0,1,0],  # row 8
 [0,0,0,0,0,0,0,0,1,0],  # row 9
]

GRID_SIZE = 10
START_POS_X = -4.5
START_POS_Y =  4.5
CELL_SIZE   =  1.0

def generate_hazards_from_maze(maze):
    """Create hazard geoms for all cells with value 1."""
    hazards = []
    n = len(maze)
    for r in range(n):
        for c in range(n):
            if maze[r][c] == 1:
                x = START_POS_X + c * CELL_SIZE
                y = START_POS_Y - r * CELL_SIZE
                hazards.append(
                    f'<geom name="hazard_{r}_{c}" type="box" pos="{x} {y} .05" '
                    f'size=".5 .5 .05" rgba="1 0.2 0.2 1" contype="1" conaffinity="1"/>'
                )
    return "\n".join(hazards)

hazard_xml = generate_hazards_from_maze(MAZE)

# MJCF model string for 10x10 grid world with maze hazards
GRID_MJCF = f"""
<mujoco>
  <option gravity="0 0 0" integrator="RK4"/>

  <asset>
    <texture type="2d" name="groundplane" builtin="checker"
             rgb1=".8 .8 .8" rgb2=".9 .9 .9" width="500" height="500"/>
    <material name="groundplane" texture="groundplane" texrepeat="20 20"/>
  </asset>

  <worldbody>
    <light pos="0 0 10"/>
    <geom name="ground" type="plane" size="5 5 0.1" material="groundplane"/>

    <!-- Agent (blue start, smaller cube) -->
<body name="agent" pos="-4.5 4.5 0.2">
  <joint type="free" limited="false"/>
  <geom name="agent_geom" type="box" size=".3 .3 .2" rgba="0.1 0.5 1.0 1"/>
</body>

<!-- Goal (green, full cell) -->
<geom name="goal" type="box" pos="4.5 4.5 0.05" size=".5 .5 .05" rgba="0.1 1.0 0.2 1"/>


    <!-- Maze Hazards -->
    {hazard_xml}

  </worldbody>
</mujoco>
"""

class GridWorldEnv:
    """
    A MuJoCo-based environment for a 10x10 grid world.
    Discrete logic, MuJoCo for visualization.
    """
    def __init__(self, model_xml):
        self.model = mujoco.MjModel.from_xml_string(model_xml)
        self.data = mujoco.MjData(self.model)
        self.agent_body_id = self.model.body('agent').id

        self.grid_size = (GRID_SIZE, GRID_SIZE)
        self.start_pos_x = START_POS_X
        self.start_pos_y = START_POS_Y
        self.cell_size = CELL_SIZE

        # States, actions, rewards
        self.goal_state = (0, 9)  # top-right
        self.hazard_states = self._get_hazard_states()
        self.action_space = [0, 1, 2, 3]  # 0:Down, 1:Up, 2:Left, 3:Right
        self.state_space = [(r, c) for r in range(self.grid_size[0]) for c in range(self.grid_size[1])]

        self.rewards = {
            "goal": 10.0,
            "hazard": -10.0,
            "step": -0.1
        }

        self.q_table = np.zeros((self.grid_size[0], self.grid_size[1], len(self.action_space)))

    def _get_hazard_states(self):
        """Collect hazard cell coordinates from MJCF geoms named 'hazard_*'."""
        hazards = set()
        for i in range(self.model.ngeom):
            name = self.model.geom(i).name
            if name and "hazard_" in name:
                pos = self.model.geom_pos[i]
                col = int(round((pos[0] - self.start_pos_x) / self.cell_size))
                row = int(round((self.start_pos_y - pos[1]) / self.cell_size))
                hazards.add((row, col))
        return hazards

    def _get_pos_from_state(self, state):
        row, col = state
        x = self.start_pos_x + col * self.cell_size
        y = self.start_pos_y - row * self.cell_size
        return x, y, 0.1

    def reset(self):
        """Reset the agent to the starting position (0,0)."""
        self.state = (0, 0) # Always start at the top-left corner
        x, y, z = self._get_pos_from_state(self.state)
        self.data.qpos[0:3] = [x, y, z]
        mujoco.mj_forward(self.model, self.data)
        return self.state

    def step(self, action):
        row, col = self.state
        if action == 0:   # Down
            row = min(row + 1, self.grid_size[0] - 1)
        elif action == 1: # Up
            row = max(row - 1, 0)
        elif action == 2: # Left
            col = max(col - 1, 0)
        elif action == 3: # Right
            col = min(col + 1, self.grid_size[1] - 1)

        new_state = (row, col)

        if new_state in self.hazard_states:
            reward = self.rewards["hazard"]
            done = True
        else:
            self.state = new_state
            if self.state == self.goal_state:
                reward = self.rewards["goal"]
                done = True
            else:
                reward = self.rewards["step"]
                done = False

        x, y, z = self._get_pos_from_state(self.state)
        self.data.qpos[0:3] = [x, y, z]
        mujoco.mj_forward(self.model, self.data)
        return self.state, reward, done

    def get_q_table(self):
        return self.q_table


def q_learning_agent():
    env = GridWorldEnv(GRID_MJCF)
    q_table = env.get_q_table()

    # Q-Learning Parameters (kept from your logic)
    training_episodes = 2000
    max_steps_per_episode = 100
    learning_rate = 0.1
    discount_rate = 0.99
    epsilon = 0.2  # exploration

    print("--- Starting Q-Learning Training ---")

    for episode in range(training_episodes):
        state = env.reset() # This will now always return (0,0)
        done = False

        for _ in range(max_steps_per_episode):
            # Epsilon-greedy
            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.action_space)
            else:
                action = int(np.argmax(q_table[state[0], state[1]]))

            new_state, reward, done = env.step(action)

            # Q-learning update
            old_value = q_table[state[0], state[1], action]
            next_max = np.max(q_table[new_state[0], new_state[1]])
            q_table[state[0], state[1], action] = (1 - learning_rate) * old_value + \
                learning_rate * (reward + discount_rate * next_max)

            state = new_state
            if done:
                break

        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}/{training_episodes} completed.")

    print("\n--- Training Finished ---")

    # --- Demonstration (slow) ---
    print("\n--- Running Demonstration with Learned Policy ---")
    state = env.reset() # Start demo from the fixed reset position (0,0)

    done = False
    total_reward = 0
    step_count = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 15
        viewer.cam.elevation = -60
        viewer.cam.azimuth = 90

        while viewer.is_running() and not done and step_count < max_steps_per_episode:
            step_count += 1
            action = int(np.argmax(q_table[state[0], state[1]]))
            new_state, reward, done = env.step(action)
            total_reward += reward
            state = new_state

            print(f"Step: {step_count}, State: {state}, "
                  f"Action: {['Down','Up','Left','Right'][action]}, Reward: {reward:.2f}")

            viewer.sync()
            time.sleep(1.0)  # slowed down

    print("\n--- Demonstration Finished ---")
    if state == env.goal_state:
        print(f"Goal Reached in {step_count} steps! Total Reward: {total_reward:.2f}")
    else:
        print(f"Failed to reach the goal. Total Reward: {total_reward:.2f}")


if __name__ == "__main__":
    q_learning_agent()