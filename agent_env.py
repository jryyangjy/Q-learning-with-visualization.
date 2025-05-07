# agent_env.py（环境定义）
"""本程序实现的是对于多智能体强化学习的环境定义，"""
import numpy as np
import matplotlib.pyplot as plt

class GridEnv:
    def __init__(self, obstacle_rate=0.15, grid_size=15, agent_num=3):

        self.max_steps = 100
        self.grid_size = grid_size
        self.agent_num = agent_num
        self.obstacles = []
        self.positions = {}
        self.paths = {}
        self.all_actions = [(0, 1), (0, -1), (-1, 0), (1, 0), (0, 0)]
  
        self.target = (self.grid_size // 2, self.grid_size // 2)
            
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) == self.target:
                    continue
                if np.random.rand() < obstacle_rate:
                   self.obstacles.append((i, j))
        self.obstacles = list(set(self.obstacles))

        for agent_i in range(self.agent_num):
            while True:
                position = (np.random.randint(0, self.grid_size),
                            np.random.randint(0, self.grid_size))
                if position not in self.obstacles and position != self.target and position not in self.positions.values():
                    self.positions[agent_i] = position
                    break

        self.paths = {agent_i: [self.positions[agent_i]] for agent_i in range(self.agent_num)}
        self.initial_positions = self.positions.copy()

    def reset(self):

        self.positions = self.initial_positions.copy()
        self.paths = {agent_i: [self.positions[agent_i]] for agent_i in range(self.agent_num)}

    def get_state(self, agent_i):

        return self.positions[agent_i]
    
    def move_agent(self, agent_i, action):

        x, y = self.get_state(agent_i)
        dx, dy = self.all_actions[action]
        new_x, new_y = x + dx, y + dy
        if (new_x < 0 or new_x >= self.grid_size or
            new_y < 0 or new_y >= self.grid_size or
            (new_x, new_y) in self.obstacles):
            return False
        else:
            for id, location in self.positions.items():
                if id != agent_i and location == (new_x, new_y) and location != self.target:
                    return False
            self.positions[agent_i] = (new_x, new_y)
            self.paths[agent_i].append((new_x, new_y))
            return True
        
    def get_reward(self, agent_i, pre_distance):
            
        x, y = self.positions[agent_i]
        target_x, target_y = self.target
        current_distance = abs(x - target_x) + abs(y - target_y)
        reward = 0.0
        if current_distance == 0:
            reward += 50.0
        else:
            reward += (pre_distance - current_distance) * 2
            if len(self.paths[agent_i]) > 2*current_distance:
                reward -= 1
            reward -= 1
        return current_distance, reward
    
    def _draw_map(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure
    
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        ax.set_aspect('equal')

        if self.obstacles:
            xs, ys = zip(*self.obstacles)
            ax.scatter(xs, ys, marker='s', s=100, color='black', label='Obstacle')

        tx, ty = self.target
        ax.scatter(tx, ty, marker='*', s=200, color='gold', label='Target')

        for agent_i, (x, y) in self.positions.items():
            ax.scatter(x, y, s=150, label=f'Agent {agent_i}')
            ax.text(x, y, str(agent_i), color='white',
                    ha='center', va='center', fontsize=12)

        ax.legend(loc='upper right')
        ax.set_title('Current Environment')
        plt.show()
        return ax  