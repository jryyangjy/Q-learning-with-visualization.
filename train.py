# train.py（训练程序）
"""
本程序用于训练基于Q学习的智能体。包括最后的可视化过程
"""
from agent_env import GridEnv
from q_learning import QLearning
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import numpy as np
from collections import defaultdict

class trainer:
    def __init__(self, config):
        self.config = config
        self.env = GridEnv(grid_size=config.size, agent_num=config.agents, obstacle_rate=config.obstacles)
        self.agents = [QLearning(learning_rate=config.alpha,
                                discount_factor=config.gamma, 
                                exploration_prob=config.epsilon,
                                action_space=len(self.env.all_actions)) 
                                for _ in range(config.agents)]
        self.episodes = config.episodes
        self.rewards = []
        self.paths_history = []
    
    def calculate_manhattan_distance(self, agent_i):
 
        x, y = self.env.get_state(agent_i)
        target_x, target_y = self.env.target
        return abs(x - target_x) + abs(y - target_y)
    
    def training(self):

        for episode in range(self.episodes):
            self.env.reset()
            decay_rate = np.power(episode/self.episodes, 3)  
            for agent in self.agents:
                agent.exploration_prob = max(0.1, self.config.epsilon * (1 - decay_rate))
            pre_distance_dict = {agent_i: self.calculate_manhattan_distance(agent_i) for agent_i in range(self.config.agents)}
            episode_reward = 0.0
            for step in range(self.env.max_steps):

                done = True                    
                for agent_i in range(self.config.agents):
                    pre_distance = pre_distance_dict[agent_i]
                    current_state = self.env.get_state(agent_i)
                    possible_action = self.agents[agent_i].get_action(current_state)
                    valid_action = self.env.move_agent(agent_i, possible_action)
                    reward_i = 0.0
                    if valid_action == False:
                        current_distance = pre_distance
                        reward_i = -10.0
                    else:
                        current_distance, reward_i = self.env.get_reward(agent_i, pre_distance)
                    next_state = self.env.get_state(agent_i)
                    self.agents[agent_i].update_q_value(current_state, possible_action, reward_i, next_state)
                    pre_distance_dict[agent_i] = current_distance
                    episode_reward += reward_i
                    if current_distance > 0:
                        done = False
                if done:
                    break
            self.rewards.append(episode_reward)
            if episode % 100 == 0:
                print(f"Episode {episode} | Avg Reward: {np.mean(self.rewards[-100:]):.2f}")
                self.env._draw_map()
                self.visualize(episode)
                self.paths_history.append((
                    episode,
                    {agent_id: [tuple(pos) for pos in path]
                    for agent_id, path in self.env.paths.items()}
                ))
            

    def visualize(self, episode):

        plt.figure(figsize=(15, 6))

        ax1 = plt.subplot(121)
        self._plot_heatmap(ax1)
        ax2 = plt.subplot(122)
        self._plot_convergence(ax2)
        
        plt.tight_layout()
        plt.savefig(f"training_ep{episode}.png")
        plt.close()
        
    def _plot_heatmap(self, ax):

        ax.clear()
        heatmap = np.zeros((self.env.grid_size, self.env.grid_size))
        
        for agent_id in range(self.config.agents):
            for (x, y) in self.agents[agent_id].q_table:
                if 0 <= x < self.env.grid_size and 0 <= y < self.env.grid_size:
                    heatmap[x][y] = np.max([
                        heatmap[x][y], 
                        np.max(self.agents[agent_id].q_table[(x, y)])
                    ])
    
        vmax = np.percentile(heatmap[heatmap > 0], 95) if np.any(heatmap) else 1
        vmin = np.min(heatmap[heatmap > 0]) if np.any(heatmap) else 0
        
        im = ax.imshow(heatmap.T,
                       cmap='hot',
                       origin='lower',
                       vmin=vmin,
                       vmax=vmax)
        ax.figure.colorbar(im, ax=ax, label='Max Q-value')
        ax.set_title(f"Q-value Heatmap (Eps {self.config.epsilon})")
    
    def _plot_convergence(self, ax):

        ax.clear()
        for agent_id in range(self.config.agents):
            q_values = [np.max(values) for values in self.agents[agent_id].q_table.values()]
            ax.plot(q_values[:10000], label=f'Agent {agent_id}')
        ax.set_title("Q-value Convergence")
        ax.legend()

class PathAnimator:
    def __init__(self, env, paths_history):
        self.env = env
        self.paths_history = paths_history
        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.colors = ['red', 'blue', 'green', 'purple', 'orange']
        
    def _init_frame(self):

        self.ax.clear()
        self.env._draw_map(ax=self.ax)
        return self.ax
    
    def _update_frame(self, frame):
 
        self.ax.clear()
        self._init_frame()
        episode, paths = self.paths_history[frame]
        
        for agent_id, path in paths.items():
            if len(path) > 1:
                xs, ys = zip(*path)
                self.ax.plot(xs, ys, '--', 
                            color=self.colors[agent_id],
                            linewidth=2,
                            alpha=0.7)
                self.ax.scatter(xs[-1], ys[-1],
                               color=self.colors[agent_id],
                               s=200, 
                               edgecolor='black',
                               zorder=10)
        self.ax.set_title(f"Episode: {episode} | Paths")
        return self.ax
    
    def animate(self, save_path="path_evolution.gif"):

        anim = FuncAnimation(self.fig, 
                            self._update_frame,
                            frames=len(self.paths_history),
                            init_func=self._init_frame,
                            interval=300)
        anim.save(save_path, writer='pillow', dpi=120)
        print(f"Animation saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=20)       
    parser.add_argument('--agents', type=int, default=3)      
    parser.add_argument('--obstacles', type=float, default=0.2)
    parser.add_argument('--episodes', type=int, default=500)  
    parser.add_argument('--alpha', type=float, default=0.2)    
    parser.add_argument('--gamma', type=float, default=0.9)    
    parser.add_argument('--epsilon', type=float, default=0.3)   
    config = parser.parse_args()

    train = trainer(config)
    train.training()
    
    animator = PathAnimator(train.env, train.paths_history)
    animator.animate()