# q_learing.py(实现强化学习来移动智能体)
"""
本程序实现了一个基于Q学习的算法，用于在网格环境中移动智能体。
"""
from collections import defaultdict
import numpy as np

class QLearning:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1, action_space=5):
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.action_space = action_space    
        self.q_table = defaultdict(lambda: np.zeros(action_space))

    def get_action(self, state: tuple):

        if np.random.rand() < self.exploration_prob:
            return np.random.randint(0, self.action_space)
        else:
            return np.argmax(self.q_table[state])
        
    def update_q_value(self, state, action, reward, next_state):

        old_q_value = self.q_table[state][action]
        next_max_q_value = np.max(self.q_table[next_state])
        self.q_table[state][action] = (1 - self.learning_rate) * old_q_value + self.learning_rate * (reward + self.discount_factor * next_max_q_value)