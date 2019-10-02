import numpy as np
from task import Task
from agents.model_1 import Model

class Hill_climbing_agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        self.model = Model(self.state_size, self.action_size)
 
        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += ( ( 0.3 )**self.count * reward ) #descount rate
        self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()

    def act(self, state):
        #get the values of the 4 actions
        action = self.model.act(state)  
        return action

    def learn(self):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.model.model.get_weights()
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.model.model.set_weights(self.best_w)
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        self.model.model.set_weights(self.model.model.get_weights() + self.noise_scale * np.random.normal(size=np.array(self.model.model.get_weights()).shape) ) # equal noise in all directions
    
 
    def save(self,name):
        self.model.save(name)