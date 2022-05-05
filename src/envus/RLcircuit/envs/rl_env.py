import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

class RLcircuitEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(100)
        self.observation_space = gym.spaces.Box(low = np.array([0.]), high = np.array([5.]), dtype=np.float64)
        self.state = np.array([0.])
                       
        
    def init_2(self, Dt, L, R):
        self.Dt = Dt
        self.L = L
        self.R = R
        
    def step(self, action):
        v = action*0.004     
        self.state[0] += self.Dt*(1/self.L)*(v - self.R*self.state[0]) 

        reward = - np.log( np.abs(1.5 - self.state[0]) )
        done = False
        if self.state[0] == 2:
            done = True

        info = {}
        return self.state, reward, done, info
        
    def reset(self):
        self.state = np.array([0.])           
        return self.state