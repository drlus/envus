import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from envus.smib import smib_dae

class SmibDAE(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, cont = False):
        self.dt = 0.05
        self.viewer = None
        
        self.cont = True
        

        # Actions:
        # v_f: (0,1.5)
        self.min_v_f = 1.0
        self.max_v_f = 5.0    
        if self.cont:
            self.action_space = spaces.Box(low=self.min_v_f, high=self.max_v_f, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(50)

        # Observations:
        # v_t: (0,1.5)
        # omega: (0,2)
        # p_t:  (-1,2)
        # q_t:  (-2,2)
        low   = np.array([ 0.0, 0.0,-1.0,-2.0], dtype=np.float32)
        high  = np.array([ 1.5, 2.0, 2.0, 2.0], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.seed()

        self.dae = smib_dae.model()
        self.dae.Dt = 0.01
        self.p_m = 0.5
        self.v_0 = 1.0
        self.v_f = 1.5
        
        
        self.dae.ini({'v_f':self.v_f,'p_m':self.p_m,'H':6.5,'v_0':self.v_0},{'delta':0.0,'v_t':1,'omega':1,'e1q':1.0,'theta_t':0.0})
        self.dae.save_xy_0('xy_0.json')
        self.t = 0.0
        

        self.v_t_ref = 1.0
        self.omega_ref = 1.0
        
        self.DV_0 = 0.05
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        v_t,omega,p_t,q_t = self.state

        p_m = self.p_m
        v_0 = self.v_0
        
        dt = self.dt
        self.t += dt
        
        if self.cont:
            v_f = np.clip(u, self.min_v_f, self.max_v_f)[0]
            self.v_f = v_f
        else:
            self.v_f = self.d2c(u)
        
        self.dae.step(self.t,{'v_f':self.v_f,'p_m':p_m,'v_0':v_0})
     
        #self.state = np.array([newth, newthdot])\n",
        self.state = self.dae.get_mvalue(['v_t','omega','p_t','q_t'])
        
        v_t,omega,p_t,q_t = self.state
        
        costs = -np.log(np.abs(self.v_t_ref - v_t)) - np.log(np.abs(p_m-p_t))
        #costs =  - np.log(np.abs(self.omega_ref - omega)*400)

        return self._get_obs(), -costs, False, {}

    def reset(self):
        

        DV_0 = self.np_random.uniform(low=-self.DV_0, high=self.DV_0)
        #self.last_u = None
        
        self.t = 0.0
        self.V_0 = 1.0
        self.dae.ini({'v_f':self.v_f,'p_m':self.p_m,'v_0':self.v_0},'xy_0.json')
        self.state = self.dae.get_mvalue(['v_t','omega','p_t','q_t'])
        self.v_0 = 1.0 + DV_0
        
        #print(f'V_0 = {self.V_0:0.3f}')
        return self._get_obs()

    def _get_obs(self):
        v_t,omega,p_t,q_t = self.state
        return np.array([v_t,omega,p_t,q_t], dtype=np.float32)

    def render(self, mode="human"):
        pass
    
    def d2c(self,action):
        
        self.v_f_c = (self.max_v_f - self.min_v_f)/(self.action_space.n)*(action) + self.min_v_f
        
        return self.v_f_c

    def c2d(self,c):
        
        d = self.action_space.n*(c - self.min_v_f)/(self.max_v_f - self.min_v_f)
        return int(d)
    
    def close(self):
        self.dae.post()
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi