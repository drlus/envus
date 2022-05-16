import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import smiba_dae

class SmibaDAE(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, cont = False):
        self.dt = 0.05
        self.viewer = None
        self.cont = cont

        # Actions:
        # v_s: (-0.1,0.1)
        self.min_v_s = -0.1
        self.max_v_s =  0.1   
        if self.cont:
            self.action_space = spaces.Box(low=self.min_v_s, high=self.max_v_s, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(11)

        # Observations:
        # v_t: (0,1.5)
        # omega: (0,2)
        # p_t:  (-1,2)
        # q_t:  (-2,2)
        low   = np.array([ 0.0, 0.0,-1.0,-2.0], dtype=np.float32)
        high  = np.array([ 1.5, 2.0, 2.0, 2.0], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.seed()

        self.dae = smiba_dae.model()
        self.dae.Dt = 0.01
        self.p_m = 0.5
        self.v_0 = 1.0
        self.v_s = 0.0
        self.v_ref = 1.0     
        self.K_avr = 100
        self.H = 6.5
        
        self.dae.ini({'K_avr':100,'v_s':0.0,'p_m':0.01},
          {'delta':0.0,'omega':1.0, 'v_t':1.0,'theta':0.0,'v_f':1.0,'e1q':1.0})
        
        self.dae.save_xy_0('xy_0.json')

        self.dae.ini({'K_avr':self.K_avr,'v_s':self.v_s,'p_m':self.p_m,'H':self.H},
                      'xy_0.json')
        
        self.t = 0.0
    
        self.omega_ref = 1.0
        
        #perturbations
        self.delta_omega_min = 0.018
        self.delta_omega_max = 0.022
        self.p_m_min = 0.2
        self.p_m_max = 0.8
        self.DV_0 = 1e-6
        self.DV_1 = 0.0
        
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
            v_s = np.clip(u, self.min_v_s, self.max_v_s)[0]
            self.v_s = v_s
        else:
            self.v_s = self.d2c(u)
        
        self.dae.step(self.t,{'K_avr':self.K_avr,'v_s':self.v_s,'p_m':self.p_m,'H':self.H,'v_0':self.v_0})     
        #self.state = np.array([newth, newthdot])\n",
        self.state = self.dae.get_mvalue(['v_t','omega','p_t','q_t'])
        if self.state[1]>1.5: print(f'SM out of step with omega = {self.state[1]}')
        
        v_t,omega,p_t,q_t = self.state
        
        costs = -(np.abs((100*(omega-1))**2 + (p_t - p_m)**2)/((100*self.delta_omega)**2))
        #costs =  - np.log(np.abs(self.omega_ref - omega)*400)

        return self._get_obs(), costs, False, {}

    def reset(self):
        

        DV_0 = self.np_random.uniform(low=-self.DV_0, high=self.DV_0)
        #self.last_u = None
        
        self.t = 0.0
        self.v_0 = 1.0
        self.v_s = 0.0
        
        self.p_m = self.np_random.uniform(low=self.p_m_min, high=self.p_m_max)
        
        self.dae.ini({'K_avr':self.K_avr,'v_s':self.v_s,'p_m':self.p_m,'H':self.H,'v_0':self.v_0},'xy_0.json')

        self.state = self.dae.get_mvalue(['v_t','omega','p_t','q_t'])
    
        self.delta_omega = self.np_random.uniform(low=self.delta_omega_min, high=self.delta_omega_max)
        #self.delta_omega = 0.02
        self.state[1] = 1 + self.delta_omega
        self.dae.xy[1] = self.state[1]
        self.v_0 = 1.0 + DV_0 + self.DV_1
        
        #print(f'V_0 = {self.V_0:0.3f}')
        return self._get_obs()

    def _get_obs(self):
        v_t,omega,p_t,q_t = self.state
        return np.array([v_t,omega,p_t,q_t], dtype=np.float32)

    def render(self, mode="human"):
        pass
    
    def d2c(self,action):
        
        self.v_f_c = (self.max_v_s - self.min_v_s)/(self.action_space.n-1)*(action) + self.min_v_s
        
        return self.v_f_c

    def c2d(self,c):
        
        d = self.action_space.n*(c - self.min_v_s)/(self.max_v_s - self.min_v_s)
        return int(d)
    
    def close(self):
        self.dae.post()
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi