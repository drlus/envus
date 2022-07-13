import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

class Spec:
    def __init__(self):
        self.max_episode_steps = 20

class Cigib(gym.Env, Spec):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, dt, v_tm_min, v_tm_max, omega_t_min, omega_t_max, p_s_ref_min, p_s_ref_max, q_s_ref_min, q_s_ref_max):
        
        # Input parameters
        self.dt = dt 
        self.v_tm_min = v_tm_min 
        self.v_tm_max = v_tm_max 
        self.omega_t_min = omega_t_min
        self.omega_t_max = omega_t_max
        self.p_s_ref_min = p_s_ref_min 
        self.p_s_ref_max = p_s_ref_max 
        self.q_s_ref_min = q_s_ref_min 
        self.q_s_ref_max = q_s_ref_max 
        
        self.spec = Spec()
        
        X_s = 0.1
        Dp_s = 0.05
        Omega_b = 2*np.pi*50

        Dtheta_t = X_s*Dp_s
        Domega_t = Dtheta_t/(dt*Omega_b)
        Dalpha_t = Domega_t/dt
        
        self.dt = dt
        self.viewer = None

        # Actions: 

        self.u_min = np.array([self.v_tm_min, self.omega_t_min], dtype=np.float32)
        self.u_max = np.array([self.v_tm_max, self.omega_t_max], dtype=np.float32)

        # self.action_space = spaces.Box(low=self.u_min, high=self.u_max, dtype=np.float32) 
        self.action_space = spaces.Box(low=np.zeros(len(self.u_max)), high=np.ones(len(self.u_max)), dtype=np.float32)         

        # Observations:
        # v_sd, v_sq, i_sd, i_sq, p_s_ref, q_s_ref, delta_t : (-1.0,1.0)
        self.obs_low   = np.array([ -1.5,-1.5,-1.5,-1.5,-1.0,-1.0, 0.0], dtype=np.float64)
        self.obs_high  = np.array([ 1.5, 1.5, 1.5, 1.5,1.0, 1.0, 2*np.pi], dtype=np.float64)        
        # self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=np.zeros(len(self.obs_low)), high=np.ones(len(self.obs_low)), dtype=np.float32)       

        self.seed()

        # parameters
        self.R_s = 0.01
        self.X_s = X_s
        self.R_g = 0.001
        self.X_g = 0.01
        self.Omega_b = Omega_b      
        
        self.t = 0.0
                
    def reset(self, random = False):
               
        self.t = 0.0

        # initial actions
        u_c = np.array([1.0, 1.0], dtype=np.float32)
        
        self.xi_p = 0
        self.xi_q = 0
        
        # refenrece step
        self.t_step_pref = 0.1
        self.t_step_qref = 0.3
        self.step_pref = np.random.uniform(low=self.p_s_ref_min, high=self.p_s_ref_max)
        self.step_qref = np.random.uniform(low=self.q_s_ref_min, high=self.q_s_ref_max)

        # initial states
        # v_sd,v_sq = 0.0,-1.0
        # i_sd,i_sq = 0.0,0.0
        # p_s,q_s = 0.0,0.0
        # p_s_ref,q_s_ref = 0.0,0.0
        
        ###################################################
        # states
        delta_t  = 0
        Domega_g = 0

        # parameters
        R_s = self.R_s
        X_s = self.X_s
        R_g = self.R_g
        X_g = self.X_g 
        Omega_b = self.Omega_b

        # actions
        v_tm = u_c[0]
        omega_t = u_c[1]
        phi_t = 0.0
        
        # references
        p_s_ref = 0.0
        q_s_ref = 0.0
        if random:
            self.p_s_ref = np.random.uniform(low=self.p_s_ref_min, high=self.p_s_ref_max)
            self.q_s_ref = np.random.uniform(low=self.q_s_ref_min, high=self.q_s_ref_max)
            p_s_ref = self.p_s_ref
            q_s_ref = self.q_s_ref

        # perturbations
        v_gm = 1.0
        RoCoF = 0.0  
        phi_g = 0.0

        # plant
        omega_g = 1.0 + Domega_g
        omega_coi = omega_g

        delta_t  = delta_t  + self.dt*Omega_b*(omega_t - omega_g)
        delta_g = 0.0
        Domega_g = Domega_g + self.dt*RoCoF

        v_tr = v_tm*np.cos(delta_t + phi_t)
        v_ti = v_tm*np.sin(delta_t + phi_t)
        v_gr = v_gm*np.cos(delta_g + phi_g)
        v_gi = v_gm*np.sin(delta_g + phi_g)
        i_sr = -((R_g + R_s)*(v_gr - v_tr) + (X_g + X_s)*(v_gi - v_ti))/((R_g + R_s)**2 + (X_g + X_s)**2)
        i_si = (-(R_g + R_s)*(v_gi - v_ti) + (X_g + X_s)*(v_gr - v_tr))/((R_g + R_s)**2 + (X_g + X_s)**2)
        v_sr = -R_s*i_sr + X_s*i_si + v_tr
        v_si = -R_s*i_si - X_s*i_sr + v_ti
        p_s  =  i_si*v_si + i_sr*v_sr
        q_s  = -i_si*v_sr + i_sr*v_si
        v_sm = (v_gr**2+v_gi**2)**0.5
        # r -> -Q, i -> D
        v_sd = v_si*np.cos(delta_t) + v_sr*np.sin(delta_t)
        v_sq = v_si*np.sin(delta_t) - v_sr*np.cos(delta_t)
        i_sd = i_si*np.cos(delta_t) + i_sr*np.sin(delta_t)
        i_sq = i_si*np.sin(delta_t) - i_sr*np.cos(delta_t)        
        ###################################################
        
        
        self.obs = self.state2zerone(np.array([v_sd, v_sq, i_sd, i_sq, p_s_ref, q_s_ref, delta_t], dtype=np.float32))
        # self.obs = self.state2zerone(np.array([v_sd, v_sq, i_sd, i_sq, p_s, q_s, p_s_ref, q_s_ref], dtype=np.float32))
        self.state = np.array([0, 0], dtype=np.float32)

        return self.obs

    def step(self, u):
        
        dt = self.dt
        self.t += dt
        
        u_c = self.zerone2u(u)
        
        # states
        delta_t  = self.state[0]
        Domega_g = self.state[1]

        # parameters
        R_s = self.R_s
        X_s = self.X_s
        R_g = self.R_g
        X_g = self.X_g 
        Omega_b = self.Omega_b

        # actions
        v_tm = u_c[0]
        omega_t = u_c[1]
        phi_t = 0.0
        
        # references
        p_s_ref = 0.0
        q_s_ref = 0.0
        if self.t > self.t_step_pref:
            p_s_ref = self.step_pref
        if self.t > self.t_step_qref:
            q_s_ref = self.step_qref    
        self.p_s_ref = p_s_ref
        self.q_s_ref = q_s_ref


        # perturbations
        v_gm = 1.0
        RoCoF = 0.0  
        phi_g = 0.0

        # plant
        omega_g = 1.0 + Domega_g
        omega_coi = omega_g

        delta_t  = delta_t  + self.dt*Omega_b*(omega_t - omega_g)
        delta_g = 0.0
        Domega_g = Domega_g + self.dt*RoCoF

        v_tr = v_tm*np.cos(delta_t + phi_t)
        v_ti = v_tm*np.sin(delta_t + phi_t)
        v_gr = v_gm*np.cos(delta_g + phi_g)
        v_gi = v_gm*np.sin(delta_g + phi_g)
        i_sr = -((R_g + R_s)*(v_gr - v_tr) + (X_g + X_s)*(v_gi - v_ti))/((R_g + R_s)**2 + (X_g + X_s)**2)
        i_si = (-(R_g + R_s)*(v_gi - v_ti) + (X_g + X_s)*(v_gr - v_tr))/((R_g + R_s)**2 + (X_g + X_s)**2)
        v_sr = -R_s*i_sr + X_s*i_si + v_tr
        v_si = -R_s*i_si - X_s*i_sr + v_ti
        p_s  =  i_si*v_si + i_sr*v_sr
        q_s  = -i_si*v_sr + i_sr*v_si
        v_sm = (v_gr**2+v_gi**2)**0.5
        # r -> -Q, i -> D
        v_sd = v_si*np.cos(delta_t) + v_sr*np.sin(delta_t)
        v_sq = v_si*np.sin(delta_t) - v_sr*np.cos(delta_t)
        i_sd = i_si*np.cos(delta_t) + i_sr*np.sin(delta_t)
        i_sq = i_si*np.sin(delta_t) - i_sr*np.cos(delta_t)


        self.xi_p += (p_s_ref - p_s)*self.dt
        self.xi_q += (q_s_ref - q_s)*self.dt
        
        # obserbations
        self.obs = self.state2zerone(np.array([v_sd, v_sq, i_sd, i_sq, p_s_ref, q_s_ref, delta_t], dtype=np.float32))
        # self.obs = self.state2zerone(np.array([v_sd, v_sq, i_sd, i_sq, p_s, q_s, p_s_ref, q_s_ref], dtype=np.float32))

        # reward
        x = np.array([p_s_ref - p_s, q_s_ref - q_s, self.xi_p, self.xi_q])
        Q = 100*np.diag([1, 1, 1, 1])
        # self.reward =  - ( x.dot(Q).dot(x) )
        self.reward =  - np.log( x.dot(Q).dot(x) )

        # updates 
        self.state[0] = delta_t 
        self.state[1] = Domega_g
        
        # done
        if self.t >= (self.spec.max_episode_steps - 1)*self.dt:
          done = True
        else:
          done = False
        
        return self.obs, self.reward, done, {}    
    
    def zerone2u(self, x):
        return (self.u_max - self.u_min)*x + self.u_min
    
    def state2zerone(self, x):
        return (x - self.obs_low)/(self.obs_high - self.obs_low)  
    
    def zerone2state(self, x):
        return (self.obs_high - self.obs_low)*x + self.obs_low

    def render(self):
        pass
    
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]  
    
    def close(self):
        self.dae.post()
        if self.viewer:
            self.viewer.close()
            self.viewer = None