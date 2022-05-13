import sympy as sym
import numpy as np
import pydae.build_cffi as cbuilder

#def build():
    
sin, cos = sym.sin,sym.cos
delta,omega,e1q,e1d = sym.symbols('delta,omega,e1q,e1d', real=True) # dynamic states
i_d,i_q,p_t,q_t,v_t,theta_t = sym.symbols('i_d,i_q,p_t,q_t,v_t,theta_t',real=True) # algebraic states
p_m,v_f,v_0 = sym.symbols('p_m,v_f,v_0',real=True) # algebraic states
X_d,X1d,T1d0,X_q,X1q,T1q0,K_delta = sym.symbols('X_d,X1d,T1d0,X_q,X1q,T1q0,K_delta',real=True)
R_a,X_l,H,D,Omega_b,omega_s,Theta_0 = sym.symbols('R_a,X_l,H,D,Omega_b,omega_s,Theta_0',real=True)
v_ref,K_avr,v_s = sym.symbols('v_ref,K_avr,v_s',real=True)                    
                    
# auxiliar equations
v_d = v_t*sin(delta - theta_t)  # park
v_q = v_t*cos(delta - theta_t)  # park

p_e = i_d*(v_d + R_a*i_d) + i_q*(v_q + R_a*i_q) # electromagnetic power

# dynamic equations
ddelta = Omega_b*(omega - omega_s) - K_delta*delta  # load angle
domega = 1/(2*H)*(p_m - p_e - D*(omega - omega_s)) # speed
de1q = 1/T1d0*(-e1q - (X_d - X1d)*i_d + v_f)
de1d = 1/T1q0*(-e1d + (X_q - X1q)*i_q)


# algrbraic equations
g_1 = v_q + R_a*i_q + X1d*i_d - e1q # stator
g_2 = v_d + R_a*i_d - X1q*i_q - e1d # stator
g_3 = i_d*v_d + i_q*v_q - p_t # active power 
g_4 = i_d*v_q - i_q*v_d - q_t # reactive power
g_5 = p_t - (v_t*v_0*sin(theta_t - Theta_0))/X_l  # network equation (p)
g_6 = q_t + (v_t*v_0*cos(theta_t - Theta_0))/X_l - v_t**2/X_l  # network equation (q)     
g_7 = -v_f + K_avr*(v_ref + v_s - v_t) + 1.0

params_dict = {'X_d':1.81,'X1d':0.3, 'T1d0':8.0,  # synnchronous machine d-axis parameters
            'X_q':1.76,'X1q':0.65,'T1q0':1.0,  # synnchronous machine q-axis parameters
            'R_a':0.003,
            'X_l': 0.02, 
            'H':3.5,'D':0.0,
            'Omega_b':2*np.pi*50,'omega_s':1.0,
            'Theta_0':0.0, 'K_delta':1e-8,
            'K_avr':100.0}

u_ini_dict = {'p_m':0.8,'v_ref':1.0,'v_s':1.0,'v_0':1.0}  # for the initialization problem
u_run_dict = {'p_m':0.8,'v_ref':1.0,'v_s':1.0,'v_0':1.0}  # for the running problem (here initialization and running problem are the same)

f_list = [ddelta,domega,de1q,de1d]
x_list = [ delta, omega, e1q, e1d]
g_list = [g_1,g_2,g_3,g_4,g_5,g_6,g_7]
y_list = [i_d,i_q,p_t,q_t,v_t,theta_t,v_f]

sys = {'name':'smiba_dae','colab':'smiba',
    'params_dict':params_dict,
    'f_list':f_list,
    'g_list':g_list,
    'x_list':x_list,
    'y_ini_list':y_list,
    'y_run_list':y_list,
    'u_run_dict':u_run_dict,
    'u_ini_dict':u_ini_dict,
    'h_dict':{'p_m':p_m,'p_e':p_e, 'v_f':v_f, 'v_s':v_s, 'v_0':v_0}}

bldr = cbuilder.builder(sys)
bldr.build()