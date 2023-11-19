# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 01:53:14 2023

@author: xiao208
"""


import sys
import os
wd = 'E:\\Code\\simplified_l5pn' # working directory
sys.path.insert(1, wd)

import numpy as np
import pickle
import sys
import time
import os
from func import comp_model
from func import parameters_two_com
from func import parameters_three_com
from func import sequences
from func import param_fit
from func import visualization
import matplotlib.pyplot as plt




step_num = 10
P = parameters_three_com.init_params(wd)
params = ['kappa_d','dist_p']
param_bounds = [[2,10],[5, 40]]
T = 300
dt = 0.05
[d_amp, p_amp, spike_num, burstiness, spike_time_list, value_lists] = param_fit.burst_test(P, params, param_bounds, step_num, T, dt)

value_lists = value_lists.T

visualization.plot_burstiness_contour(params,value_lists, spike_time_list, T = T)

#%%
P = parameters_three_com.init_params(wd)
T = 1000
dt = 0.05
P['dist'] = np.asarray([0.001,0.001,20.0,40.0])
P['g_na_p'] = 0
P['g_nad_p'] = 20-P['g_na_p']
P['g_na_d'] = 0
P['g_nad_d'] = 20
P['N'] = np.asarray([np.inf, np.inf, 2e4, 2e4])
# param_fit.loop_noise_tau(P, T, dt, taus = np.asarray([1,3,5,10,20,50,80,100,200,500]), amp = 0.05, N = 5, secs = [0, 2])
for tau in [80]:
    param_fit.loop_noise_amp(P, T, dt, tau = tau)