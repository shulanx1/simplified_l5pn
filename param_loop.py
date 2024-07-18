# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 01:53:14 2023

@author: xiao208
"""
#%%

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

