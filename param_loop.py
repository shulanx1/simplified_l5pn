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
param_bounds = [[2,10],[1, 100]]
T = 300
dt = 0.05
[d_amp, p_amp, spike_num, burstiness, spike_time_list, value_lists] = param_fit.burst_test(P, params, param_bounds, step_num, T, dt)

value_lists = value_lists.T

visualization.plot_burstiness_contour(params,value_lists, spike_time_list, T = T)

# edges = np.arange(0,3,0.1)
# spike_num = np.zeros(burstiness.shape)
# for k, spike_time in enumerate(spike_time_list):
#     spike_time = spike_time[np.where(spike_time>25)[0]]
#     isi = np.diff(spike_time)
#     idx_break = np.where(isi>(T/len(spike_time))*0.5)[0]
#     idx_break = np.concatenate((np.array([0]).astype('int64'), idx_break))
#     spike_num[k] = np.mean(np.diff(idx_break))
#     isi_hist = np.histogram(isi/(T/len(spike_time)), edges)
#     burstiness[k] = np.sum(isi_hist[0][:5])/len(isi)

# plt.figure()
# plt.contourf(value_lists[0].reshape(step_num,step_num), value_lists[1].reshape(step_num ,step_num), burstiness.reshape(step_num ,step_num), cmap = 'coolwarm',vmin = 0, vmax = 1)
# plt.xlabel(params[0])
# plt.ylabel(params[1])
# plt.colorbar()

# plt.figure()
# plt.contourf(value_lists[0].reshape(step_num,step_num), value_lists[1].reshape(step_num ,step_num), spike_num.reshape(step_num ,step_num), cmap = 'coolwarm',vmin = 1)
# plt.xlabel(params[0])
# plt.ylabel(params[1])
# plt.colorbar()
