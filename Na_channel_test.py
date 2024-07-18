# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:21:35 2023

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
from func import post_analysis
P = parameters_three_com.init_params(wd)
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.random import RandomState

from func.l5_biophys import *

c = nad(-75)
c.dist = 300
v1 = np.arange(-100,50, 0.1)
# plt.figure()
# ax = plt.gca()
# ax.plot(v1, c.I1I2_a(v1), color = colors[0], linewidth = 1)
# ax2 = ax.twinx()
# ax2.plot(v1, c.I2I1_a(v1), color = colors[1], linewidth = 1)
# ax.set_ylabel("I1-I2",color=colors[0])
# ax2.set_ylabel("I2-I1",color=colors[1])
dark = np.asarray([0,0,0])/256
light = np.asarray([200,200,200])/256

dist = np.arange(5,55,5)
colors_d = np.linspace(light, dark, len(dist))
plt.figure()
for (k, d) in enumerate(dist):
    c.dist = d
    a = c.I1I2_a(v1)
    b = c.I2I1_a(v1)
    plt.plot(v1, b/(a+b), color = colors_d[k])
plt.show()

plt.figure()
for (k, d) in enumerate(dist):
    c.dist = d
    a = c.I1I2_a(v1)
    b = c.I2I1_a(v1)
    plt.plot(v1, 1/(a+b), color = colors_d[k])
plt.show()
#%% v-clamp temperature and distance dependence
freq = 11
dist = np.asarray([10.0,20.0,30.0])
temp = np.asarray([28.0, 34.0])
param_fit.vclamp_nad_test(0.05, freq = freq, dist = dist, temp = temp)

#%% v-clammp
dt = 0.05
freq = 11 #Hz
pulse_width = 5 #ms
v_init = -70.0
tau = 1

cycle = np.concatenate([40.0*np.ones([np.floor(pulse_width/dt).astype(int)]), v_init*np.ones([np.floor(1000/freq/dt).astype(int)-np.floor(pulse_width/dt).astype(int)])],axis = 0)
v_command_temp = np.concatenate([v_init*np.ones([np.floor(200/dt).astype(int)]), np.tile(cycle, 10)], axis = 0)
rand_norm = norm
rand_norm.random_state=RandomState(seed=None)
v_command_temp1 = v_command_temp + rand_norm.rvs(0,5, size = v_command_temp.shape)
t = dt*np.arange(v_command_temp.shape[0])
conv_f = t*np.exp(-t/(tau))
conv_f = conv_f/np.sum(conv_f) #normalize
wn_conv = np.convolve(rand_norm.rvs(0,5, size = v_command_temp.shape), conv_f)
v_command_temp2 = v_command_temp + wn_conv[:v_command_temp.shape[0]]*5


dist = np.asarray([10.0,10.0,10.0])
N = np.asarray([2e3,2e4,2e5])
v_command  = np.tile(v_command_temp2, (len(dist), 1))
t = dt*np.arange(v_command.shape[1])

gates = np.zeros([4, v_command.shape[0], v_command.shape[1]])
I_nad = np.zeros(v_command.shape)
G_nad = np.zeros(v_command.shape)
c = nad(v_command[:,0], dist = dist, N = N)
gates[0, :, 0] = c.O1
gates[1, :, 0] = c.C1
gates[2, :, 0] = c.I1
gates[3, :, 0] = c.I2
for k, t_c in enumerate(t[0:-1]):
    gates[:,:,k + 1] = c.update(v_command[:, k], gates[:, :, k], dt)

for i in range(gates.shape[1]):
    G_nad[i,:] = c.g_s(gates[:, i, :])
    I_nad[i,:] = G_nad[i,:]*(v_command[i,:]-c.E)

colors = np.asarray([[128,128,128],[61,139,191], [119,177,204], [6,50,99]])
colors = colors/256
plt.figure()
for i in range(len(dist)):
    plt.plot(t, G_nad[i,:], color = colors[i+1])
plt.xlim([100,1200])
plt.show()





# %%
# ## plot Nav1.6 dynamic
# from func.l5_biophys import *

# c = nad(-75)
# c.dist = 300
# v1 = np.arange(-100,50, 0.1)
# plt.figure()
# ax = plt.gca()
# ax.plot(v1, c.I1I2_a(v1), color = colors[1], linewidth = 1)
# ax2 = ax.twinx()
# ax2.plot(v1, c.I2I1_a(v1), color = colors[0], linewidth = 1)
# ax.set_ylabel("I1-I2",color=colors[1])
# ax2.set_ylabel("I2-I1",color=colors[0])

# plt.figure()
# ax = plt.gca()
# ax.plot(v1, c.I1C1_a(v1), color = colors[1], linewidth = 1)
# ax2 = ax.twinx()
# ax2.plot(v1, c.C1I1_a(v1), color = colors[2], linewidth = 1)
# ax.set_ylabel("I1-C1",color=colors[1])
# ax2.set_ylabel("C1-I1",color=colors[2])

# plt.figure()
# ax = plt.gca()
# ax.plot(v1, c.I1O1_a(v1), color = colors[1], linewidth = 1)
# ax2 = ax.twinx()
# ax2.plot(v1, c.O1I1_a(v1), color = colors[3], linewidth = 1)
# ax.set_ylabel("I1-O1",color=colors[1])
# ax2.set_ylabel("O1-I1",color=colors[3])

# plt.figure()
# ax = plt.gca()
# ax.plot(v1, c.C1O1_a(v1), color = colors[2], linewidth = 1)
# ax2 = ax.twinx()
# ax2.plot(v1, c.O1C1_a(v1), color = colors[3], linewidth = 1)
# ax.set_ylabel("C1-O1",color=colors[2])
# ax2.set_ylabel("O1-C1",color=colors[3])