# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:50:20 2023

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
from func import post_analysis
P = parameters_three_com.init_params(wd)
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from func.l5_biophys import *
from func.param_fit import *
from scipy.stats import norm
from numpy.random import RandomState

def change_i_spike_rate(rates, cell, comp = 'soma', amp = None, amp_ratio = 2):
    idx = np.where(cell.sec_i[0]==cell.P[comp])
    if amp != None:
        rates[0][idx] = amp
    elif amp_ratio != None:
        rates[0][idx] = rates[0][idx]*amp_ratio
    return rates

def change_e_spike_rate(rates, cell, comp = 'soma', amp = None, amp_ratio = 2):
    idx = np.where(cell.sec_e[0]==cell.P[comp])
    if amp != None:
        rates[0][idx] = amp
    elif amp_ratio != None:
        rates[0][idx] = rates[0][idx]*amp_ratio
    return rates

cell = comp_model.CModel(P, verbool = False)
mu = 0.5
mu_i = 1
sigma = 2# 0.05
#
v_init = -70.0
dt = 0.05
T = 300

rates_e, temp = sequences.lognormal_rates(1, P['N_e'], P['N_i'], mu, sigma)
temp, rates_i = sequences.lognormal_rates(1, P['N_e'], P['N_i'], mu_i, 3)

rates_e = [np.zeros(P['N_e'])]
rates_i =[ np.zeros(P['N_i'])]
S_e = sequences.build_rate_seq(rates_e[0], 0, T)
S_i = sequences.build_rate_seq(rates_i[0], 0, T)
#%%

cell.P['dist'] = np.asarray([0.001,0.001,20.0,40.0])
cell.P['g_na_p'] = 0
cell.P['g_nad_p'] = 20-cell.P['g_na_p']
cell.P['g_na_d'] = 0
cell.P['g_nad_d'] = 20
cell.P['N'] = np.asarray([np.inf, np.inf, 2e4, 2e4])


t, soln, stim = cell.simulate(0, T, dt, v_init, S_e, S_i, I_inj=[0.0], inj_site = [0])

colors = np.asarray([[128,128,128],[61,139,191], [119,177,204], [6,50,99]])
colors = colors/256
plt.figure()
ax = plt.subplot(211)
v = soln[0]
t_spike = spike_times(dt, v)
t_dspike = d_spike_times(dt,v, t_spike)
for i in [0,2,3]:
    ax.plot(t, soln[0][i], color = colors[i], linewidth=0.75)
ax.scatter(t_spike, v[0, np.floor(t_spike/dt).astype(np.int32)])
ax.scatter(t_dspike.T[0], v[2, np.floor(t_dspike.T[0]/dt).astype(np.int32)])
ax.scatter(t_dspike.T[1], v[3, np.floor(t_dspike.T[1]/dt).astype(np.int32)])
ax.set_xlim([0,T])
comp = 2
ax = plt.subplot(212)
for i in [0,2,3]:
    ax.plot(t, soln[1][7+i,comp,:].T, color = colors[i,:])
ax.set_xlim([0,T])
ax.set_ylim([0,1])
plt.show()

plt.figure()
plt.plot(t, soln[1][10,3,:],color = colors[3])
plt.plot(t, soln[1][10,2,:],color = colors[2])
plt.legend(['I2','I1'])
plt.xlim([0,T])
plt.show()



# %%
