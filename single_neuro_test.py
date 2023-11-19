# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:50:20 2023

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
from func import post_analysis
P = parameters_three_com.init_params(wd)
import matplotlib.pyplot as plt
from func.l5_biophys import *
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

#%

cell.P['dist'] = np.asarray([0.001,0.001,20.0,40.0])
cell.P['g_na_p'] = 0
cell.P['g_nad_p'] = 20-cell.P['g_na_p']
cell.P['g_na_d'] = 0
cell.P['g_nad_d'] = 20
cell.P['N'] = np.asarray([np.inf, np.inf, 2e4, 2e4])

#%%
# t, soln, stim = cell.simulate(0, T, dt, v_init, S_e, S_i, I_inj=[0.0], inj_site = [0])
tau = 50
t = np.arange(0, T+dt, dt)
rand_norm = norm
rand_norm.random_state=RandomState(seed=None)
conv_f = t*np.exp(-t/(tau))
conv_f = conv_f/np.sum(conv_f) #normalize
wn_conv = np.convolve(rand_norm.rvs(0,2, size = t.shape), conv_f)
#%%
I_inj2 = wn_conv[:t.shape[0]]/np.max(np.abs(wn_conv[:t.shape[0]]))*0.05
t, soln, stim = cell.simulate(0, T, dt, v_init, S_e, S_i, I_inj=np.asarray([-0.00*np.ones(t.shape),I_inj2]), inj_site = [2,0])

colors = np.asarray([[128,128,128],[61,139,191], [119,177,204], [6,50,99]])
colors = colors/256
plt.figure()
ax = plt.subplot(211)
for i in [0,2,3]:
    ax.plot(t, soln[0][i], color = colors[i], linewidth=0.75)
ax.set_xlim([50,T])

# plt.figure()
# plt.plot(t, soln[1][10,3,:],color = colors[3])
# plt.plot(t, soln[1][10,2,:],color = colors[2])
# plt.xlim([0,T])

comp = 2
ax = plt.subplot(212)
for i in [3]:
    ax.plot(t, soln[1][7+i,comp,:].T, color = colors[i,:])
ax.set_xlim([50,T])
ax.set_ylim([0.7,1])

plt.figure()
plt.plot(t, I_inj2)
ax.set_xlim([50,T])
# c = nad(v_init, dist = P['dist'][comp])
# plt.figure()
# plt.plot(c.I1C1_a(soln[0][comp,:])*soln[1][9,comp,:])
# plt.plot(c.I1I2_a(soln[0][comp,:])*soln[1][9,comp,:])



# I_axial = post_analysis.calc_axial_current(cell, soln[0], soln[1])
# [g_ion,I_ion] = post_analysis.calc_ionic_current(cell, soln[0], soln[1])
# I_net_axial = np.sum(I_axial, axis = 0)
# plt.figure()
# for i in [0,2,3]:
#     plt.plot(t, I_net_axial[i], color = colors[i], linewidth=0.75)
# plt.xlim([0,T])

# plt.figure()
# for i in [2]:
#     plt.plot(t, I_ion[i,0,:]+I_ion[i,5,:], color = colors[i], linewidth=0.75)
# plt.xlim([370,395])
# plt.ylim([-2000, 100])


# plt.figure()
# for i in [3]:
#     plt.plot(t, I_ion[i,3,:], color = colors[i], linewidth=0.75)
# plt.xlim([345,370])

# plt.xlim([75, 125])
# plt.figure()
# plt.plot(t, soln[1][10,2,:], color = colors[3], linewidth = 0.75)
# plt.plot(t, soln[1][10,3,:], color = colors[0], linewidth = 0.75)
# plt.xlim([0,T])
# plt.ylim([0.95,1])

# ## plot gradient
# f = cell.grad_ion(soln, t, dt)
# plt.figure()
# for i in [0,2,3]:
#     plt.plot(t, f[0][i], color = colors[i], linewidth=0.75)
# plt.xlim([0,T])
# plt.xlim([75, 125])
# plt.figure()
# plt.plot(t, f[0][15]) # ca
# plt.xlim([75, 125])

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