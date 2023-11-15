# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 21:34:22 2023

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
import matplotlib.pyplot as plt

def spike_times(dt, v):
    """ Get spike times from voltage trace.

    Parameters
    ----------
    dt : float
        simulation timestep
    v : ndarray
        compartment voltages v=v[compartment, time]
    Returns
    -------
    t_spike : ndarray
        spike times
    """
    thresh_cross = np.where(v[0, :] > 0)[0]
    if thresh_cross.size > 0:
        spikes = np.where(np.diff(thresh_cross) > 1)[0] + 1
        spikes = np.insert(spikes, 0, 0)
        spikes = thresh_cross[spikes]
        t_spike = spikes*dt - 2
    else:
        t_spike = np.array([])
    return t_spike

def burst_test(P, params, param_bounds, step_num = 10, T = 200, dt = 0.05):
    cell = comp_model.CModel(P, verbool = False)
    v_init = -75.0
    rates_e = np.zeros(P['N_e'])
    rates_i = np.zeros(P['N_i'])
    S_e = sequences.build_rate_seq(rates_e, 0, T)
    S_i = sequences.build_rate_seq(rates_i, 0, T)
    v = []
    for bound in param_bounds:
        v_temp = np.linspace(bound[0], bound[1], step_num)
        v.append(v_temp)
    if len(v)==1:
        value_lists = np.asarray([v[0]])
        value_lists = value_lists.T
    else:
        value_lists_temp = np.meshgrid(v[0], v[1])
        value_lists = value_lists_temp
        for i in range(len(value_lists_temp)):
            value_lists[i] = value_lists_temp[i].ravel()
        value_lists = np.asarray(value_lists)
        value_lists = value_lists.T

    d_amp = np.zeros(value_lists.shape[0])
    p_amp = np.zeros(value_lists.shape[0])
    spike_num = np.zeros(value_lists.shape[0])
    burstiness = np.zeros(value_lists.shape[0])
    spike_time_list = []
    edges = np.arange(0,3,0.1)
    for k, value_list in enumerate(value_lists):
        print('trail No. %d \n' % k)
        for param, value in zip(params, value_list):
            print('setting param %s to be %3f \n' %(param, value))
            if param == 'dist_p':
                cell.P['dist'][2] = value
            else:
                cell.P[param] = value
                if param == 'g_na_d':
                    cell.P['g_nad_d'] = 20-cell.P['g_na_d']
                if param == 'g_na_p':
                    cell.P['g_nad_p'] = 30-cell.P['g_na_p']
                if param == 'g_nad_d':
                    cell.P['g_na_d'] = 20-cell.P['g_nad_d']
                if param == 'g_nad_p':
                    cell.P['g_na_p'] = 30-cell.P['g_nad_p']

        I_inj = -0.03#0.0
        while spike_num[k] < 2:
            t, soln, stim = cell.simulate(0, T, dt, v_init, S_e, S_i, I_inj=I_inj, inj_site = 0)
            v = soln[0]
            t_spike = spike_times(dt, v)
            spike_num[k] = len(t_spike)
            if len(t_spike) == 0:
                I_inj = I_inj + 0.1
            d_amp[k] = (np.max(v[3])-np.min(v[3]))
            p_amp[k] = (np.max(v[2])-np.min(v[2]))
            isi = np.diff(t_spike)
            isi_hist = np.histogram(isi/(T/spike_num[k]), edges)
            burstiness[k] = np.sum(isi_hist[0][:5])/len(isi)
        spike_time_list.append(t_spike)
        print('finishing trial No. %d, spike num %d, burstiness %3f' %(k, spike_num[k], burstiness[k]))
    return [d_amp, p_amp, spike_num, burstiness, spike_time_list, value_lists]


