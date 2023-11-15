# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:49:41 2023

@author: xiao208
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_burstiness_contour(params,value_lists, spike_time_list, T = 200, dt = 0.05, step_num = 10, if_transpose = False):
    edges = np.arange(0,3,0.1)
    spike_num = np.zeros(value_lists.shape[1])
    burstiness = np.zeros(value_lists.shape[1])
    for k, spike_time in enumerate(spike_time_list):
        spike_time = spike_time[np.where(spike_time>25)[0]]
        isi = np.diff(spike_time)
        idx_break = np.where(isi>(T/len(spike_time))*0.5)[0]
        idx_break = np.concatenate((np.array([0]).astype('int64'), idx_break))
        spike_num[k] = np.mean(np.diff(idx_break))
        isi_hist = np.histogram(isi/(T/len(spike_time)), edges)
        burstiness[k] = np.sum(isi_hist[0][:5])/len(isi)
    if if_transpose:
        plt.figure()
        plt.contourf(value_lists[1].reshape(step_num,step_num), value_lists[0].reshape(step_num ,step_num), burstiness.reshape(step_num ,step_num), cmap = 'coolwarm',vmin = 0, vmax = 1)
        plt.xlabel(params[1])
        plt.ylabel(params[0])
        plt.colorbar()

        plt.figure()
        plt.contourf(value_lists[1].reshape(step_num,step_num), value_lists[0].reshape(step_num ,step_num), spike_num.reshape(step_num ,step_num), cmap = 'coolwarm',vmin = 1)
        plt.xlabel(params[1])
        plt.ylabel(params[0])
        plt.colorbar()
    else:
        plt.figure()
        plt.contourf(value_lists[0].reshape(step_num,step_num), value_lists[1].reshape(step_num ,step_num), burstiness.reshape(step_num ,step_num), cmap = 'coolwarm',vmin = 0, vmax = 1)
        plt.xlabel(params[0])
        plt.ylabel(params[1])
        plt.colorbar()

        plt.figure()
        plt.contourf(value_lists[0].reshape(step_num,step_num), value_lists[1].reshape(step_num ,step_num), spike_num.reshape(step_num ,step_num), cmap = 'coolwarm',vmin = 1)
        plt.xlabel(params[0])
        plt.ylabel(params[1])
        plt.colorbar()