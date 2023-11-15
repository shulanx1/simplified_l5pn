# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:07:06 2023

@author: xiao208
"""

import numpy as np
from func.l5_biophys import *

def calc_ionic_current(cell, v, gates):
    P = cell.P
    E_r, E_e, E_i, E_na, E_k, E_ca, tauA, tauN, tauG,\
    active_n, r_na, gamma, decay,tau_m = (P['E_l'], P['E_e'], P['E_i'], P['E_na'], P['E_k'], P['E_ca'], \
                P['tauA'], P['tauN'], P['tauG'],  P['active_n'], P['r_na'], P['gamma'], P['decay'], P['tau_m'])

    g_ion_bar = np.asarray([[cell.P['g_na'], cell.P['g_k'], cell.P['g_m'] ,cell.P['g_ca'], cell.P['g_kca'],cell.P['g_nad']],
                        [cell.P['g_na_b'],cell.P['g_k_b'], cell.P['g_m_b'] ,cell.P['g_ca_b'], cell.P['g_kca_b'],cell.P['g_nad_b']],
                        [cell.P['g_na_p'], cell.P['g_k_p'], cell.P['g_m_p'] ,cell.P['g_ca_p'], cell.P['g_kca_p'], cell.P['g_nad_p']],
                        [cell.P['g_na_d'], cell.P['g_k_d'], cell.P['g_m_d'] ,cell.P['g_ca_d'], cell.P['g_kca_d'], cell.P['g_nad_d']]])
    g_ion_bar = g_ion_bar.T #mS/cm^2

    area = cell.area #um^2
    v_0 = v[:,0]
    channels = [na(v_0[0]), kv(v_0[0]), im(v_0[0]) ,ca(v_0[0]), kca(v_0[0]), nad(v_0[0]),CaDynamics_E2()]
    g_ion = np.zeros((gates.shape[1], len(channels)-1, gates.shape[2])) #nS
    I_ion = np.zeros((gates.shape[1], len(channels)-1, gates.shape[2])) #pA
    for i in range(gates.shape[1]):
        gates_comp = gates[:,i,:]
        g_ion[i,0,:] = g_ion_bar[0,i]*channels[0].g_s(gates_comp[0,:], gates_comp[1,:])*area[i]*1e6 #na
        g_ion[i,1,:] = g_ion_bar[1,i]*channels[1].g_s(gates_comp[2,:])*area[i]*1e6 #k
        g_ion[i,2,:] = g_ion_bar[2,i]*channels[2].g_s(gates_comp[3,:])*area[i]*1e6 #im
        g_ion[i,3,:] = g_ion_bar[3,i]*channels[3].g_s(gates_comp[4,:], gates_comp[5,:])*area[i]*1e6 #ca
        g_ion[i,4,:] = g_ion_bar[4,i]*channels[4].g_s(gates_comp[6, :])*area[i]*1e6 #sk
        g_ion[i,5,:] = g_ion_bar[5,i]*channels[5].g_s(gates_comp[7:11, :])*area[i]*1e6 #nad

        for ii in range(g_ion.shape[1]):
            I_ion[i,ii,:] = g_ion[i,ii,:]*(v[i,:]-channels[ii].E) #pA

    return g_ion, I_ion

def calc_axial_current(cell, v, gates):
    Q = cell.Q
    I_axial = np.zeros((Q.shape[0], Q.shape[1], v.shape[1]))
    for i in range(Q.shape[0]):
        for ii in range(Q.shape[1]):
            I_axial[i,ii,:] = Q[i,ii]*(v[ii,:]-v[i,:])
    return I_axial





