"""
Parameters for three compartment neuron
"""
import numpy as np
import os


def init_params(wd):
    N_e = 100
    N_i = 50
    soma = [0]
    basal = [1]
    oblique = [2]
    apical = [3]
    locs_e = np.array(
        basal + oblique + apical)  # location of excitatory synapses
    locs_i = np.array(
        basal + soma + oblique + apical)  # location of inhibitory synapses

    dist = np.array([0.001,0.001,10.0,40.0]) # location of each compartment (distance to soma)
    temp = np.asarray([34, 34, 34, 34]) # temperature of each compartment
    N = np.asarray([np.inf,np.inf,2e5, 2e5]) # number of nad channels in each compartment

    E_e = 0.  # excitatory reversal potential (mv)
    E_i = -75.  # inhibitory reversal potential (mv)
    tauA = np.array([0.1, 2.])  # AMPA synapse rise and decay time (ms)
    g_max_A = 0.2 * 1e-3  # AMPA conductance (uS)
    tauN = np.array([2., 75.])  # NMDA synapse rise and decay time (ms)
    g_max_N = 0.4 * 1e-3  # NMDA conductance (uS)
    tauG = np.array([1., 5.])  # GABA synapse rise and decay time (ms)
    g_max_G = 0.8 * 1e-3  # GABA conductance (uS)
    active_n = False #True
    tau_m = 25
    r_na = 0.  # NMDA/AMPA ratio

    E_l = -70.0 # leak reversal
    E_k = -90.0 # K channel reversal
    E_na = 50.0 # Na reversal
    E_ca = 140.0 # Ca reversal
    gamma = 0.0006 # percentage of free Ca
    decay = 35.7 # Ca buffer time constant 200ms

    # axo-somatic compartment
    g_l = 0.03 # axo-somatic leak conductance (mS/cm2)
    c_m = 0.75 # saxo-omatic specific capacitance (uF/cm-2)
    g_na = 3000 # axo-somatic Na conductance (mS/cm2)
    g_nad = 0.0
    g_k = 300# 150 # axo-somatic K conductance (mS/cm2)
    g_m = 0.0 # axo-somatic Im conductance (mS/cm2)
    g_ca = 0.0 # axo-somatic Ca conductance (mS/cm2)
    g_kca = 0.0 # axo-somatic SK conductance (mS/cm2)
    area = 600 # axo-somatic membrane surface area (um2)

    # basal compartment
    g_l_b = 0.03 # basal leak conductance (mS/cm2)
    c_m_b = 0.75 # basal specific capacitance (uF/cm-2)
    g_na_b = 0.0 # basal Na conductance (mS/cm2)
    g_nad_b = 0.0
    g_k_b = 0.0 # basal K conductance (mS/cm2)
    g_m_b = 0.0 # basal Im conductance (mS/cm2)
    g_ca_b = 0.0 # basal Ca conductance (mS/cm2)
    g_kca_b = 0.0 # basal SK conductance (mS/cm2)
    rho_b = 1 # basal area/somatic area
    kappa_b = 10 # basal-somatic coupling resistance (MOhm)

    # distal compartment
    g_l_d = 0.03 # dendritic leak conductance (mS/cm2)
    c_m_d = 0.75 # dendritic specific capacitance (uF/cm-2)
    g_na_d = 0.0 # 0.8*4 # dendritic Na conductance (mS/cm2)
    g_nad_d = 20
    g_k_d = 0.0 # dendritic K conductance (mS/cm2)
    g_m_d = 0.1 # dendritic Im conductance (mS/cm2)
    g_ca_d = 0.3 # dendritic Ca conductance (mS/cm2)
    g_kca_d = 3 # dendritic SK conductance (mS/cm2)
    rho_d = 20 # dendritic area/somatic area
    kappa_d = 5 # dendritic - proximal coupling resistance (MOhm)

    # proximal compartment
    g_l_p = 0.03 # proximal dendritic leak conductance (mS/cm2)
    c_m_p = 0.75 # proximal dendritic specific capacitance (uF/cm-2)
    g_na_p = 0 # proximal dendritic Na conductance (mS/cm2)
    g_nad_p = 20
    g_k_p = 0.1*4 # proximal dendritic K conductance (mS/cm2)
    g_m_p = 0.0 # proximal dendritic Im conductance (mS/cm2)
    g_ca_p = 0. # proximal dendritic Ca conductance (mS/cm2)
    g_kca_p = 0.0 # proximal dendritic SK conductance (mS/cm2)
    rho_p = 15 # dendritic area/somatic area
    kappa_p = 5 # proximal-somatic coupling resistance (MOhm)

    g_ion = np.asarray([[g_na, g_k, g_m ,g_ca, g_kca,g_nad], [g_na_b,g_k_b, g_m_b ,g_ca_b, g_kca_b,g_nad_b],[g_na_p, g_k_p, g_m_p ,g_ca_p, g_kca_p, g_nad_p], [g_na_d, g_k_d, g_m_d ,g_ca_d, g_kca_d, g_nad_d]])
    g_ion = g_ion.T

    P = {
        'soma': soma,
        'basal': basal,
        'oblique': oblique,
        'apical': apical,
        'locs_e': locs_e,
        'locs_i': locs_i,
        'N_e' : N_e,
        'N_i' : N_i,
        'E_l' : E_l, 
        'E_k' : E_k, 
        'E_na' : E_na,
        'E_ca' : E_ca ,
        'gamma' : gamma,
        'decay' : decay, 
        'tau_m': tau_m,
        'dist': dist,
        'temp': temp,
        'N': N,

        'E_e': E_e,
        'E_i': E_i,
        'tauA': tauA,
        'tauN': tauN,
        'tauG': tauG,
        'g_max_A': g_max_A,
        'g_max_N': g_max_N,
        'g_max_G': g_max_G,
        'active_n' : active_n,
        'r_na': r_na,

        # axo-somatic compartment
        'g_l' : g_l,
        'c_m' : c_m,
        'area' : area,

        # basal compartment
        'g_l_b' : g_l_b,
        'c_m_b' : c_m_b,
        'rho_b' : rho_b,
        'kappa_b' : kappa_b,

        # dendritic compartment
        'g_l_d' : g_l_d, 
        'c_m_d' : c_m_d,
        'rho_d' : rho_d,
        'kappa_d' : kappa_d,

        # proximal compartment
        'g_l_p' : g_l_p,
        'c_m_p' : c_m_p,
        'rho_p' : rho_p,
        'kappa_p' : kappa_p,

        # ionic conductances
        'g_ion': g_ion,
        'g_na': g_na,
        'g_k': g_k,
        'g_m': g_m,
        'g_ca': g_ca,
        'g_kca': g_kca,
        'g_nad': g_nad,


        'g_na_b': g_na_b,
        'g_k_b': g_k_b,
        'g_m_b': g_m_b,
        'g_ca_b': g_ca_b,
        'g_kca_b': g_kca_b,
        'g_nad_b': g_nad_b,


        'g_na_p': g_na_p,
        'g_k_p': g_k_p,
        'g_m_p': g_m_p ,
        'g_ca_p': g_ca_p,
        'g_kca_p': g_kca_p,
        'g_nad_p': g_nad_p,

        'g_na_d': g_na_d,
        'g_k_d': g_k_d,
        'g_m_d': g_m_d,
        'g_ca_d': g_ca_d,
        'g_kca_d': g_kca_d,
        'g_nad_d': g_nad_d


    }
    return P
