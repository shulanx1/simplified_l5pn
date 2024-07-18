#!/usr/bin/env python3
"""
Compartmental model class and functions for simulation and training.
"""
import numpy as np
import numba as nb
import json

from func.l5_biophys import *

class CModel:
    """Compartmental model object for simulation of dendritic computation and
    learning.

    Parameters
    ----------
        P : dict
        model and simulation parameters

    Attributes
    ----------
        P : dict
            model and simulation paramaters
        L_s, a_s : ndarray
            segment lengths and radii
        C : ndarray
            adjacency matrix for segments
        Q : ndarray
            axial conductance matrix
        sec_e, sec_i : ndarray
            synapse locations
        seg_e, seg_i : ndarray
            synapse segment numbers
        seg2sec : ndarray
            sections and positions for each segment
        b_type_e, b_type_i : list
            branch types (basal/apical/soma) for all synapses
        H_e, H_i : ndarray
            projection matrices (inputs->compartments)
        w_e, w_i : ndarray
            synaptic weights
        g_ops : ndarray
            list of Gaussian elimination steps for solve
        f_ops : ndarray
            list of forward sub steps for solve
    """

    def __init__(self, P, verbool = False):
        self.P = P
        N_e = P['N_e']
        N_i = P['N_i']
        locs_e = P['locs_e']
        locs_i = P['locs_i']
        self.verbool = verbool
        self.area = np.asarray([P['area'], P['area']*P['rho_b'], P['area']*P['rho_p'], P['area']*P['rho_d']])*1e-8  # cm^2
        self.cm = np.asarray([P['c_m'],P['c_m_b'], P['c_m_p'], P['c_m_d']])
        self.gpas = np.asarray([P['g_l'],P['g_l_b'], P['g_l_p'], P['g_l_d']])
        self.gamma = P['gamma']
        self.decay = P['decay']
        self.dend = np.append(P['basal'], P['oblique'])
        self.axon= np.asarray([])
        self.apic = np.asarray(P['apical'])
        self.soma = np.asarray(P['soma'])
        self.g_ion = self.P['g_ion']

        self.sec_e = self.synapse_locations(locs_e, N_e)
        self.sec_i = self.synapse_locations(locs_i, N_i)
        self.C = np.asarray([[0,1,1,0], [1,0,0,0], [1,0,0,1],[0,0,1,0]])
        self.Q = self.build_axial_mat()
        self.H_e, self.seg_e = self.syn_mat(self.sec_e)
        self.H_i, self.seg_i = self.syn_mat(self.sec_i)
        self.g_ops = self.gauss_ops()
        self.f_ops = self.fsub_ops()

        self.w_e = np.array(self.sec_e.shape[1]*[P['g_max_A'] + P['g_max_N']])
        self.w_i = np.array(self.sec_i.shape[1]*[P['g_max_G']])

    def synapse_locations(self,secs, N):
        syns = []
        for sec,i in enumerate(secs):
            syns.append([sec]*np.round(N*self.area[sec]/np.sum(self.area[secs])).astype('uint'))
        syns = np.concatenate(syns)
        if len(syns)<N:
            syns = np.append(syns,np.asarray([secs[-1]]*(N-len(syns))))
        syns = np.asarray([syns, np.zeros((syns.shape[0]))])
        syns = syns.astype('uint')
        return syns


    def g_axial(self, a_i, a_j, L_i, L_j, R_a):
        """Axial conductance from compartment j to i in unbranched section."""
        return (a_i*a_j**2)/(R_a*L_i*(L_j*a_i**2 + L_i*a_j**2))

    def g_axial_b(self, a_i, a_j, L_i, L_j, a_k, L_k, R_a):
        """Axial conductance from compartment j to i through branch point."""
        return ((a_i*a_j**2)/(L_i**2*L_j))/(R_a*(a_i**2/L_i + a_j**2/L_j +
                sum(a_k**2 / L_k)))
    
    def build_axial_mat(self):
        """Build and return axial conductance matrix Q."""

        self.area = np.asarray([self.P['area'], self.P['area']*self.P['rho_b'], self.P['area']*self.P['rho_p'], self.P['area']*self.P['rho_d']])*1e-8  # cm^2
        kappa = np.asarray([self.P['kappa_b'], self.P['kappa_p'], self.P['kappa_d']])
        C = self.C
        Q = np.zeros(C.shape)
        Q[0,1] = (1/kappa[0]*1e-3)/(self.area[0])   # mS/cm^2
        Q[1,0] = (1/kappa[0]*1e-3)/(self.area[1])
        Q[0,2] = (1/kappa[1]*1e-3)/(self.area[0])
        Q[2,0] = (1/kappa[1]*1e-3)/(self.area[2])
        Q[2,3] = (1/kappa[2]*1e-3)/(self.area[2])
        Q[3,2] = (1/kappa[2]*1e-3)/(self.area[3])
        Q = Q + np.diag(-np.sum(Q, axis=1))
        return Q

    def set_weights(self, w_e, w_i):
        """Set synaptic weights."""
        self.w_e = w_e
        self.w_i = w_i

    def syn_mat(self, syns):
        """Matrix to project conductances to specified compartments with
        conversion to current in units of mS/area.

        Parameters
        ----------
        syns : ndarray
            locations of synapses

        Returns
        -------
        H : ndarray
            synapse->segment projection matrix
        syn_segs : ndarray
            segment locations of synapses

        """
        H = np.zeros((self.C.shape[0], syns.shape[1]))

        for k, s in enumerate(syns[0]):
            H[s, k] = 1/(1e3*self.area[s])
        syn_segs = syns
        return H, syn_segs

    def gauss_ops(self):
        """Returns sequence of pivots and targets for Gaussian elimination in
        solve.
        """
        targets = [np.where(self.C[:k, k])[0] for k in range(self.C.shape[0])]
        g_ops = []
        for k in range(1, self.C.shape[0]):
            for target in targets[k]:
                g_ops.append([k, target])
        g_ops = np.array(g_ops[::-1])
        return g_ops

    def fsub_ops(self):
        """Return array of non-zero elements for forward substitution in solve.
        """
        Q = self.C + np.diag(np.arange(self.C.shape[0]))
        row_reduce(Q, self.g_ops)
        Q[np.abs(Q) < 1e-10] = 0
        np.fill_diagonal(Q, 0)
        f_ops = np.vstack((np.where(Q)[0], np.where(Q)[1])).T
        return f_ops

    def init_IC(self, v_init):
        """ Inititialse voltage, gating and synaptic variables.

        Parameters
        ----------
        v_init : int
            initial voltage

        Returns
        -------
        v0 : list
            initial voltage in all compartments
        gate0 : list
            initial states of gating variables
        syn0 : list
            initial states of synapse kinetics states
        """
        N_e, N_i = self.P['N_e'], self.P['N_i']
        v0 = len(self.C)*[v_init]
        gate0 = []
        # gate0 = [nak.m_inf(v_init), nak.h_inf(v_init), nak.n_inf(v_init),
        #          nak.p_inf(v_init), Ih.m_inf(v_init)]
        syn0 = [np.zeros((2, N_e)), np.zeros((2, N_e)), np.zeros((2, N_i))]
        return v0, gate0, syn0

    def set_IC(self, soln, stim, t0):
        """ Set conditions from specific time point in previous simulation.

        Parameters
        ----------
        soln :  list
            solution returned by `simulate`
        stim : list
            conductance states returned by `simulate`
        t0 : int
            time index to extract model states

        Returns
        -------
        v0 : list
            initial voltage in all compartments
        gate0 : list
            initial states of gating variables
        syn0 : list
            initial states of synapse kinetics states
        """
        ind_e, ind_i = stim[0], stim[1]
        v0 = soln[0][:, t0]
        gate0 = []
        # for i in range(1, len(soln)):
        #     gate0 .append(soln[i][:,t0])
        syn0 = [np.zeros((2, self.P['N_e'])), np.zeros((2, self.P['N_e'])),
            np.zeros((2,self.P['N_i']))]
        syn0[0][:, ind_e] = np.vstack((stim[2][:, t0], stim[3][:, t0]))
        syn0[1][:, ind_e] = np.vstack((stim[4][:, t0], stim[5][:, t0]))
        syn0[2][:, ind_i] = np.vstack((stim[6][:, t0], stim[7][:, t0]))
        return v0, gate0, syn0

    def simulate(self, t_0,t_1, dt, IC, S_e, S_i, I_inj=0, inj_site = 0, break_flag=False):
        """Simulate instance of CModel using input sequences S_e and S_i from
        initial conditions IC. Records detailed synaptic state variables.

        Parameters
        ----------
        t_0, t_1 : int
            start and end times of simulation
        dt : float
            timestep
        IC : array_like
            initial conditions for all state variables (v0, gate0, syn0)
        S_e, S_i : array_like
            presynaptic spike times for each E and I synapse
        I_inj : int
            injected current at soma (default 0)
        inj_site: int
            current injection site (default 0 for soma)
        break_flag : bool
            interupt simulation at time of first spike (default False)

        Returns
        -------
        t : ndarray
            time vector
        soln : list
            arrays of model states (voltage and gating variables) [v, m, h, n, p, hcn]
        stim :  list
            arrays of synaptic conductance and kinetic states and associated
            indices [ind_e, ind_i, A_r, A_d, N_r, N_d, G_r, G_d]

         """
        P = self.P
        E_r, E_e, E_i, E_na, E_k, E_ca, tauA, tauN, tauG,\
        active_n, r_na, gamma, decay,tau_m, dist, temp, N = (P['E_l'], P['E_e'], P['E_i'], P['E_na'], P['E_k'], P['E_ca'], \
                    P['tauA'], P['tauN'], P['tauG'],  P['active_n'], P['r_na'], P['gamma'], P['decay'], P['tau_m'], P['dist'], P['temp'],P['N'])

        g_ion = np.asarray([[self.P['g_na'], self.P['g_k'], self.P['g_m'] ,self.P['g_ca'], self.P['g_kca'],self.P['g_nad']],
                            [self.P['g_na_b'],self.P['g_k_b'], self.P['g_m_b'] ,self.P['g_ca_b'], self.P['g_kca_b'],self.P['g_nad_b']],
                            [self.P['g_na_p'], self.P['g_k_p'], self.P['g_m_p'] ,self.P['g_ca_p'], self.P['g_kca_p'], self.P['g_nad_p']],
                            [self.P['g_na_d'], self.P['g_k_d'], self.P['g_m_d'] ,self.P['g_ca_d'], self.P['g_kca_d'], self.P['g_nad_d']]])
        g_ion = g_ion.T
        self.P['g_ion'] = g_ion

        g_ion = self.P['g_ion']
        self.Q = self.build_axial_mat()
        dend =self.dend
        axon = self.axon
        apic = self.apic
        soma = self.soma
        cm = self.cm
        gamma = self.gamma
        decay = self.decay

        gca = self.g_ion[3,:]*1e-3

        t = np.arange(t_0, t_1+dt, dt)
        if isinstance(IC, (int, float)):
            v_0, gate_0, syn_0 = self.init_IC(IC)
        else:
            v_0, gate_0, syn_0 = IC

        M = self.Q.shape[0]
        Id = np.eye(M)
        d_inds = np.diag_indices(M)
        try:
            ind_e = np.where(S_e[:, 0] < t_1)[0]
            ind_i = np.where(S_i[:, 0] < t_1)[0]
        except IndexError:
            ind_e = np.where(S_e < t_1)[0]
            ind_i = np.where(S_i < t_1)[0]
        w_e = self.w_e[ind_e]
        w_i = self.w_i[ind_i]
        H_e = self.H_e[:, ind_e]
        H_i = self.H_i[:, ind_i]
        A_r, A_d, N_r, N_d, G_r, G_d = build_stim2(t, dt, syn_0[0][:, ind_e],
                                    syn_0[1][:, ind_e], syn_0[2][:, ind_i],
                                    S_e[ind_e], S_i[ind_i], tauA, tauN, tauG)

        if len(np.asarray(I_inj).shape)>1:
            I_inj = np.asarray(I_inj)
            for i in range(I_inj.shape[0]):
                I_inj[i,:] *= 1/(self.area[inj_site[i]])*1e-3
        else:
            I_inj *= 1/(self.area[inj_site])*1e-3  # uA/cm^2



        a_inds = np.arange(M)
        M_active = len(a_inds)

        v = np.zeros((M, len(t)))
        gates = []

        n_dend = np.asarray([i for i in a_inds if i not in dend])
        v[:, 0] = v_0
        gates = np.zeros((12, M, len(t)))
        channels = [na(v[:, 0], temp = temp), kv(v[:, 0], temp = temp), im(v[:, 0], temp = temp) ,ca(v[:, 0], temp = temp), kca(v[:, 0], temp = temp), nad(v_0, dist = dist, N = N, temp = temp),CaDynamics_E2()]
        gates[0] = np.zeros((M, len(t)))  # na_m
        gates[1] = np.zeros((M, len(t)))  # na_h
        gates[0, :, 0] = channels[0].m
        gates[1, :, 0] = channels[0].h
        gates[2] = np.zeros((M, len(t)))  # kv_m
        gates[2,:, 0] = channels[1].m
        gates[3] = np.zeros((M, len(t)))  # Im_m
        gates[3, :, 0] = channels[2].m
        gates[4] = np.zeros((M, len(t)))  # ca_m
        gates[5] = np.zeros((M, len(t)))  # ca_h
        gates[4, :, 0] = channels[3].m
        gates[5, :, 0] = channels[3].h
        gates[6] = np.zeros((M, len(t)))  #kca_m
        gates[6, :, 0] = channels[4].m
        gates[7] = np.zeros((M, len(t)))  # nad_O1
        gates[8] = np.zeros((M, len(t)))  # nad_C1
        gates[9] = np.zeros((M, len(t)))  # nad_I1
        gates[10] = np.zeros((M, len(t)))  # nad_I2
        gates[7, :, 0] = channels[5].O1
        gates[8, :, 0] = channels[5].C1
        gates[9, :, 0] = channels[5].I1
        gates[10, :, 0] = channels[5].I2
        gates[11] = np.zeros((M, len(t)))   # cai
        gates[11, :, 0] = channels[6].ca


        J = dt*(self.Q.T*1/cm).T
        q = self.Q[d_inds]
        g_a = H_e@(w_e/(1 + r_na)*(A_d - A_r).T).T
        g_n = H_e@(w_e*r_na/(1 + r_na)*(N_d - N_r).T).T
        g_g = H_i@(w_i*(G_d - G_r).T).T

        ica = np.zeros((M, len(t)))
        if active_n:
            update_J = update_jacobian
            rhs = dvdt
        else:
            update_J = update_jacobian_pas
            rhs = dvdt_pas

        for k in range(1, len(t)):
            gates[0, :, k] = gates[0, :, k-1] + (1 - np.exp(-dt/channels[0].mtau(v[:, k - 1])))*(channels[0].minf(v[:, k - 1]) - gates[0, :, k-1])
            gates[1, :, k] = gates[1, :, k-1] + (1 - np.exp(-dt/channels[0].htau(v[:, k - 1])))*(channels[0].hinf(v[:, k - 1]) - gates[1,  :, k-1])
            gates[2, :, k] = gates[2, :, k-1] + (1 - np.exp(-dt/channels[1].mtau(v[:, k - 1])))*(channels[1].minf(v[:, k - 1]) - gates[2, :, k-1])
            gates[3, :, k] = gates[3, :, k-1] + (1 - np.exp(-dt/channels[2].mtau(v[:, k - 1])))*(channels[2].minf(v[:, k - 1]) - gates[3, :, k-1])
            gates[4, :, k] = gates[4, :, k-1] + (1 - np.exp(-dt/channels[3].mtau(v[:, k - 1])))*(channels[3].minf(v[:, k - 1]) - gates[4, :, k-1])
            gates[5, :, k] = gates[5, :, k-1] + (1 - np.exp(-dt/channels[3].htau(v[:, k - 1])))*(channels[3].hinf(v[:, k - 1]) - gates[5, :, k-1])
            gates[6, :, k] = gates[6, :, k-1] + (1 - np.exp(-dt/channels[4].mtau(gates[11, :, k - 1])))*(channels[4].minf(gates[11, :, k - 1]) - gates[6, :, k-1])
            gates[7:11, :, k] = channels[5].update(v[:, k - 1], gates[7:11, :, k-1], dt)
            # gates[7, :, k] = gates[7, :, k-1] + dt*(channels[5].C1O1_a(v[:, k - 1])*gates[8, :, k-1] - channels[5].O1C1_a(v[:,k-1])*gates[7, :, k-1] + channels[5].I1O1_a(v[:, k - 1])*gates[9, :, k-1] - channels[5].O1I1_a(v[:, k - 1])*gates[7,:,k-1])
            # gates[8, :, k] = gates[8, :, k-1] + dt*(channels[5].O1C1_a(v[:, k - 1])*gates[7, :, k-1]  - channels[5].C1O1_a(v[:,k-1])*gates[8, :, k-1] + channels[5].I1C1_a(v[:, k - 1])*gates[9, :, k-1] - channels[5].C1I1_a(v[:, k - 1])*gates[8, :, k-1])
            # gates[9, :, k] = gates[9, :, k-1] + dt*(channels[5].O1I1_a(v[:, k - 1])*gates[7, :, k-1] - channels[5].I1O1_a(v[:,k-1])*gates[9, :, k-1] + channels[5].C1I1_a(v[:, k - 1])*gates[8, :, k-1] - channels[5].I1C1_a(v[:, k - 1])*gates[9, :, k-1] + channels[5].I2I1_a(v[:,k-1])*gates[10, :, k-1] - channels[5].I1I2_a(v[:,k-1])*gates[9, :, k-1])
            # # gates[10,:, k] = gates[10, :, k-1] + dt*(channels[5].I1I2_a(v[:, k - 1])*gates[9, :, k-1]  - channels[5].I2I1_a(v[:,k-1])*gates[10, :, k-1])
            # gates[10,:,k] = 1-np.sum(gates[7:10,:,k], axis = 0)
            for kk in range(M):
                ica[kk, k] = gca[kk]*channels[3].g_s(gates[4, kk, k-1], gates[5, kk, k-1])*(v[kk, k-1] - channels[3].E)
            gates[11,:,k] = channels[6].update(ica[:,k], gates[11, :, k-1], gamma, decay, dt)


            update_J(J, q, v[:, k-1], g_a[:, k], g_n[:, k], g_g[:, k],
                            E_e, tau_m, cm, dt, d_inds)
            f = rhs(v[:, k-1], g_a[:, k], g_n[:, k], g_g[:, k], self.Q, E_r,
                    E_e, E_i, tau_m, cm)
            if len(np.asarray(I_inj).shape)>1:
                f[inj_site] += I_inj[:,k]
            else:
                f[inj_site] += I_inj
            f *= dt/cm
            a = Id - J	 # note to future self: J multiplied by dt in update step
            gates_current = gates[:,:,k].reshape((gates.shape[0], M))

            g_a_ion = np.zeros((g_ion.shape[0], M))
            g_b_ion = np.zeros((g_ion.shape[0], M))

            g_a_ion[0,:] = channels[0].g_s(gates_current[0, :], gates_current[1, :])      # na
            g_b_ion[0,:] = channels[0].g_s(gates_current[0, :], gates_current[1, :])*channels[0].E

            g_a_ion[1,:] = channels[1].g_s(gates_current[2, :])       #kv
            g_b_ion[1,:] = channels[1].g_s(gates_current[2, :])*channels[1].E

            g_a_ion[2,:] = channels[2].g_s(gates_current[3, :])       #im
            g_b_ion[2,:] = channels[2].g_s(gates_current[3, :])*channels[2].E

            g_a_ion[3,:] = channels[3].g_s(gates_current[4, :], gates_current[5, :])  #ca
            g_b_ion[3,:] = channels[3].g_s(gates_current[4, :], gates_current[5, :])*channels[3].E

            g_a_ion[4,:] = channels[4].g_s(gates_current[6, :])     #sk
            g_b_ion[4,:] = channels[4].g_s(gates_current[6, :])*channels[4].E

            g_a_ion[5,:] = channels[5].g_s(gates_current[7:11, :])      # nad
            g_b_ion[5,:] = channels[5].g_s(gates_current[7:11, :])*channels[5].E

            a[a_inds, a_inds] += dt/cm[a_inds]*np.sum(g_ion*g_a_ion, axis = 0)
            b = v[:, k-1] + f - J@v[:, k-1]
            b[a_inds] += dt/cm[a_inds]*np.sum(g_ion*g_b_ion, axis = 0)
            v[:, k] = solve(a, b, self.g_ops, self.f_ops)
            if self.verbool:
                print('%d th time point solved'% k)
            if v[0, k] > 0 and break_flag:
                break
        soln = [v, gates, ica]
        stim = [ind_e, ind_i, A_r, A_d, N_r, N_d, G_r, G_d]

        return t, soln, stim

    def grad_w(self, soln, stim, t, dt, Z_e, Z_i, z_ind_e, z_ind_i):
        """Compute gradients associated with individual input
        spikes using solution from `simulate2'. Z_e and Z_i are expanded copies
        of the input pattern between those times (one spike per synapse; see
        `sequences.rate2temp`).

        Parameters
        ----------
        soln : list
            arrays of model states (voltage and gating variables)
            [v, m, h, n, p]. See `simulate2`.
        stim : list
            arrays of synaptic conductance states and associated indices
            [ind_e, ind_i, A_r, A_d, N_r, N_d, G_r, G_d]
        t : ndarray
            simulation time vector
        dt : float
            timestep from original forward simulation
        Z_e, Z_i : array_like
            presynaptic spike time for dummy copies of E and I synapses
        z_ind_e, z_ind_i :
            original indices of dummy synapses

        Returns
        -------
        f_e, f_i : ndarray
            dv_soma/dw for E and I synapses as a function of time
        gates_Y_e, gates_Y_i: dgate/dw for E and I synapses as a function of time
        """

        P = self.P
        E_l, E_e, E_i, E_na, E_k, E_ca, tauA, tauN, tauG,\
        active_n, r_na, gamma, decay,tau_m = (P['E_l'], P['E_e'], P['E_i'], P['E_na'], P['E_k'], P['E_ca'], \
                    P['tauA'], P['tauN'], P['tauG'],  P['active_n'], P['r_na'], P['gamma'], P['decay'], P['tau_m'])
        # [cm, gpas, gna, gkv, gcah, gcal, gih, gim, gnap, gkt, gkp, gsk,gamma, decay, dend, axon, apic, soma]  = self.insert_biophysical_L5()
        g_ion = self.g_ion
        dend = self.dend
        axon = self.axon
        apic = self.apic
        soma = self.soma
        cm = self.cm

        ind_e, ind_i = stim[0], stim[1]
        M = self.Q.shape[0]
        a_inds = np.arange(M)

        M_active = len(a_inds)
        ZA, ZN, ZG = build_stim(t, dt, Z_e, Z_i, tauA, tauN, tauG)

        N_e = len(Z_e)
        N_i = len(Z_i)
        Hz_e = np.zeros((self.H_e.shape[0], len(z_ind_e)))
        Hz_i = np.zeros((self.H_i.shape[0], len(z_ind_i)))
        Hz_e[np.where(self.H_e)[0][z_ind_e], np.arange(len(z_ind_e))] = self.H_e[
        np.where(self.H_e)[0][z_ind_e], np.where(self.H_e)[1][z_ind_e]]
        Hz_i[np.where(self.H_i)[0][z_ind_i], np.arange(len(z_ind_i))] = self.H_i[
        np.where(self.H_i)[0][z_ind_i], np.where(self.H_i)[1][z_ind_i]]
        he_inds = (np.where(Hz_e)[0], np.where(Hz_e)[1])
        hi_inds = (np.where(Hz_i)[0], np.where(Hz_i)[1] + N_e)

        a_inds = np.arange(M)
        M_active = len(a_inds)

        v = np.zeros((M, len(t)))
        gates = []

        n_dend = np.asarray([i for i in a_inds if i not in dend])
        v = soln[0]
        gates = soln[1]
        v_0 = v[:,0]

        channels = [na(v_0[0]), kv(v_0[0]), im(v_0[0]) ,ca(v_0[0]), kca(v_0[0]), nad(v_0, dist),CaDynamics_E2()]

        GA = stim[3] - stim[2]
        GN = stim[5] - stim[4]
        GG = stim[7] - stim[6]

        w_e = self.w_e[ind_e]
        w_i = self.w_i[ind_i]
        H_e = self.H_e[:, ind_e]
        H_i = self.H_i[:, ind_i]
        dhQ = dt*(self.Q.T*1/cm).T


        gates_a = np.zeros(gates.shape[:2])
        gates_b = np.zeros(gates.shape[:2])
        gates_c = np.zeros(gates.shape[:2])
        gates_d = np.zeros(gates.shape[:2])


        g_a_ion = np.zeros((g_ion.shape[0], M, gates.shape[2]))

        g_a_ion = np.zeros((g_ion.shape[0], M, gates.shape[2]))

        g_a_ion[0,:,:] = channels[0].g_s(gates[0, :,:], gates[1, :,:])      # na

        g_a_ion[1,:,:] = channels[1].g_s(gates[2, :,:])       #kv

        g_a_ion[2,:,:] = channels[2].g_s(gates[3, :,:])       #im

        g_a_ion[3,:,:] = channels[3].g_s(gates[4, :,:], gates[5, :,:])     #ca

        g_a_ion[4,:,:] = np.zeros((1,M,gates.shape[2]))

        g_a_ion[5,:,:] = channels[5].g_s(gates[7, :,:], gates[8, :,:], gates[9, :,:])  #nad




        if active_n:
            g_s = (H_e@(w_e/(1 + r_na)*GA.T).T + H_e@(w_e*r_na/(1 + r_na)*GN.T).T*sigma(v) -
                H_e@(w_e*r_na/(1 + r_na)*GN.T).T*d_sigma(v)*(E_e - v) +
                H_i@(w_i*GG.T).T)
            g_s = (g_s.T + cm/tau_m).T
            gw_e = 1/(1 + r_na)*(Hz_e.T@(E_e - v))*ZA + r_na/(1 + r_na)*(Hz_e.T@(
            (E_e - v)*sigma(v)))*ZN
        else:
            g_s = (H_e@(w_e/(1 + r_na)*GA.T).T + H_e@(w_e*r_na/(1 + r_na)*GN.T).T +
                H_i@(w_i*GG.T).T)
            g_s = (g_s.T + cm/tau_m).T
            gw_e = 1/(1 + r_na)*(Hz_e.T@(E_e - v))*ZA + r_na/(1 + r_na)*(Hz_e.T@(
            E_e - v))*ZN

        for k in range(g_a_ion.shape[0]):
            g_s += (g_ion[k]*g_a_ion[k,:,:].T).T

        gw_i = (Hz_i.T@(E_i - v))*ZG

        gates_Y = np.zeros((gates.shape[0], N_e + N_i, M))
        c = np.zeros((g_ion.shape[0], 3,N_e + N_i, v.shape[1]))
        B = np.zeros((M, N_e + N_i))
        f_soma = B[0, :]
        f_e = np.zeros((M,N_e,v.shape[1]))
        f_i = np.zeros((M,N_i,v.shape[1]))
        for k in range(1, v.shape[1]):
            k = k - 1
            gates_a[0, :] = channels[0].m_a(v[:, k], gates[0, :, k], gates[1, :, k])
            gates_a[1, :] = channels[0].h_a(v[:, k], gates[0, :, k], gates[1, :, k])
            gates_a[2,:] = channels[1].m_a(v[:,k], gates[2,:, k])
            gates_a[3, :] = channels[2].m_a(v[:,k], gates[3, :, k])
            gates_a[4, :] = channels[3].m_a(v[:,k], gates[4,:, k], gates[5, :, k])
            gates_a[5, :] = channels[3].h_a(v[:,k], gates[4, :, k], gates[5, :, k])
            gates_a[6, :] = np.zeros(gates_a[5,:].shape)
            gates_a[7, :] = channels[5].m_a(v[:,k], gates[7,:, k], gates[8, :, k], gates[9, :, k])
            gates_a[8, :] = channels[5].h_a(v[:,k], gates[7, :,k], gates[8, :, k], gates[9, :, k])
            gates_a[9, :] = channels[5].z_a(v[:,k], gates[7, :, k], gates[8, :,k], gates[9, :, k])

            gates_b[0, :] = channels[0].m_b(v[:, k], gates[0, :, k], gates[1, :, k])
            gates_b[1, :] = channels[0].h_b(v[:, k], gates[0, :, k], gates[1, :, k])
            gates_b[2,:] = channels[1].m_b(v[:,k], gates[2,:, k])
            gates_b[3, :] = channels[2].m_b(v[:,k], gates[3, :, k])
            gates_b[4, :] = channels[3].m_b(v[:,k], gates[4,:, k], gates[5, :, k])
            gates_b[5, :] = channels[3].h_b(v[:,k], gates[4, :, k], gates[5, :, k])
            gates_b[6, :] = np.zeros(gates_b[5,:].shape)
            gates_b[7, :] = channels[5].m_b(v[:,k], gates[7,:, k], gates[8, :, k], gates[9, :, k])
            gates_b[8, :] = channels[5].h_b(v[:,k], gates[7, :,k], gates[8, :, k], gates[9, :, k])
            gates_b[9, :] = channels[5].z_b(v[:,k], gates[7, :, k], gates[8, :,k], gates[9, :, k])

            gates_b[gates_b==0] = np.inf

            k = k + 1
            gates_c[0, :] = channels[0].m_c(v[:, k], gates[0, :, k], gates[1, :, k])
            gates_c[1, :] = channels[0].h_c(v[:, k], gates[0, :, k], gates[1, :, k])
            gates_c[2,:] = channels[1].m_c(v[:,k], gates[2,:, k])
            gates_c[3, :] = channels[2].m_c(v[:,k], gates[3, :, k])
            gates_c[4, :] = channels[3].m_c(v[:,k], gates[4,:, k], gates[5, :, k])
            gates_c[5, :] = channels[3].h_c(v[:,k], gates[4, :, k], gates[5, :, k])
            gates_c[6, :] = np.zeros(gates_c[5,:].shape)
            gates_c[7, :] = channels[5].m_c(v[:,k], gates[7,:, k], gates[8, :, k], gates[9, :, k])
            gates_c[8, :] = channels[5].h_c(v[:,k], gates[7, :,k], gates[8, :, k], gates[9, :, k])
            gates_c[9, :] = channels[5].z_c(v[:,k], gates[7, :, k], gates[8, :,k], gates[9, :, k])

            gates_d[0, :] = channels[0].m_d(v[:, k], gates[0, :, k], gates[1, :, k])
            gates_d[1, :] = channels[0].h_d(v[:, k], gates[0, :, k], gates[1, :, k])
            gates_d[2,:] = channels[1].m_d(v[:,k], gates[2,:, k])
            gates_d[3, :] = channels[2].m_d(v[:,k], gates[3, :, k])
            gates_d[4, :] = channels[3].m_d(v[:,k], gates[4,:, k], gates[5, :, k])
            gates_d[5, :] = channels[3].h_d(v[:,k], gates[4, :, k], gates[5, :, k])
            gates_d[6, :] = np.zeros(gates_d[5,:].shape)
            gates_d[7, :] = channels[5].m_d(v[:,k], gates[7,:, k], gates[8, :, k], gates[9, :, k])
            gates_d[8, :] = channels[5].h_d(v[:,k], gates[7, :,k], gates[8, :, k], gates[9, :, k])
            gates_d[9, :] = channels[5].z_d(v[:,k], gates[7, :, k], gates[8, :,k], gates[9, :, k])

            for kk in range(gates.shape[0]):
                gates_Y[kk] += (gates_a[kk, :]/gates_b[kk, :]*B.T - gates_Y[kk])*(1 - np.exp(-dt*gates_b[kk, :]))
            A = np.diag(1 + dt/cm*g_s[:, k]) - dhQ
            B[he_inds] += dt/cm[self.seg_e[z_ind_e]]*gw_e[:, k]
            B[hi_inds] += dt/cm[self.seg_i[z_ind_i]]*gw_i[:, k]
            Y_temp = np.zeros(B.shape)
            for kk in range(gates.shape[0]):
                Y_temp += (gates_c[kk,:]*gates_Y[kk,:,:]).T
            B[a_inds, :] += (dt/cm[a_inds]*Y_temp.T).T
            solve_grad(A, B, self.g_ops, self.f_ops)
            f_e[:,:, k] = B[:,:N_e]
            f_i[:,:, k] = B[:,N_e:]
            c1_temp = np.zeros((gates.shape[0], M, N_e + N_i))
            for kk in range(gates.shape[0]):
                c1_temp[kk,:,:] = (gates_d[kk,:]*gates_Y[kk,:,:]).T
            c_temp = np.zeros((g_ion.shape[0], M, N_e + N_i))
            c_temp[0,:,:] = c1_temp[0, :,:] + c1_temp[1, :,:]      # na
            c_temp[1,:,:] = c1_temp[2, :,:]       #kv
            c_temp[2,:,:] = c1_temp[3, :,:]       #im
            c_temp[3,:,:] = c1_temp[4, :,:] +  c1_temp[5, :,:]     #ca
            c_temp[4,:,:] = np.zeros((M, N_e + N_i))
            c_temp[5,:,:] = c1_temp[7, :,:] + c1_temp[8, :,:]+ c1_temp[9, :,:]  #nad

            c[:,:,:,k] = c_temp[:,[0,1,800], :]

            if self.verbool:
                print('%d th time point solved' % k)
        c_e = c[:, :,:N_e, :]
        c_i = c[:, :,N_e:, :]
        return f_e, f_i, c_e, c_i

    def grad_ion(self, soln, t, dt):
        """Compute gradients associated with individual input
        spikes using solution from `simulate2'. Z_e and Z_i are expanded copies
        of the input pattern between those times (one spike per synapse; see
        `sequences.rate2temp`).

        Parameters
        ----------
        soln : list
            arrays of model states (voltage and gating variables)
            [v, m, h, n, p]. See `simulate2`.
        stim : list
            arrays of synaptic conductance states and associated indices
            [ind_e, ind_i, A_r, A_d, N_r, N_d, G_r, G_d]
        t : ndarray
            simulation time vector
        dt : float
            timestep from original forward simulation


        Returns
        -------
        f: ndarray
            dv_soma/dion channels
        """

        P = self.P
        E_l, E_e, E_i, E_na, E_k, E_ca, tauA, tauN, tauG,\
        active_n, r_na, gamma, decay,tau_m, dist = (P['E_l'], P['E_e'], P['E_i'], P['E_na'], P['E_k'], P['E_ca'], \
                    P['tauA'], P['tauN'], P['tauG'],  P['active_n'], P['r_na'], P['gamma'], P['decay'], P['tau_m'], P['dist'])
        # [cm, gpas, gna, gkv, gcah, gcal, gih, gim, gnap, gkt, gkp, gsk,gamma, decay, dend, axon, apic, soma]  = self.insert_biophysical_L5()

        g_ion = np.asarray([[self.P['g_na'], self.P['g_k'], self.P['g_m'] ,self.P['g_ca'], self.P['g_kca'],self.P['g_nad']],
                            [self.P['g_na_b'],self.P['g_k_b'], self.P['g_m_b'] ,self.P['g_ca_b'], self.P['g_kca_b'],self.P['g_nad_b']],
                            [self.P['g_na_p'], self.P['g_k_p'], self.P['g_m_p'] ,self.P['g_ca_p'], self.P['g_kca_p'], self.P['g_nad_p']],
                            [self.P['g_na_d'], self.P['g_k_d'], self.P['g_m_d'] ,self.P['g_ca_d'], self.P['g_kca_d'], self.P['g_nad_d']]])
        g_ion = g_ion.T
        self.P['g_ion'] = g_ion

        self.g_ion = g_ion
        dend = self.dend
        axon = self.axon
        apic = self.apic
        soma = self.soma
        cm = self.cm

        M = self.Q.shape[0]
        a_inds = np.arange(M)
        M_active = len(a_inds)


        v = np.zeros((M, len(t)))
        gates = []

        n_dend = np.asarray([i for i in a_inds if i not in dend])
        v = soln[0]
        gates = soln[1]
        v_0 = v[:,0]

        channels = [na(v_0[0]), kv(v_0[0]), im(v_0[0]) ,ca(v_0[0]), kca(v_0[0]), nad(v_0, dist),CaDynamics_E2()]


        dhQ = dt*(self.Q.T*1/cm).T


        gates_a = np.zeros(gates.shape[:2])
        gates_b = np.zeros(gates.shape[:2])
        gates_c = np.zeros(gates.shape[:2])
        gates_d = np.zeros(gates.shape[:2])


        g_a_ion = np.zeros((g_ion.shape[0], M, gates.shape[2]))

        g_b_ion = np.zeros((g_ion.shape[0], M, gates.shape[2]))

        g_a_ion[0,:,:] = channels[0].g_s(gates[0, :,:], gates[1, :,:])      # na

        g_a_ion[1,:,:] = channels[1].g_s(gates[2, :,:])       #kv

        g_a_ion[2,:,:] = channels[2].g_s(gates[3, :,:])       #im

        g_a_ion[3,:,:] = channels[3].g_s(gates[4, :,:], gates[5, :,:])     #ca

        g_a_ion[4,:,:] = np.zeros((1,M,gates.shape[2]))

        g_a_ion[5,:,:] = channels[5].g_s(gates[7:11, :,:])  #nad


        g_b_ion[0,:,:] = channels[0].g_s(gates[0, :,:], gates[1, :,:])*(channels[0].E-v)      # na

        g_b_ion[1,:,:] = channels[1].g_s(gates[2, :,:])*(channels[1].E-v)       #kv

        g_b_ion[2,:,:] = channels[2].g_s(gates[3, :,:])*(channels[2].E-v)       #im

        g_b_ion[3,:,:] = channels[3].g_s(gates[4, :,:], gates[5, :,:])*(channels[3].E-v)      #ca

        g_b_ion[4,:,:] = np.zeros((1,M,gates.shape[2]))*(channels[4].E-v)

        g_b_ion[5,:,:] = channels[5].g_s(gates[7:11, :,:])*(channels[5].E-v)   #nad


        g_s = np.zeros(g_a_ion.shape[1:])
        for k in range(g_a_ion.shape[0]):
            g_s += (g_ion[k]*g_a_ion[k,:,:].T).T

        g_e = np.zeros([np.product(g_ion.shape), g_a_ion.shape[1], g_a_ion.shape[2]])
        g_ion_temp = np.reshape(g_ion,np.product(g_ion.shape))
        for k in range(np.product(g_ion.shape)):
            g_e[k,:,:] = (g_ion_temp[k]*g_b_ion[np.floor(k/M).astype('int8'),k-M*np.floor(k/M).astype('int8'),:].T).T

        B = np.zeros((M, np.product(g_ion.shape)))
        f = np.zeros((M,np.product(g_ion.shape),v.shape[1]))

        gates_Y = np.zeros((gates.shape[0],np.product(g_ion.shape), M))

        for k in range(1, v.shape[1]):
            k = k - 1
            gates_a[0, :] = channels[0].m_a(v[:, k], gates[0, :, k], gates[1, :, k])
            gates_a[1, :] = channels[0].h_a(v[:, k], gates[0, :, k], gates[1, :, k])
            gates_a[2,:] = channels[1].m_a(v[:,k], gates[2,:, k])
            gates_a[3, :] = channels[2].m_a(v[:,k], gates[3, :, k])
            gates_a[4, :] = channels[3].m_a(v[:,k], gates[4,:, k], gates[5, :, k])
            gates_a[5, :] = channels[3].h_a(v[:,k], gates[4, :, k], gates[5, :, k])
            gates_a[6, :] = np.zeros(gates_a[5,:].shape)
            gates_a[7:10, :] = channels[5].gates_a(v[:,k], gates[7:10,:, k])


            gates_b[0, :] = channels[0].m_b(v[:, k], gates[0, :, k], gates[1, :, k])
            gates_b[1, :] = channels[0].h_b(v[:, k], gates[0, :, k], gates[1, :, k])
            gates_b[2,:] = channels[1].m_b(v[:,k], gates[2,:, k])
            gates_b[3, :] = channels[2].m_b(v[:,k], gates[3, :, k])
            gates_b[4, :] = channels[3].m_b(v[:,k], gates[4,:, k], gates[5, :, k])
            gates_b[5, :] = channels[3].h_b(v[:,k], gates[4, :, k], gates[5, :, k])
            gates_b[6, :] = np.zeros(gates_b[5,:].shape)
            gates_b[7:10, :] = channels[5].gates_b(v[:,k], gates[7:10,:, k])


            gates_b[gates_b==0] = np.inf

            k = k + 1
            gates_c[0, :] = channels[0].m_c(v[:, k], gates[0, :, k], gates[1, :, k])
            gates_c[1, :] = channels[0].h_c(v[:, k], gates[0, :, k], gates[1, :, k])
            gates_c[2,:] = channels[1].m_c(v[:,k], gates[2,:, k])
            gates_c[3, :] = channels[2].m_c(v[:,k], gates[3, :, k])
            gates_c[4, :] = channels[3].m_c(v[:,k], gates[4,:, k], gates[5, :, k])
            gates_c[5, :] = channels[3].h_c(v[:,k], gates[4, :, k], gates[5, :, k])
            gates_c[6, :] = np.zeros(gates_c[5,:].shape)
            gates_c[7:10, :] = channels[5].gates_c(v[:,k], gates[7:10, :, k])

            gates_d[0, :] = channels[0].m_d(v[:, k], gates[0, :, k], gates[1, :, k])
            gates_d[1, :] = channels[0].h_d(v[:, k], gates[0, :, k], gates[1, :, k])
            gates_d[2,:] = channels[1].m_d(v[:,k], gates[2,:, k])
            gates_d[3, :] = channels[2].m_d(v[:,k], gates[3, :, k])
            gates_d[4, :] = channels[3].m_d(v[:,k], gates[4,:, k], gates[5, :, k])
            gates_d[5, :] = channels[3].h_d(v[:,k], gates[4, :, k], gates[5, :, k])
            gates_d[6, :] = np.zeros(gates_d[5,:].shape)
            gates_d[7:10, :] = channels[5].gates_d(v[:,k], gates[7:10,:, k])


            for kk in range(gates.shape[0]):
                gates_Y[kk] += (gates_a[kk, :]/gates_b[kk, :]*B.T - gates_Y[kk])*(1 - np.exp(-dt*gates_b[kk, :]))
            A = np.diag(1 + dt/cm*g_s[:, k]) - dhQ
            B += dt/(g_e[:, :,k]*cm).T
            Y_temp = np.zeros(B.shape)
            for kk in range(gates.shape[0]):
                Y_temp += (gates_c[kk,:]*gates_Y[kk,:,:]).T
            B[a_inds, :] += (dt/cm[a_inds]*Y_temp.T).T
            solve_grad(A, B, self.g_ops, self.f_ops)
            f[:,:, k] = B

            if self.verbool:
                print('%d th time point solved' % k)

        return f

@nb.jit(nopython=True, cache=True)
def kernel(t, tau_r, tau_d):
    """Returns value of double-exponential synaptic kernel.

    Parameters
    ---------
    t : float
        time
    tau_r, tau_d : float
        rise and decay time constants
    """
    t_peak = tau_r*tau_d/(tau_d - tau_r)*np.log(tau_d/tau_r)
    Z = np.exp(-t_peak/tau_d) - np.exp(-t_peak/tau_r)
    return 1/Z*(np.exp(-t/tau_d) - np.exp(-t/tau_r))


@nb.jit(nopython=True, cache=True)
def g(t, S, tau_r, tau_d):
    """Returns vector of synaptic conductances for sequence S at time t.

    Parameters
    ----------
    t : float
        time
    S : ndarray
        presynaptic spike times
    tau_r, tau_d : float
        rise and decay time constants
    """
    s_vec = (t - S)
    for i in range(s_vec.shape[0]):
            for j in range(s_vec.shape[1]):
                if ~(s_vec[i, j] > 0):
                    s_vec[i, j] = 0
    return np.sum(kernel(s_vec, tau_r, tau_d), axis=1)


def sigma(v):
    """NMDA voltage nonlinearity.

    Parameters
    ----------
    v : array_like
        voltage (mV)
    """
    return 1/(1 + 1/3.75*np.exp(-0.062*v))


def d_sigma(v):
    """Derivative of NMDA nonlinearity with respect to v.

    Parameters
    ----------
    v : array_like
        voltage (mV)
    """
    return 0.062*sigma(v)*(1 - sigma(v))


@nb.jit(nopython=True, cache=True)
def build_stim(t, dt, S_e, S_i, tauA, tauN, tauG):
    """AMPA, NMDA and GABA conductance time series.

    Parameters
    ----------
    t : ndarray
        time vector
    dt : float
        timestep
    S_e, S_i : ndarray
        presynaptic spike times for each E and I synapse
    tauA, tauN, tauG : list
        rise and decay time constants for AMPA, NMDA, GABA receptors

    Returns
    -------
    GA, GN, GG : ndarray
        conductance states as a function of time for AMPA, NMDA, GABA receptors
    """
    GA = build_G(t, dt, S_e, tauA[0], tauA[1])
    GN = build_G(t, dt, S_e, tauN[0], tauN[1])
    GG = build_G(t, dt, S_i, tauG[0], tauG[1])
    return GA, GN, GG


@nb.jit(nopython=True, cache=True)
def build_G(t, dt, S, tau_r, tau_d):
    """Build synaptic conductance time series using two-state scheme

    Parameters
    ----------
    t : ndarray
        time vector
    dt : float
        timestep
    S : ndarray
        presynaptic spike times for set of synapses
    tau_r, tau_d : float
        rise and decay time constants

    Returns
    -------
    G : ndarray
        conductance states as a function of time
    """
    G = np.zeros((len(S), len(t)))
    r = np.zeros(len(S))
    d = np.zeros(len(S))
    alpha_r = np.exp(-dt/tau_r)
    alpha_d = np.exp(-dt/tau_d)
    t_peak = tau_r*tau_d/(tau_d - tau_r)*np.log(tau_d/tau_r)
    Z = np.exp(-t_peak/tau_d) - np.exp(-t_peak/tau_r)
    for k, t_k in enumerate(t):
        r *= alpha_r
        d *= alpha_d
        dt_k = S - t_k
        ind = np.where((dt_k > 0) & (dt_k < dt))
        for j, i in enumerate(ind[0]):
            r[i] += 1/Z*np.exp(-dt_k[i, ind[1][j]]/tau_r)
            d[i] += 1/Z*np.exp(-dt_k[i, ind[1][j]]/tau_d)
        G[:, k] = d - r
    return G


@nb.jit(nopython=True, cache=True)
def build_stim2(t, dt, a_init, n_init, g_init, S_e, S_i, tauA, tauN, tauG):
    """AMPA, NMDA and GABA conductance time series with detailed kinetic states.

    Parameters
    ----------
    t : ndarray
        time vector
    dt : float
        timestep
    a_init, n_init, g_init : list
        initial conditions for AMPA, NMDA, GABA receptors
    S_e, S_i : ndarray
        presynaptic spike times for each E and I synapse
    tauA, tauN, tauG : list
        rise and decay time constants for AMPA, NMDA, GABA receptors

    Returns
    -------
    A_r, A_d, N_r, N_d, G_r, G_d : ndarray
        kinetic states as a function of time for AMPA, NMDA, GABA receptors
    """
    A_r, A_d = build_G2(t, dt, S_e, tauA[0], tauA[1], a_init[0], a_init[1])
    N_r, N_d = build_G2(t, dt, S_e, tauN[0], tauN[1], n_init[0], n_init[1])
    G_r, G_d = build_G2(t, dt, S_i, tauG[0], tauG[1], g_init[0], g_init[1])
    return A_r, A_d, N_r, N_d, G_r, G_d


@nb.jit(nopython=True, cache=True)
def build_G2(t, dt, S, tau_r, tau_d, r_init, d_init):
    """Build synaptic conductance time series using two-state scheme, and return
    kinetic states

    Parameters
    ----------
    t : ndarray
        time vector
    dt : float
        timestep
    S : ndarray
        presynaptic spike times for set of synapses
    tau_r, tau_d : float
        rise and decay time constants
    r_init, d_init : ndarray
        kinetics state initial conditions

    Returns
    -------
    R, D : ndarray
        kinetic states as a function of time
    """
    R = np.zeros((len(S), len(t)))
    D = np.zeros((len(S), len(t)))
    r = r_init
    d = d_init
    R[:, 0] = r_init
    D[:, 0] = d_init
    alpha_r = np.exp(-dt/tau_r)
    alpha_d = np.exp(-dt/tau_d)
    t_peak = tau_r*tau_d/(tau_d - tau_r)*np.log(tau_d/tau_r)
    Z = np.exp(-t_peak/tau_d) - np.exp(-t_peak/tau_r)
    for k, t_k in enumerate(t[1:]):
        r *= alpha_r
        d *= alpha_d
        dt_k = S - t_k
        ind = np.where((dt_k > 0) & (dt_k < dt))
        for j, i in enumerate(ind[0]):
            r[i] += 1/Z*np.exp(-dt_k[i, ind[1][j]]/tau_r)
            d[i] += 1/Z*np.exp(-dt_k[i, ind[1][j]]/tau_d)

        R[:, k+1] = r
        D[:, k+1] = d
    return R, D


def dvdt(v, g_a, g_n, g_g, Q, E_r, E_e, E_i, tau_m, cm):
    """ Returns right-hand side of ODE system.

    Parameters
    ----------
    v : ndarray
        voltage in all compartments
    g_a, g_n, gg : ndarray
        AMPA, NMDA GABA conductance states
    Q : ndarray
        axial conductance matrix
    E_r, E_e, E_i : int
        resting, E, and I reversal potentials
    tau_m : float
        membrane time constant
    I_inj : float
        injected current at soma
    """
    return (g_a*(E_e - v) + g_n*sigma(v)*(E_e - v) + g_g*(E_i - v) + cm/tau_m*(E_r - v) + Q@v)


def update_jacobian(J, q, v, g_a, g_n, g_g, E_e, tau_m, cm, dt, d_inds):
    """Update ODE Jacobian matrix.

    Parameters
    ----------
    J : ndarry
        Jacobian matrix
    q : ndarray
        diagonal of axial conductance matrix
    v : ndarray
        voltage in all compartments
    g_a, g_n, gg : ndarray
        AMPA, NMDA GABA conductance states
    E_e : int
        resting potential
    tau_m : float
        membrane time constant
    dt : float
        time step
    d_inds : tuple
        diagonal indices of J
    """
    J[d_inds] = dt/cm*(-g_a - g_n*sigma(v) + g_n*d_sigma(v)*(E_e - v) - g_g - cm/tau_m + q)


def dvdt_pas(v, g_a, g_n, g_g, Q, E_r, E_e, E_i, tau_m, cm):
    """ Returns right-hand side of ODE system with Ohmic NMDA receptors

    Parameters
    ----------
    v : ndarray
        voltage in all compartments
    g_a, g_n, gg : ndarray
        AMPA, NMDA GABA conductance states
    Q : ndarray
        axial conductance matrix
    E_r, E_e, E_i : int
        resting, E, and I reversal potentials
    tau_m : float
        membrane time constant
    I_inj : float
        injected current at soma
    """
    return (g_a*(E_e - v) + g_n*(E_e - v) + g_g*(E_i - v)) + (cm/tau_m*(E_r - v) +
            Q@v)


def update_jacobian_pas(J, q, v, g_a, g_n, g_g, E_e, tau_m, cm, dt, d_inds):
    """Update ODE Jacobian matrix with Ohmic NMDa receptors

    Parameters
    ----------
    J : ndarry
        Jacobian matrix
    q : ndarray
        diagonal of axial conductance matrix
    v : ndarray
        voltage in all compartments (unused)
    g_a, g_n, gg : ndarray
        AMPA, NMDA GABA conductance states
    E_e : int
        resting potential (unused)
    tau_m : float
        membrane time constant
    dt : float
        time step
    d_inds : tuple
        diagonal indices of J
    """
    J[d_inds] = dt/cm*(-g_a - g_n - g_g - cm/tau_m + q)


def solve(Q, b, g_ops, f_ops):
    """Solve linear system of equations Qx=b with Gaussian elimination
    (using v[0] as soma requires clearing upper triangle first).

    Parameters
    ----------
    Q : ndarray
        coefficient matrix
    b : ndarray
        right-hand side
    g_ops : ndarray
        sequence of operations for Gaussian elimination
    g_ops : ndarray
        sequence of operations for forward substitution

    Returns
    -------
    x : ndarray
        solution
    """
    gauss_elim(Q, b, g_ops)
    x = b
    forward_sub(Q, x, f_ops)
    return x


# def solve_grad_l5(Q, B, g_ops, f_ops):
#     """Solve linear system of matrix equations QX=B with Gaussian elimination
#     (using v[0] as soma requires clearing upper triangle first). Note: modifies
#     B in place to produce solution X.

#     Parameters
#     ----------
#     Q : ndarray
#         coefficient matrix
#     B : ndarray
#         right-hand side
#     g_ops : ndarray
#         sequence of operations for Gaussian elimination
#     f_ops : ndarray
#         sequence of operations for forward substitution

#     """
#     gauss_elim_mat(Q, B, g_ops)
#     X = B
#     forward_sub_mat(Q, X, f_ops)
#     return X

def solve_grad(Q, B, g_ops, f_ops):
    """Solve linear system of matrix equations QX=B with Gaussian elimination
    (using v[0] as soma requires clearing upper triangle first). Note: modifies
    B in place to produce solution X.

    Parameters
    ----------
    Q : ndarray
        coefficient matrix
    B : ndarray
        right-hand side
    g_ops : ndarray
        sequence of operations for Gaussian elimination
    f_ops : ndarray
        sequence of operations for forward substitution

    """
    gauss_elim_mat(Q, B, g_ops)
    X = B
    forward_sub_mat(Q, X, f_ops)


# @nb.jit(nopython=True, cache=True)
def gauss_elim(Q, b, g_ops):
    """Gaussian elimination (upper triangle cleared)

    Parameters
    ----------
    Q : ndarray
        coefficient matrix
    b : ndarray
        right-hand side
    g_ops : ndarray
        sequence of operations for Gaussian elimination
    """
    for k in range(g_ops.shape[0]):
        p = g_ops[k][0]
        t = g_ops[k][1]
        if t != p-1:
            b[t] -= Q[t, p]/Q[p, p]*b[p]
            Q[t, :] = Q[t, :] - Q[t, p]/Q[p, p]*Q[p, :]
        else:
            b[t] -= Q[t, p]/Q[p, p]*b[p]
            Q[t, t] = Q[t, t] - Q[t, p]/Q[p, p]*Q[p, t]
            Q[t, p] = 0


# @nb.jit(nopython=True, cache=True)
def gauss_elim_mat(Q, B, g_ops):
    """Gaussian elimination (upper triangle cleared) for matrix system

    Parameters
    ----------
    Q : ndarray
        coefficient matrix
    B : ndarray
        right-hand side
    g_ops : ndarray
        sequence of operations for Gaussian elimination
    """
    for k in range(g_ops.shape[0]):
        p = g_ops[k][0]
        t = g_ops[k][1]
        if t != p-1:
            B[t, :] = B[t, :] - Q[t, p]/Q[p, p]*B[p, :]
            Q[t, :] = Q[t, :] - Q[t, p]/Q[p, p]*Q[p, :]
        else:
            B[t, :] = B[t, :] - Q[t, p]/Q[p, p]*B[p, :]
            Q[t, t] = Q[t, t] - Q[t, p]/Q[p, p]*Q[p, t]
            Q[t, p] = 0


# @nb.jit(nopython=True, cache=True)
def row_reduce(Q, g_ops):
    """Row reduction of Q to precompute forward subtitution operations.

    Parameters
    ----------
    Q : ndarray
        matrix to be reduced
    g_ops : ndarray
        sequence of operations for Gaussian elimination of Q
    """
    for k in range(g_ops.shape[0]):
        p = g_ops[k][0]
        t = g_ops[k][1]
        Q[t, :] = Q[t, :] - Q[t, p]/Q[p, p]*Q[p, :]


# @nb.jit(nopython=True, cache=True)
def forward_sub(Q, x, f_ops):
    """Forward substitution after gauss_elim.

    Parameters
    ----------
    Q : ndarray
        row-reduced matrix
    x : ndarray
        view to rhs b.
    f_ops : ndarray
        sequence of operations for forward substitution
    """
    x /= np.diag(Q)
    for k in range(f_ops.shape[0]):
        r = f_ops[k][0]
        c = f_ops[k][1]
        x[r] -= Q[r, c]/Q[r, r]*x[c]


# @nb.jit(nopython=True, cache=True)
def forward_sub_mat(Q, X, f_ops):
    """Forward substitution for matrix system after gauss_elim_mat

    Parameters
    ----------
    Q : ndarray
        row-reduced matrix
    X : ndarray
        view to rhs B.
    f_ops : ndarray
        sequence of operations for forward substitution
    """
    q = np.expand_dims(np.diag(Q), 1)
    X /= q
    for k in range(f_ops.shape[0]):
        r = f_ops[k][0]
        c = f_ops[k][1]
        X[r, :] -= Q[r, c]/Q[r, r]*X[c, :]
