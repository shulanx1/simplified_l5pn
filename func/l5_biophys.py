# -*- coding: utf-8 -*-
"""
Created on Fri May 12 12:59:10 2023

@author: xiao208
"""


import numpy as np
import cmath
from scipy.optimize import minimize, LinearConstraint
from scipy.stats import norm
from numpy.random import RandomState

class na:
    def __init__(self, v, E = 50.0, N = np.inf, temp = 34.0):
        self.m = self.minf(v)
        self.h = self.hinf(v)
        if np.asarray(N).shape == np.asarray(v).shape:
            self.N = N
        else:
            if isinstance(N, (int,float,np.float64, np.int8, np.int16, np.int32)):
                self.N = N*np.ones(v.shape)
            else:
                self.N = N[0]*np.ones(v.shape)

        if np.asarray(temp).shape == np.asarray(v).shape:
            self.qt = 2.3**((temp-21)/10)
        else:
            if isinstance(temp, (int,float,np.float64, np.int8, np.int16, np.int32)):
                self.qt = 2.3**((temp-21)/10)*np.ones(v.shape)
            else:
                self.qt = 2.3**((temp[0]-21)/10)*np.ones(v.shape)

        self.E = E

    def update(self, v, dt):
        # self.m = self.m + (1-np.exp(-dt/self.mtau(v)))*(self.minf(v)-self.m)
        # self.h = self.h + (1-np.exp(-dt/self.htau(v)))*(self.hinf(v)-self.h)
        rand_norm = norm
        rand_norm.random_state=RandomState(seed=None)
        Zm = rand_norm.rvs(0,1, size = self.m.shape)
        Zh = rand_norm.rvs(0,1, size = self.h.shape)
        R0 = Zm*np.sqrt(np.abs(self.malpha(v)*(1-self.m)+self.mbeta(v)*self.m)/(self.N*dt*3))
        R1 = Zh*np.sqrt(np.abs(self.halpha(v)*(1-self.h)+self.hbeta(v)*self.h)/(self.N*dt*3))

        self.m = self.m + (1-np.exp(-dt/self.mtau(v)))*(self.minf(v)-self.m) + R0*dt
        self.h = self.h + (1-np.exp(-dt/self.htau(v)))*(self.hinf(v)-self.h) + R1*dt

    def malpha(self, v):
        return (0.182 * (v + 38.))/(1-(np.exp(-(v + 38.)/9.)))

    def d_malpha(self, v):
        return 0.182*((1. - np.exp(-(v + 38.)/9.)) - 1/9.*(v + 38.)*np.exp(-(v + 38.)/9.))/(1. - np.exp(-(v + 38.)/9.))**2

    def mbeta(self, v):
        return (-0.124 * (v + 38.))/(1-(np.exp((v + 38.)/9.)))

    def d_mbeta(self, v):
        return -0.124*((1. - np.exp((v + 38.)/9.)) + 1/9.*(v + 38.)*np.exp((v + 38.)/9.))/(1. - np.exp((v + 38.)/9.))**2

    def mtau(self,v):
        return (1/(self.malpha(v) + self.mbeta(v)))/self.qt
    
    def minf(self, v):
        return self.malpha(v)/(self.malpha(v) + self.mbeta(v))
    
    def halpha(self, v):
        return (-0.024 * (v + 65.))/(1-(np.exp((v + 65.)/6.)))

    def d_halpha(self, v):
        return -0.024*((1. - np.exp((v + 65.)/6.)) + 1/6.*(v + 65.)*np.exp((v + 65.)/6.))/(1. - np.exp((v + 65.)/6.))**2

    def hbeta(self, v):
        return (0.02 * (v + 65.))/(1-(np.exp(-(v + 65.)/6.)))

    def d_hbeta(self, v):

        return 0.02*((1. - np.exp(-(v - (-65.))/6.)) - 1/6.*(v - (-65.))*np.exp(-(v - (-65.))/6.))/(1. - np.exp(-(v - (-65.))/6.))**2

    def htau(self, v):
        return (1/(self.halpha(v) + self.hbeta(v)))/self.qt
    
    def hinf(self, v):
        return self.halpha(v)/(self.halpha(v) + self.hbeta(v))

    def g_s(self, m, h):
        """
        scaling term for conductance i = gbar*g_s(t, v)

        """
        return m**3*h

    def d_minf(self, v):
        ma = self.malpha(v)
        mb = self.mbeta(v)
        d_ma = self.d_malpha(v)
        d_mb = self.d_mbeta(v)
        return (mb*d_ma-ma*d_mb)/((ma+mb)**2)

    def d_mtau(self, v):
        ma = self.malpha(v)
        mb = self.mbeta(v)
        d_ma = self.d_malpha(v)
        d_mb = self.d_mbeta(v)
        return -1/self.qt*(d_ma + d_mb)/((ma + mb)**2)

    def d_hinf(self, v):
        ha = self.halpha(v)
        hb = self.hbeta(v)
        d_ha = self.d_halpha(v)
        d_hb = self.d_hbeta(v)
        return (hb*d_ha-ha*d_hb)/((ha+hb)**2)

    def d_htau(self, v):
        ha = self.halpha(v)
        hb = self.hbeta(v)
        d_ha = self.d_halpha(v)
        d_hb = self.d_hbeta(v)
        return -1/self.qt*(d_ha + d_hb)/((ha + hb)**2)

    def m_a(self, v, m, h):
        """
        scaling term for partial gradiant computation s_dm = a*s_v - b*s_m

        """
        d_minf = self.d_minf(v)
        mtau = self.mtau(v)
        d_mtau = self.d_mtau(v)
        minf = self.minf(v)
        return (d_minf/mtau)-(minf-m)/(mtau**2)*d_mtau

    def m_b(self, v, m, h):
        """
        scaling term for partial gradiant computation s_dm = a*s_v - b*s_m

        """
        return 1/self.mtau(v)

    def h_a(self, v, m, h):
        """
        scaling term for partial gradiant computation s_dh = a*s_v - b*s_h

        """
        d_hinf = self.d_hinf(v)
        htau = self.htau(v)
        d_htau = self.d_htau(v)
        hinf = self.hinf(v)
        return (d_hinf/htau)-(hinf-h)/(htau**2)*d_htau

    def h_b(self, v, m, h):
        """
        scaling term for partial gradiant computation s_dh = a*s_v - b*s_h

        """
        return 1/self.htau(v)

    def m_c(self, v, m, h):
        """
        scaling term for partial gradiant computation s_dv = c*s_m

        """
        return 3*(m**2*h)*(self.E - v)

    def h_c(self, v, m, h):
        """
        scaling term for partial gradiant computation s_dv = c*s_h

        """
        return (m**3)*(self.E - v)

    def m_d(self, v, m, h):
        """
        scaling term for partial gradiant computation s_dv = c*s_m

        """
        return 3*(m**2*h)

    def h_d(self, v, m, h):
        """
        scaling term for partial gradiant computation s_dv = c*s_h

        """
        return (m**3)

class nad:
    def __init__(self, v, dist = 300, N = np.inf, E = 50.0, temp = 34):
        # self.O1 = self.O1inf(v)  # O1
        # self.C1 = self.C1inf(v)  # C1
        # self.I1 = self.I1inf(v) # I2
        # self.I2 = 1- self.O1 - self.C1 - self.I1  # I2
        if isinstance(v, list):
            v = np.asarray(v)
        if np.asarray(dist).shape == np.asarray(v).shape:
            self.dist = dist
        else:
            if isinstance(dist, (int,float,np.float64, np.int8, np.int16, np.int32)):
                self.dist = dist*np.ones(v.shape)
            else:
                self.dist = dist[0]*np.ones(v.shape)

        if np.asarray(N).shape == np.asarray(v).shape:
            self.N = N
        else:
            if isinstance(N, (int,float,np.float64, np.int8, np.int16, np.int32)):
                self.N = N*np.ones(v.shape)
            else:
                self.N = N[0]*np.ones(v.shape)

        if np.asarray(temp).shape == np.asarray(v).shape:
            self.qt = 2.3**((temp-21)/10)
        else:
            if isinstance(temp, (int,float,np.float64, np.int8, np.int16, np.int32)):
                self.qt = 2.3**((temp-21)/10)*np.ones(v.shape)
            else:
                self.qt = 2.3**((temp[0]-21)/10)*np.ones(v.shape)
        self.E = E
        self.C1O1b2	  = 14
        self.C1O1v2    = 0
        self.C1O1k2	  = -6
        self.O1C1b1	  = 4
        self.O1C1v1    = -48
        self.O1C1k1	  = 9
        self.O1C1b2	  = 0
        self.O1C1v2    = 0
        self.O1C1k2	  = -5.1
        self.O1I1b1	  = 1
        self.O1I1v1	  = -42
        self.O1I1k1	  = 12
        self.O1I1b2	  = 5
        self.O1I1v2	  = 10
        self.O1I1k2	  = -12
        self.persist = 0
        self.I1C1b1	  = 0.2
        self.I1C1v1	  = -65
        self.I1C1k1	  = 10
        self.C1I1b2	  = 0.2
        self.C1I1v2	  = -65
        self.C1I1k2	  = -11
        self.I1I2b2	  = 0.022
        self.I1I2v2	  = -25
        self.I1I2k2	  = -5
        self.slowdown = 0.2
        self.I2I1b1	  = 0.0018
        self.I2I1v1	  = -50
        self.I2I1k1	  = 12
        [self.O1, self.C1, self.I1, self.I2] = self.calc_inf(v)



    # def O1inf(self,v):
    #     return 0

    # def C1inf(self, v):
    #     # return 0.25
    #     return 0.25

    # def I1inf(self,v):
    #     # return 0.25
    #     return 0.5

    def calc_inf(self, v):
        if isinstance(self.dist, (int,float,np.float64, np.int8, np.int16, np.int32)):
            dist = self.dist
            M_temp = np.asarray([[-self.O1I1_a(v)-self.O1C1_a(v), self.C1O1_a(v), self.I1O1_a(v), 0],
              [self.O1C1_a(v), -self.C1I1_a(v)-self.C1O1_a(v), self.I1C1_a(v), 0],
              [self.O1I1_a(v), self.C1I1_a(v), -self.I1O1_a(v)-self.I1C1_a(v)-self.I1I2_a(v), self.I2I1_a(v)],
              [0,0,self.I1I2_a(v), -self.I2I1_a(v)]])
    
            def f(x):
                y = np.dot(M_temp, x)
                return np.dot(y, y)
    
            cons = ({'type': 'eq', 'fun': lambda x: x.sum() - 1})
            linear_constraint = LinearConstraint(np.eye(4), np.zeros(4), np.ones(4))
            res = minimize(f, [0, 0.25, 0.5, 0.25], method='SLSQP', constraints=[cons,linear_constraint],
                                    options={'disp': False})
            infs = res['x']
            if len(np.where(infs<0)[0])>0:
                idx_zero = np.where(infs<0)[0]
                for i in idx_zero:
                    infs[i] = 0
                infs[3] = 1-np.sum(infs[:3])
            O1_inf = infs[0]
            C1_inf = infs[1]
            I1_inf = infs[2]
            I2_inf = infs[3]
        else:
            O1_inf = np.zeros(self.dist.shape)
            C1_inf = np.zeros(self.dist.shape)
            I1_inf = np.zeros(self.dist.shape)
            I2_inf = np.zeros(self.dist.shape)
            dist_temp = self.dist
            qt_temp = self.qt
            for (k, (dist,v1, qt)) in enumerate(zip(dist_temp, v, qt_temp)):
                self.dist = dist
                self.qt = qt
                M_temp = np.asarray([[-self.O1I1_a(v1)-self.O1C1_a(v1), self.C1O1_a(v1), self.I1O1_a(v1), 0],
                  [self.O1C1_a(v1), -self.C1I1_a(v1)-self.C1O1_a(v1), self.I1C1_a(v1), 0],
                  [self.O1I1_a(v1), self.C1I1_a(v1), -self.I1O1_a(v1)-self.I1C1_a(v1)-self.I1I2_a(v1), self.I2I1_a(v1)],
                  [0,0,self.I1I2_a(v1), -self.I2I1_a(v1)]])
    
                def f(x):
                    y = np.dot(M_temp, x)
                    return np.dot(y, y)
        
                cons = ({'type': 'eq', 'fun': lambda x: x.sum() - 1})
                linear_constraint = LinearConstraint(np.eye(4), np.zeros(4), np.ones(4))
                res = minimize(f, [0, 0.25, 0.5, 0.25], method='SLSQP', constraints=[cons,linear_constraint],
                                        options={'disp': False})
                infs = res['x']
                if len(np.where(infs<0)[0])>0:
                    idx_zero = np.where(infs<0)[0]
                    for i in idx_zero:
                        infs[i] = 0
                    infs[3] = 1-np.sum(infs[:3])
                O1_inf[k] = infs[0]
                C1_inf[k] = infs[1]
                I1_inf[k] = infs[2]
                I2_inf[k] = infs[3]

            self.dist = dist_temp
            self.qt = qt_temp


        return O1_inf, C1_inf, I1_inf, I2_inf


    def rates(self,v, b, vv, k):
        Arg=(v-vv)/k

        rates2 = (b/(1+np.exp(Arg)))
        if isinstance(rates2, (np.float64, np.int8, np.int16, np.int32)):
            if Arg<-50:
                rates2=b
            elif Arg>50:
                rates2=0
        else:
            rates2[np.where(Arg<-50)[0]] = b
            rates2[np.where(Arg>50)[0]] = 0

        return rates2

    def d_rates(self,v, b, vv, k):
        Arg=(v-vv)/k
    	
        # if Arg<-50:
        #     d_rates2=0
        # elif Arg>50:
        #     d_rates2=0
        # else:
        #     d_rates2 = -b/k*(np.exp(Arg))/((1 + np.exp(Arg))**2)
        d_rates2 = -b/k*(np.exp(Arg))/((1 + np.exp(Arg))**2)
        if isinstance(d_rates2, (np.float64, np.int8, np.int16, np.int32)):
            if Arg<-50:
                d_rates2=0
            elif Arg>50:
                d_rates2=0
        else:
            d_rates2[np.where(Arg<-50)[0]] = 0
            d_rates2[np.where(Arg>50)[0]] = 0
        return d_rates2

    def update(self, v, gates, dt):
        if isinstance(self.dist, (int,float,np.float64, np.int8, np.int16, np.int32)):
            self.O1 = gates[0]
            self.C1 = gates[1]
            self.I1 = gates[2]
            self.I2 = gates[3]
            rand_norm = norm
            rand_norm.random_state=RandomState(seed=None)
            Z = rand_norm.rvs(0,1, size = 4)
            R0 = Z[0]*np.sqrt(np.abs(self.C1O1_a(v)*self.C1+self.O1C1_a(v)*self.O1)/(self.N*dt))
            R1 = Z[1]*np.sqrt(np.abs(self.C1I1_a(v)*self.C1+self.I1C1_a(v)*self.I1)/(self.N*dt))
            R2 = Z[2]*np.sqrt(np.abs(self.O1I1_a(v)*self.O1+self.I1O1_a(v)*self.I1)/(self.N*dt))
            R3 = Z[3]*np.sqrt(np.abs(self.I2I1_a(v)*self.I2+self.I1I2_a(v)*self.I1)/(self.N*dt))

            O1 = self.O1 + dt*(self.C1O1_a(v)*self.C1 - self.O1C1_a(v)*self.O1 + self.I1O1_a(v)*self.I1 - self.O1I1_a(v)*self.O1 + R0 - R2)
            C1 = self.C1 + dt*(self.O1C1_a(v)*self.O1 - self.C1O1_a(v)*self.C1 + self.I1C1_a(v)*self.I1 - self.C1I1_a(v)*self.C1 - R0 + R1)
            I1 = self.I1 + dt*(self.O1I1_a(v)*self.O1 - self.I1O1_a(v)*self.I1 + self.C1I1_a(v)*self.C1 - self.I1C1_a(v)*self.I1 + self.I2I1_a(v)*self.I2 - self.I1I2_a(v)*self.I1 - R1 - R3 + R2)
            I2 = 1 - self.O1 - self.C1 - self.I1
            gates_u = np.asarray([O1,C1,I1,I2])
            if len(np.where(gates_u<0)[0])>0:
                idx_r = np.where(gates_u<0)[0]
                gates_u[idx_r]=0
                idx_r1 = np.setdiff1d(np.arange(4), idx_r)
                gates_u[idx_r1[-1]] = 1-np.sum(gates_u[np.setdiff1d(np.arange(4), idx_r1[-1])])


        else:
            a = np.zeros(gates.shape)
            self.O1 = gates[0]
            self.C1 = gates[1]
            self.I1 = gates[2]
            self.I2 = gates[3]
            rand_norm = norm
            rand_norm.random_state=RandomState(seed=None)
            Z = rand_norm.rvs(0,1, size = [4, gates.shape[1]])
            R0 = Z[0]*np.sqrt(np.abs(self.C1O1_a(v)*self.C1+self.O1C1_a(v)*self.O1)/(self.N*dt))
            R1 = Z[1]*np.sqrt(np.abs(self.C1I1_a(v)*self.C1+self.I1C1_a(v)*self.I1)/(self.N*dt))
            R2 = Z[2]*np.sqrt(np.abs(self.O1I1_a(v)*self.O1+self.I1O1_a(v)*self.I1)/(self.N*dt))
            R3 = Z[3]*np.sqrt(np.abs(self.I2I1_a(v)*self.I2+self.I1I2_a(v)*self.I1)/(self.N*dt))

            O1 = self.O1 + dt*(self.C1O1_a(v)*self.C1 - self.O1C1_a(v)*self.O1 + self.I1O1_a(v)*self.I1 - self.O1I1_a(v)*self.O1 + R0 - R2)
            C1 = self.C1 + dt*(self.O1C1_a(v)*self.O1 - self.C1O1_a(v)*self.C1 + self.I1C1_a(v)*self.I1 - self.C1I1_a(v)*self.C1 - R0 + R1)
            I1 = self.I1 + dt*(self.O1I1_a(v)*self.O1 - self.I1O1_a(v)*self.I1 + self.C1I1_a(v)*self.C1 - self.I1C1_a(v)*self.I1 + self.I2I1_a(v)*self.I2 - self.I1I2_a(v)*self.I1 - R1 - R3 + R2)
            I2 = np.ones(self.O1.shape) - self.O1 - self.C1 - self.I1
            gates_u = np.asarray([O1,C1,I1,I2])
            for i in range(gates_u.shape[1]):
                if len(np.where(gates_u[:,i]<0)[0])>0:
                    idx_r = np.where(gates_u[:,i]<0)[0]
                    gates_u[idx_r, i]=0
                    idx_r1 = np.setdiff1d(np.arange(4), idx_r)
                    gates_u[idx_r1[-1],i] = 1-np.sum(gates_u[np.setdiff1d(np.arange(4), idx_r1[-1]),i])
        return gates_u



    def C1O1_a(self, v):
        return self.qt*(self.rates(v, self.C1O1b2, self.C1O1v2, self.C1O1k2))

    def O1C1_a(self, v):
        return self.qt*(self.rates(v, self.O1C1b1, self.O1C1v1, self.O1C1k1) + self.rates(v, self.O1C1b2, self.O1C1v2, self.O1C1k2))

    def O1I1_a(self, v):
        return 0.5*self.qt*(self.rates(v, self.O1I1b1, self.O1I1v1, self.O1I1k1) + self.rates(v, self.O1I1b2, self.O1I1v2, self.O1I1k2))

    def I1O1_a(self, v):
        return self.persist*self.O1I1_a(v)

    def I1C1_a(self,v):
        return self.qt*(self.rates(v, self.I1C1b1, self.I1C1v1, self.I1C1k1))
    
    def C1I1_a(self, v):
        return self.qt*(self.rates(v, self.C1I1b2, self.C1I1v2, self.C1I1k2))
    
    def I1I2_a(self, v):
        return self.slowdown*self.dist*self.qt*(self.rates(v, self.I1I2b2, self.I1I2v2, self.I1I2k2))

    def I2I1_a(self, v):

        return self.slowdown*self.qt*(self.rates(v, self.I2I1b1, self.I2I1v1, self.I2I1k1))

    def d_C1O1_a(self, v):
        return self.qt*(self.d_rates(v, self.C1O1b2, self.C1O1v2, self.C1O1k2))

    def d_O1C1_a(self, v):
        return self.qt*(self.d_rates(v, self.O1C1b1, self.O1C1v1, self.O1C1k1) + self.rates(v, self.O1C1b2, self.O1C1v2, self.O1C1k2))

    def d_O1I1_a(self, v):
        return 0.5*self.qt*(self.d_rates(v, self.O1I1b1, self.O1I1v1, self.O1I1k1) + self.d_rates(v, self.O1I1b2, self.O1I1v2, self.O1I1k2))

    def d_I1O1_a(self, v):
        return self.persist*self.O1I1_a(v)

    def d_I1C1_a(self,v):
        return self.qt*(self.d_rates(v, self.I1C1b1, self.I1C1v1, self.I1C1k1))
    
    def d_C1I1_a(self, v):
        return self.qt*(self.d_rates(v, self.C1I1b2, self.C1I1v2, self.C1I1k2))
    
    def d_I1I2_a(self, v):
        return self.slowdown*self.dist*self.qt*(self.d_rates(v, self.I1I2b2, self.I1I2v2, self.I1I2k2))

    def d_I2I1_a(self, v):
        return self.slowdown*self.qt*(self.d_rates(v, self.I2I1b1, self.I2I1v1, self.I2I1k1))

    def g_s(self, gates):
        """
        gates = [O1,C1,I1, I2]
        scaling term for conductance i = gbar*g_s(t, v)

        """
        return gates[0]


    def gates_a(self, v, gates):
        """
        scaling term for partial gradiant computation s_dm = a - b*s_m

        """
        a = np.zeros([gates.shape[0], gates.shape[1]])
        self.M1 = self.M(v, gates)
        [self.R, self.l] = self.eigen_M(v, gates, self.M1)
        dist_temp = self.dist
        for k, (gate, v1, dist) in enumerate(zip(gates.T, v, dist_temp)):
            self.dist = dist
            O1 = gate[0]
            C1 = gate[1]
            I1 = gate[2]
            a[:,k] = np.linalg.inv(self.R[k])@np.asarray([self.d_C1O1_a(v1)*C1 + self.d_I1O1_a(v1)*I1 - (self.d_O1I1_a(v1)+self.d_O1C1_a(v1))*O1,
                            self.d_O1C1_a(v1)*O1 + self.d_I1C1_a(v1)*I1 - (self.d_C1I1_a(v1)+self.d_C1O1_a(v1))*C1,
                            (self.d_O1I1_a(v1)-self.d_I2I1_a(v1))*O1 + (self.d_C1I1_a(v1)-self.d_I2I1_a(v1))*C1 + self.d_I2I1_a(v1) - (self.d_I2I1_a(v1) + self.d_I1O1_a(v1)+self.d_I1C1_a(v1)+self.d_I1I2_a(v1))*I1])
        self.dist = dist_temp
        return a


    def M(self, v, gates):
        M = []
        dist_temp = self.dist
        for k, (v1, gate, dist) in enumerate(zip(v, gates.T, dist_temp)):
            self.dist = dist
            M_temp = np.asarray([[-self.O1I1_a(v1)-self.O1C1_a(v1), self.C1O1_a(v1), self.I1O1_a(v1)],
                      [self.O1C1_a(v1), -self.C1I1_a(v1)-self.C1O1_a(v1), self.I1C1_a(v1)],
                      [self.O1I1_a(v1)-self.I2I1_a(v1), self.C1I1_a(v1)-self.I2I1_a(v1), -self.I1O1_a(v1)-self.I1C1_a(v1)-self.I1I2_a(v1)-self.I2I1_a(v1)]])
            M.append(M_temp)
        self.dist = dist_temp
        return M

    def eigen_M(self, v, gates, M1):
        R = []
        l = []
        for k, (v1, gate, M) in enumerate(zip(v, gates.T, M1)):
            [S_temp, Q_temp] = np.linalg.eig(M)
            R.append(Q_temp)
            l.append(S_temp)

        return R, l

    def gates_b(self, v, gates):
        """
        scaling term for partial gradiant computation s_dm = a - b*s_m

        """
        # O1 = gates[0]
        # C1 = gates[1]
        # I1 = gates[2]
        # I2 = gates[3]
        # M = np.asarray([[-self.O1I1_a(v)-self.O1C1_a(v), self.C1O1_a(v), self.I1O1_a(v), 0],
        #                  [self.O1C1_a(v), -self.C1I1_a(v)-self.C1O1_a(v), self.I1C1_a(v), 0],
        #                  [self.O1I1_a(v), self.C1I1_a(v), -self.I1O1_a(v)-self.I1C1_a(v)-self.I1I2_a(v), self.I2I1_a(v)],
        #                  [0,0,self.I1I2_a(v), -self.I2I1_a(v)]])

        return -np.asarray(self.l).T


    def gates_c(self, v, gates):
        """
        scaling term for partial gradiant computation s_dv = c*s_m + d*s_v

        """
        self.M1 = self.M(v, gates)
        [self.R, self.l] = self.eigen_M(v, gates, self.M1)
        c = np.zeros([gates.shape[0], gates.shape[1]])
        for k, (v1, gate) in enumerate(zip(v, gates.T)):
            c[:,k] = self.R[k]@np.asarray([(self.E - v1), 0, 0])
        return c

    def gates_d(self, v, gates):
        """
        scaling term for partial gradiant computation s_dv = c*s_m + d*s_v

        """
        d = np.zeros([gates.shape[0], gates.shape[1]])
        for k, (v1, gate) in enumerate(zip(v, gates.T)):
            d[:,k] = np.asarray([gate[0], 0, 0])# @self.VU[k]
        return d




class kv:
    def __init__(self, v, E = -90.0, N = np.inf, temp = 34.0):
        self.m = self.minf(v)
        self.E = E
        if np.asarray(N).shape == np.asarray(v).shape:
            self.N = N
        else:
            if isinstance(N, (int,float,np.float64, np.int8, np.int16, np.int32)):
                self.N = N*np.ones(v.shape)
            else:
                self.N = N[0]*np.ones(v.shape)

        if np.asarray(temp).shape == np.asarray(v).shape:
            self.qt = 2.3**((temp-21)/10)
        else:
            if isinstance(temp, (int,float,np.float64, np.int8, np.int16, np.int32)):
                self.qt = 2.3**((temp-21)/10)*np.ones(v.shape)
            else:
                self.qt = 2.3**((temp[0]-21)/10)*np.ones(v.shape)

    def update(self, v, dt):
        # self.m = self.m + (self.minf(v)-self.m)/self.mtau(v)*dt
        # self.h = self.h + (self.hinf(v)-self.h)/self.htau(v)*dt
        rand_norm = norm
        rand_norm.random_state=RandomState(seed=None)
        Zm = rand_norm.rvs(0,1, size = self.m.shape)

        R0 = Zm*np.sqrt(np.abs(self.malpha(v)*(1-self.m)+self.mbeta(v)*self.m)/(self.N*dt))    
        self.m = self.m + (1-np.exp(-dt/self.mtau(v)))*(self.minf(v)-self.m) + R0*dt

    def malpha(self, v):
        return 0.02*(v - 25.)/(1. - np.exp(-(v - 25.)/9.))

    def d_malpha(self, v):
    	return 0.02*(1. - np.exp(-(v - 25.)/9.) - 1/9.*(v - 25.)*
    			np.exp(-(v - 25.)/9.))/(1. - np.exp(-(v - 25.)/9.))**2
        
    def mbeta(self, v):
        return -0.002*(v - 25.)/(1. - np.exp((v - 25.)/9.))
    
    
    def d_mbeta(self, v):
    	return -0.002* (1. - np.exp((v - 25.)/9.) + 1/9.*(v - 25.)*
    			np.exp((v - 25.)/9.))/(1. - np.exp((v - 25.)/9.))**2

    def mtau(self,v):
        return (1/(self.malpha(v) + self.mbeta(v)))/self.qt
    
    def minf(self, v):
        return self.malpha(v)/(self.malpha(v) + self.mbeta(v))

    def d_minf(self, v):
        ma = self.malpha(v)
        mb = self.mbeta(v)
        d_ma = self.d_malpha(v)
        d_mb = self.d_mbeta(v)
        return (mb*d_ma-ma*d_mb)/((ma+mb)**2)

    def d_mtau(self, v):
        ma = self.malpha(v)
        mb = self.mbeta(v)
        d_ma = self.d_malpha(v)
        d_mb = self.d_mbeta(v)
        return -1/self.qt*(d_ma + d_mb)/((ma + mb)**2)

    def g_s(self, m):
        """
        scaling term for conductance i = gbar*g_s(t, v)

        """
        return m

    def m_a(self, v, m):
        """
        scaling term for partial gradiant computation s_dm = a*s_v - b*s_m

        """
        d_minf = self.d_minf(v)
        mtau = self.mtau(v)
        d_mtau = self.d_mtau(v)
        minf = self.minf(v)
        return (d_minf/mtau)-(minf-m)/(mtau**2)*d_mtau

    def m_b(self, v, m):
        """
        scaling term for partial gradiant computation s_dm = a*s_v - b*s_m

        """
        return 1/self.mtau(v)

    def m_c(self, v, m):
        """
        scaling term for partial gradiant computation s_dv = c*s_m

        """
        return (self.E - v)

    def m_d(self, v, m):
        """
        scaling term for partial gradiant computation s_dv = c*s_m

        """
        return 1

class im:
    def __init__(self, v, E = -90.0, temp = 34.0):
        self.m = self.minf(v)
        self.E = E

        if np.asarray(temp).shape == np.asarray(v).shape:
            self.qt = 2.3**((temp-21)/10)
        else:
            if isinstance(temp, (int,float,np.float64, np.int8, np.int16, np.int32)):
                self.qt = 2.3**((temp-21)/10)*np.ones(v.shape)
            else:
                self.qt = 2.3**((temp[0]-21)/10)*np.ones(v.shape)

    def update(self, v, dt):
        self.m = self.m + (1-np.exp(-dt/self.mtau(v)))*(self.minf(v)-self.m)

    def minf(self, v):
        return self.malpha(v)/(self.malpha(v) + self.mbeta(v))


    def mtau(self, v):
    	return 1/(self.malpha(v) + self.mbeta(v))/self.qt
    
    
    def malpha(self, v):
    	return 3.3e-3 *np.exp(0.1*(v + 35))

    
    def mbeta(self, v):
    	return 3.3e-3 *np.exp(-0.1*(v + 35))
    
    
    def d_minf(self, v):
        ma = self.malpha(v)
        mb = self.mbeta(v)
        d_ma = self.d_malpha(v)
        d_mb = self.d_mbeta(v)
        return (mb*d_ma-ma*d_mb)/((ma+mb)**2)
    
    
    def d_mtau(self, v):
        ma = self.malpha(v)
        mb = self.mbeta(v)
        d_ma = self.d_malpha(v)
        d_mb = self.d_mbeta(v)
        return -1/self.qt*(d_ma + d_mb)/((ma + mb)**2)
    
    
    def d_malpha(self, v):
    	return 3.3e-4*np.exp(0.1*(v + 35))
    
    
    def d_mbeta(self, v):
    	return -3.3e-4*np.exp(-0.1*(v + 35))

    def g_s(self, m):
        """
        scaling term for conductance i = gbar*g_s(t, v)

        """
        return m

    def m_a(self, v, m):
        """
        scaling term for partial gradiant computation s_dm = a*s_v - b*s_m

        """
        d_minf = self.d_minf(v)
        mtau = self.mtau(v)
        d_mtau = self.d_mtau(v)
        minf = self.minf(v)
        return (d_minf/mtau)-(minf-m)/(mtau**2)*d_mtau

    def m_b(self, v, m):
        """
        scaling term for partial gradiant computation s_dm = a*s_v - b*s_m

        """
        return 1/self.mtau(v)

    def m_c(self, v, m):
        """
        scaling term for partial gradiant computation s_dv = c*s_m

        """
        return self.E - v

    def m_d(self, v, m):
        return 1

class ca:
    def __init__(self, v, E = 120.0, temp = 34.0):
        self.m = self.minf(v)
        self.h = self.hinf(v)
        self.E = E
        if np.asarray(temp).shape == np.asarray(v).shape:
            self.qt = 2.3**((temp-21)/10)
        else:
            if isinstance(temp, (int,float,np.float64, np.int8, np.int16, np.int32)):
                self.qt = 2.3**((temp-21)/10)*np.ones(v.shape)
            else:
                self.qt = 2.3**((temp[0]-21)/10)*np.ones(v.shape)

    def update(self, v, dt):
        # self.m = self.m + (self.minf(v)-self.m)/self.mtau(v)*dt
        # self.h = self.h + (self.hinf(v)-self.h)/self.htau(v)*dt
        self.m = self.m + (1-np.exp(-dt/self.mtau(v)))*(self.minf(v)-self.m)
        self.h = self.h + (1-np.exp(-dt/self.htau(v)))*(self.hinf(v)-self.h)


    def malpha(self, v):
        return (0.055 * (v + 27.))/(1-(np.exp(-(v + 27.)/3.8)))

    def d_malpha(self, v):
        return 0.055*((1. - np.exp(-(v + 27.)/3.8)) - 1/3.8*(v + 27.)*np.exp(-(v + 27.)/3.8))/(1. - np.exp(-(v + 27.)/3.8))**2

    def mbeta(self, v):
        return 0.94*np.exp(-(v + 75)/17)

    def d_mbeta(self, v):
        return -0.94/27*np.exp(-(v + 75)/17)

    def mtau(self,v):
        return (1/(self.malpha(v) + self.mbeta(v)))/self.qt
    
    def minf(self, v):
        return self.malpha(v)/(self.malpha(v) + self.mbeta(v))
    
    def halpha(self, v):
        return 0.000457*np.exp(-(v + 13)/50)

    def d_halpha(self, v):
        return -0.000457/50*np.exp(-(v + 13)/50)

    def hbeta(self, v):
        return 0.0065/(1 + np.exp(-(v + 15)/28))

    def d_hbeta(self, v):
        return 0.0065/28*np.exp(-(v + 15)/28)/(1 + np.exp(-(v + 15)/28))**2

    def htau(self, v):
        return (1/(self.halpha(v) + self.hbeta(v)))/self.qt
    
    def hinf(self, v):
        return self.halpha(v)/(self.halpha(v) + self.hbeta(v))

    def g_s(self, m, h):
        """
        scaling term for conductance i = gbar*g_s(t, v)

        """
        return m**2*h

    def d_minf(self, v):
        ma = self.malpha(v)
        mb = self.mbeta(v)
        d_ma = self.d_malpha(v)
        d_mb = self.d_mbeta(v)
        return (mb*d_ma-ma*d_mb)/((ma+mb)**2)

    def d_mtau(self, v):
        ma = self.malpha(v)
        mb = self.mbeta(v)
        d_ma = self.d_malpha(v)
        d_mb = self.d_mbeta(v)
        return -1/self.qt*(d_ma + d_mb)/((ma + mb)**2)

    def d_hinf(self, v):
        ha = self.halpha(v)
        hb = self.hbeta(v)
        d_ha = self.d_halpha(v)
        d_hb = self.d_hbeta(v)
        return (hb*d_ha-ha*d_hb)/((ha+hb)**2)

    def d_htau(self, v):
        ha = self.halpha(v)
        hb = self.hbeta(v)
        d_ha = self.d_halpha(v)
        d_hb = self.d_hbeta(v)
        return -1/self.qt*(d_ha + d_hb)/((ha + hb)**2)

    def m_a(self, v, m, h):
        """
        scaling term for partial gradiant computation s_dm = a*s_v - b*s_m

        """
        d_minf = self.d_minf(v)
        mtau = self.mtau(v)
        d_mtau = self.d_mtau(v)
        minf = self.minf(v)
        return (d_minf/mtau)-(minf-m)/(mtau**2)*d_mtau

    def m_b(self, v, m, h):
        """
        scaling term for partial gradiant computation s_dm = a*s_v - b*s_m

        """
        return 1/self.mtau(v)

    def h_a(self, v, m, h):
        """
        scaling term for partial gradiant computation s_dh = a*s_v - b*s_h

        """
        d_hinf = self.d_hinf(v)
        htau = self.htau(v)
        d_htau = self.d_htau(v)
        hinf = self.hinf(v)
        return (d_hinf/htau)-(hinf-h)/(htau**2)*d_htau

    def h_b(self, v, m, h):
        """
        scaling term for partial gradiant computation s_dh = a*s_v - b*s_h

        """
        return 1/self.htau(v)

    def m_c(self, v, m, h):
        """
        scaling term for partial gradiant computation s_dv = c*s_m

        """
        return 2*(m*h)*(self.E - v)

    def h_c(self, v, m, h):
        """
        scaling term for partial gradiant computation s_dv = c*s_h

        """
        return (m**2)*(self.E - v)

    def m_d(self, v, m, h):
        """
        scaling term for partial gradiant computation s_dv = c*s_m

        """
        return 2*(m*h)

    def h_d(self, v, m, h):
        """
        scaling term for partial gradiant computation s_dv = c*s_h

        """
        return (m**2)
    



class kca:
    def __init__(self, ca =  1e-4, E = -90.0, temp = 34.0):
        self.m = self.minf(ca)
        self.E = E
        if np.asarray(temp).shape == np.asarray(ca).shape:
            self.qt = 2.3**((temp-21)/10)
        else:
            if isinstance(temp, (int,float,np.float64, np.int8, np.int16, np.int32)):
                self.qt = 2.3**((temp-21)/10)*np.ones(ca.shape)
            else:
                self.qt = 2.3**((temp[0]-21)/10)*np.ones(ca.shape)

    def update(self, ca, dt):
        # self.m = self.m + (self.minf(v)-self.m)/self.mtau(v)*dt
        # self.h = self.h + (self.hinf(v)-self.h)/self.htau(v)*dt
        self.m = self.m + (1-np.exp(-dt/self.mtau(ca)))*(self.minf(ca)-self.m)

    def mtau(self,ca):
        return 1
    
    def minf(self, ca):
        if hasattr(ca, '__len__'):
            for k, ca_i in enumerate(ca):
                if ca_i <= 1e-7:
                    ca[k] = 1e-7
        else:
            if ca <= 1e-7:
                ca = 1e-7
        return  1/(1 + (0.00043/ca)**4.8)

    def g_s(self, m):
        """
        scaling term for conductance i = gbar*g_s(t, v)

        """
        return m

    def d_minf(self, ca, dca):
        return 0.00043*4.8*(0.00043/ca)**3.8*dca/(ca**2)*(self.minf(ca))**2

    def d_mtau(self, ca, dca):
        return 0

    def m_a(self, ca, dca):
        """
        scaling term for partial gradiant computation s_dm = a*s_v - b*s_m

        """
        d_minf = self.d_minf(ca, dca)
        mtau = self.mtau(ca)
        return d_minf/mtau

    def m_b(self, ca = 0, dca = 0):
        """
        scaling term for partial gradiant computation s_dm = a*s_v - b*s_m

        """
        return 1

    def m_c(self, v, m):
        """
        scaling term for partial gradiant computation s_dv = c*s_m

        """
        return self.E - v

    def m_d(self, v, m):
        """
        scaling term for partial gradiant computation s_dv = c*s_m

        """
        return 1

class CaDynamics_E2:
    def __init__(self, ica = 0):
        self.minCai = 1e-4
        self.depth = 0.1
        self.ca = self.minCai

    def update(self, ica, ca, gamma, decay, dt):
        ca = ca + dt*(-10000*ica*gamma/(2*96500*self.depth) - (ca - self.minCai)/decay)
        return ca


