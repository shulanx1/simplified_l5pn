# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:21:35 2023

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

dist = np.arange(5,155,10)
colors_d = np.linspace(light, dark, len(dist))
plt.figure()
for (k, d) in enumerate(dist):
    c.dist = d
    a = c.I1I2_a(v1)
    b = c.I2I1_a(v1)
    plt.plot(v1, b/(a+b), color = colors_d[k])

plt.figure()
for (k, d) in enumerate(dist):
    c.dist = d
    a = c.I1I2_a(v1)
    b = c.I2I1_a(v1)
    plt.plot(v1, 1/(a+b), color = colors_d[k])