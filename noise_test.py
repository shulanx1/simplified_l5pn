# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:54:23 2023

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






