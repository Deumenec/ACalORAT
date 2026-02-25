# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 10:30:11 2026

@author: dhuerta

Test to check if orbit displacement in BPMs is proportional to dispersion
"""

import os
import numpy as np
from pathlib import Path
import at
import time
import copy

from ACalORAT import numerical
from ACalORAT import read
from ACalORAT import AnaORM
from ACalORAT import plot_utils
from ACalORAT import math_utils

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

SAVE = ROOT / "outputs" / "ALBAII_CFD_no_sext"

os.chdir(ROOT)
if not os.path.exists(SAVE):
    os.mkdir(SAVE)


###############################################################################
# Parameters to pass for the calculations
###############################################################################
def get_mcf(ring):
    if (ring.is_6d==True):
        ring.disable_6d()
        mcf = ring.mcf
        ring.enable_6d()
        return mcf
    return ring.mcf

    
lattice_file   = 'ring_a2.mat' #Read ALBA II lattice ring_a2.mat or THERING.mat to read the ALBA one
results        = 'A2' #A1 for the ALBA lattice and A2 for the ALBAII lattice and CFDA2
direction      = 'v' #v: vertical h: horizontal (SI NOMÉS ES FA EL CÀLCUL D'UNA)
step_exp       =  3
step           =  10**(-step_exp)
read_numerical =  True
dispersion     =  True  #Important ja que sino tot petaria amb la cromaticitat! calcular les matrius amb dispersió.
lin_all        =  False  #To turn off higher order multipoles
max_ind        =  2     #cutoff index in polynomB
RF_corr        =  False
calc_dq        =  False
calc_dCFD      =  True


###############################################################################
# Reading the lattice parameters
###############################################################################

lattice_path = ROOT / "data" /lattice_file
ring, ind = read.ALBAII(lattice_path)

#Initial values
val_mcf = get_mcf(ring)
optics = at.get_optics(ring, refpts = range(len(ring)))[2]
disps = np.array([optics["dispersion"][i][0] for i in range(len(ring))])[ind["bpm"]]
c_orbit0 = at.find_orbit6(ring, refpts=ind["bpm"])[1] 

#Change ring parameters
ring_freq = ring.get_rf_frequency()
ring.set_cavity(Frequency=ring_freq+step)

#Get new orbit
c_orbit1 = at.find_orbit6(ring, refpts=ind["bpm"])[1] 

#Calculate derivative with only the horizontal displacement
x0 = np.array([i[0] for i in c_orbit0])
x1 = np.array([i[0] for i in c_orbit1])

disp_num = - (val_mcf*ring_freq )*(x1-x0)/step 


###########Dispersion test in bpms######################
cORM = AnaORM.AnaORM(ring,"h" ,ind)
cORM.assign_optics()
cORM.dip.correct_entrance() #Already correcting for the hef
cORM.bpm.broadcasters(0, 2)
cORM.dip.broadcasters(1, 2)
disp0 = cORM.ni_sum(cORM.bpm, cORM.dip)
dispReal = dispersion(ring)
########################################################

#plt.plot(disp0)
#plt.plot(dispReal)



