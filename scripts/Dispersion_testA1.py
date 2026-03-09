# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 10:30:11 2026

@author: dhuerta

Test to check if orbit displacement in BPMs is proportional to dispersion
Also built to check numerically if the derivative of dispersion is well 
calculated at least in the x transverse dimension!
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

    
lattice_file   = 'THERING.mat' #Read ALBA II lattice ring_a2.mat or THERING.mat to read the ALBA one
results        = 'A1' #A1 for the ALBA lattice and A2 for the ALBAII lattice and CFDA2
direction      = 'v' #v: vertical h: horizontal (SI NOMÉS ES FA EL CÀLCUL D'UNA)
step_exp       =  5
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
ring, ind0 = read.ALBA(lattice_path)

#Initial values
val_mcf = get_mcf(ring)
optics = at.get_optics(ring, refpts = range(len(ring)))[2]
disps = np.array([optics["dispersion"][i][0] for i in range(len(ring))])[ind0["bpm"]]
c_orbit0 = at.find_orbit6(ring, refpts=ind0["bpm"])[1] 

#Change ring parameters
ring_freq = ring.get_rf_frequency()
ring.set_cavity(Frequency=ring_freq+step)

#Get new orbit
c_orbit1 = at.find_orbit6(ring, refpts=ind0["bpm"])[1] 

#Calculate derivative with only the horizontal displacement
x0 = np.array([i[0] for i in c_orbit0])
x1 = np.array([i[0] for i in c_orbit1])

disp_num = - (val_mcf*ring_freq )*(x1-x0)/step 

spl = 1

def split_el(ring, i, num):
    """Given a ring element i, the ring is modified to have that element 
    split num times handling frontier and length"""
    split_el = copy.deepcopy(ring[i])
    if hasattr(split_el,"Length"):
        split_el.Length = split_el.Length/num
    else:
        raise TypeError(f"Intentant dividir l'element {i} prim")
    ring.pop(i)
    if hasattr(split_el,"BendingAngle"):
        split_el.BendingAngle = split_el.BendingAngle/num
    for j in range(num): ring.insert(i, copy.deepcopy(split_el), copy_elements=True)
    for j in range(num-1): 
        ring[i+j+1].EntranceAngle = 0
        ring[i+j].ExitAngle = 0        
    return 

def split_fam(ring, ind, num, ind0):
    """"Split a family of elements in a ring each one in a certain number returns a dict """
    i = 0
    ind_c = copy.copy(ind)
    split_dict = {}
    while i<len(ind):
        split_dict.update({ind_c[i]:np.array(range(ind_c[i], ind_c[i]+num))})
        split_el(ring, ind_c[i], num)
        ind_c += num-1
        for fam in ind0:
            if type(fam) == np.ndarray: 
                fam += num-1
        i += 1
    return split_dict

split_dict = split_fam(ring, ind0["dip"], spl, ind0)

ind = read.find_ind_ALBA(ring)

###########Dispersion test in bpms######################
cORM = AnaORM.AnaORM(ring,"h" ,ind)
cORM.assign_optics()

#cORM.dip.correct_entrance() #Already correcting for the hef
#cORM.dip.correct_strength()
cORM.bpm.broadcasters(0, 2)
cORM.dip.broadcasters(1, 2)
disp0 = cORM.ni_sum(cORM.bpm, cORM.dip)
dispReal = optics["dispersion"][ind0["bpm"], 0]
########################################################

plt.plot(dispReal, label = "real")
plt.plot(disp0, label = "analitíca")
plt.title("Dipole originated dispersion")
plt.legend()
plt.show()


"""
##Numerical derivative of dispersion in bpms with respect to quadrupoles######

def disp_i_dk(disp_i, ring, quad, step):
    """
    #For a given quad, we calculate the derivative of dispersion in bpms in the 
    #h transverse dimension with respect to changing its strength.
    #quad: index of the changed quad
"""
    ring = copy.deepcopy(ring)
    ring[quad].K += step
    optics = at.get_optics(ring, refpts=ind["bpm"])
    new_disp_i = optics[2]["dispersion"][:, 0]
    return (new_disp_i-disp_i)/step

di_dk = np.zeros((len(ind["quad"]),len(ind["bpm"]))) 

if False:
    for i in range(len(ind["quad"])):
        di_dk[i] = disp_i_dk(dispReal, ring, ind["quad"][i], 1e-5)
    np.save("di_dk",di_dk )
    
di_dk = np.load("di_dk.npy")

cORM = AnaORM.AnaORM(ring,"h" ,ind)
cORM.assign_optics()
cORM.dip.correct_entrance()
cORM.bpm.broadcasters(1, 3)
cORM.quad.broadcasters(0, 3)
cORM.dip.broadcasters(2, 3)

di_dk_ana = cORM.dni_dqk_sum(cORM.bpm, cORM.dip, cORM.quad)

"""


