 # -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 10:30:11 2026

@author: dhuerta

Test to check if orbit displacement in BPMs is proportional to dispersion
Also built to check numerically if the derivative of dispersion is well 
calculated at least in the x transverse dimension! Without Correctors

We see how the formula for dispersion in ALBAII is perfect!
"""

import os
import numpy as np
from pathlib import Path
import at
import copy


from ACalORAT import read
from ACalORAT import AnaORM


import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

SAVE = ROOT / "outputs" / "ALBAII_CDF_quad_disp"

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
step_exp       =  5
step           =  10**(-step_exp)
calculate      =  False


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
#cORM.dip.correct_strength()

cORM.bpm.broadcasters(0, 2)
cORM.dip.broadcasters(1, 2)
disp0 = cORM.ni_sum(cORM.bpm, cORM.dip)
dispReal = optics["dispersion"][ind["bpm"], 0]
########################################################

plt.plot(dispReal, label = "Numerical")
plt.plot(disp0, label = "Analytical")
plt.title("Dipole originated dispersion")
plt.xlabel("BPM")
plt.legend()
plt.show()


##Numerical derivative of dispersion in bpms with respect to quadrupoles######

def disp_i_dk(disp_i, ring, quad, step):
    """
    For a given quad, we calculate the derivative of dispersion in bpms in the 
    h transverse dimension with respect to changing its strength.
    quad: index of the changed quad
    """
    ring = copy.deepcopy(ring)
    ring[quad].PolynomB[1] += step
    optics = at.get_optics(ring, refpts=ind["bpm"])
    new_disp_i = optics[2]["dispersion"][:, 0]
    
    ring[quad].PolynomB[1] -= 2*step
    optics = at.get_optics(ring, refpts=ind["bpm"])
    new_disp_j = optics[2]["dispersion"][:, 0]
    
    return (new_disp_i-new_disp_j)/(2*step)

di_dk = np.zeros((len(ind["dip"]),len(ind["bpm"]))) 


if calculate:
    for i in range(len(ind["dip"])):
        di_dk[i] = disp_i_dk(dispReal, ring, ind["dip"][i], step)
    np.save(SAVE / "di_dk",di_dk )
    
di_dk = np.load(SAVE / "di_dk.npy")

ind = read.find_ind_ALBAII(ring)



#########TEST USING THE NEW INTEGRAL FORMULA FOR DISPERSION DERIVATIVE
ring, ind = read.ALBAII(lattice_path)

cORM2 =  AnaORM.AnaORM(ring,"h" ,ind)
cORM2.assign_optics()
cORM2.CFD.correct_entrance()
cORM2.bpm.broadcasters(1, 2)
cORM2.CFD.broadcasters(0, 2)

di_dk_ana = cORM2.dni_dqk_integral(cORM2.bpm, cORM2.CFD)

CFD = 2

plt.plot(di_dk[:, CFD], label = "Numerical")
plt.plot(di_dk_ana[:, CFD], label = "Analytical")
plt.title(f"Derivative of dispersion by changing B[1] in CFD {CFD}")
plt.xlabel("BPM")
plt.legend()
plt.show()

error = np.std(di_dk-di_dk_ana, axis = 0)

plt.plot(error)
plt.title("Dispersion derivative standard deviation")
plt.xlabel("BPM")
plt.show()



