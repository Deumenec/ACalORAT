# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:15:19 2026

@author: dhuerta

Script testing the analytical formula for the dispersion derivative with respect to
changing the dipole component of a CFD only.
"""

import os
from pathlib import Path
import numpy as np
import at
import at
import copy

from ACalORAT import numerical
from ACalORAT import read
from ACalORAT import AnaORM
from ACalORAT import plot_utils

ROOT = Path(__file__).resolve().parent.parent
SAVE = ROOT / "outputs" / "ALBAII_CFD_NOFEEDBACK_ONLY_BEND"

###############################################################################
# Parameters to pass for the calculations                                     #
###############################################################################

    
lattice_file   = 'ring_a2.mat' #Read ALBA II lattice ring_a2.mat or THERING.mat to read the ALBA one
lattice_folder = 'lattices' #Important quan treballis amb aquests!
results        =  SAVE  #A1 for the ALBA lattice and A2 for the ALBAII lattice and CFDA2
direction      = 'h' #v: vertical h: horizontal (SI NOMÉS ES FA EL CÀLCUL D'UNA)
step_exp       =  8
step           =  10**(-step_exp)
read_numerical =  True
dispersion     =  False
lin_all        =  False  #To turn off higher order multipoles
max_ind        =  2     #cutoff index in polynomB

###############################################################################
# Reading the lattice parameters
###############################################################################

ring, ind = read.ALBAII(ROOT  / "data" / "ring_a2.mat")


###############################################################################
# Configuration of file name for the different options used
###############################################################################

if lin_all == True:
    linname = "l"+ str(max_ind)+"_"  
if lin_all == False:
    linname = "nl_"
    
if dispersion == True:
    dsname = "d_"    
if dispersion == False:
    dsname = "nd_" 

prefix = linname + dsname

if dispersion == False:
    ring.disable_6d()

if lin_all == True: #DESACTIVA TOTS ELS Sextupols i ordres superiors 
    for element in filter(at.checkattr("PolynomB"), ring):
        #print(element.FamName)
        i=max_ind
        while(i !=0):
            if len(element.PolynomB)>i:
                element.PolynomB[i] = 0
                i +=1
            else:
                i=0

for i in ind["cor"]["h"]: ring[i].KickAngle = np.array([0,0])
for i in ind["cor"]["v"]: ring[i].KickAngle = np.array([0,0])

###############################################################################
# Calculating the numerical dispersion derivative with respect to changing dip in CFDs
###############################################################################


def disp_i_dk(ring, quad):
    """
    For a given quad, we calculate the derivative of dispersion in bpms in the 
    h transverse dimension with respect to changing its strength.
    quad: index of the changed quad
    """
    ring = copy.deepcopy(ring)
    ring[quad].PolynomB[0] += step
    optics = at.get_optics(ring, refpts=ind["bpm"])
    new_disp_i = optics[2]["dispersion"][:, 0]
    
    ring[quad].PolynomB[0] -= 2*step
    optics = at.get_optics(ring, refpts=ind["bpm"])
    new_disp_j = optics[2]["dispersion"][:, 0]
    
    return (new_disp_i-new_disp_j)/(2*step)

"""   
di_dk = np.zeros((len(ind["dip"]),len(ind["bpm"]))) 


if False:
    for i in range(len(ind["dip"])):
        di_dk[i] = disp_i_dk(ring, ind["dip"][i])
    np.save(SAVE / "di_dk",di_dk )
    
di_dk = np.load(SAVE / "di_dk.npy")




cORM = AnaORM.AnaORM(ring,"h", ind)
cORM.assign_optics()
cORM.bpm.broadcasters(1, 2)
cORM.dip.broadcasters(0, 2)
cORM.cor.broadcasters(0, 2)

di_dk_num = -(cORM.dip.LengthB**()) * cORM.Rab_thick2_K(cORM.bpm, cORM.dip)
denergy_test   = cORM.ddip_denergy(cORM.dip)[:, None] *cORM.bpm.dispersionB

"""

def disp_i_dkick(ring, cor):
    """
    For a given quad, we calculate the derivative of dispersion in bpms in the 
    h transverse dimension with respect to changing its strength.
    quad: index of the changed quad
    """
    ring = copy.deepcopy(ring)
    ring[cor].KickAngle[0] += step
    optics = at.get_optics(ring, refpts=ind["bpm"])
    new_disp_i = optics[2]["dispersion"][:, 0]
    
    ring[cor].KickAngle[0] -= 2*step
    optics = at.get_optics(ring, refpts=ind["bpm"])
    new_disp_j = optics[2]["dispersion"][:, 0]
    
    return (new_disp_i-new_disp_j)/(2*step)

di_dk = np.zeros((len(ind["cor"]["h"]),len(ind["bpm"]))) 


if True:
    ring.disable_6d()
    for i in range(len(ind["cor"]["h"])):
        di_dk[i] = disp_i_dkick(ring, ind["cor"]["h"][i])
    np.save(SAVE / "di_dk",di_dk )
    ring.enable_6d()
    
di_dk = np.load(SAVE / "di_dk.npy")

cORM = AnaORM.AnaORM(ring,"h", ind)
cORM.assign_optics()
cORM.bpm.broadcasters(1, 2)
cORM.cor.broadcasters(0, 2)

di_dk_num = cORM.Rab_thick2_(cORM.bpm, cORM.cor)

