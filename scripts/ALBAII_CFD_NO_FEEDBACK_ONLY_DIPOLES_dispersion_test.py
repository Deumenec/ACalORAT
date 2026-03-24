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
step_exp       =  9
step           =  10**(-step_exp)
compute        =  False
dispersion     =  True
lin_all        =  True  #To turn off higher order multipoles
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

 
di_dk = np.zeros((len(ind["dip"]),len(ind["bpm"]))) 


if compute:
    for i in range(len(ind["dip"])):
        di_dk[i] = disp_i_dk(ring, ind["dip"][i])
    np.save(SAVE / "di_dk",di_dk )
    
di_dk = np.load(SAVE / "di_dk.npy")




cORM = AnaORM.AnaORM(ring,"h", ind)
cORM.assign_optics()
cORM.bpm.broadcasters(1, 2)
cORM.dip.broadcasters(0, 2)
cORM.quad.broadcasters(0, 2)

di_dk_ana =  - (cORM.Rab_thick2_K(cORM.bpm, cORM.dip) + cORM.Rab_thick2_disp_K(cORM.bpm, cORM.dip))*cORM.dip.LengthB

aa = cORM.Rab_thick2_disp_K(cORM.bpm, cORM.dip)

cORM = AnaORM.AnaORM(ring,"h", ind)
cORM.assign_optics()
cORM.bpm.broadcasters(1, 3)
cORM.dip.broadcasters(0, 3)
cORM.quad.broadcasters(2, 3)
cORM.CFD.broadcasters(2, 3)

denergy   = cORM.ddip_denergy(cORM.dip)[:, None, None]*cORM.dip.KB/cORM.dip.BendB

quad1 = cORM.dni_dqk_integral(cORM.bpm, cORM.CFD)
quad2 = cORM.dni_dqk_integral(cORM.bpm, cORM.quad)

delta_K_CFD  = -cORM.CFD.KB * denergy
delta_K_quad = -cORM.quad.KB * denergy

# Multiply the derivative by Delta K and sum!
chromatic_disp =np.real( np.sum(quad1 * delta_K_CFD, axis=2) 
                 + np.sum(quad2 * delta_K_quad, axis=2) )

di_dk_test = di_dk_ana + chromatic_disp


#def disp_i_dkick(ring, cor):
    

"""
    For a given quad, we calculate the derivative of dispersion in bpms in the 
    h transverse dimension with respect to changing its strength.
    quad: index of the changed quad
    """
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


if compute:
    for i in range(len(ind["cor"]["h"])):
        di_dk[i] = disp_i_dkick(ring, ind["cor"]["h"][i])
    np.save(SAVE / "di_dk",di_dk )
    
di_dk = np.load(SAVE / "di_dk.npy")

cORM = AnaORM.AnaORM(ring,"h", ind)
cORM.assign_optics()
cORM.bpm.broadcasters(1, 2)
cORM.cor.broadcasters(0, 2)

di_dk_num = cORM.Rab_thick2_(cORM.bpm, cORM.cor)
"""

