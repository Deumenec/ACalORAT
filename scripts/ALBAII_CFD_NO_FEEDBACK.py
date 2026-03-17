#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar Mar  12 12:02:20 2026

@author: deumenec

Code testing if the formula for the Orbit Response Matrix is working well for
CFDs changing both quad and dipole moment as fixed by the required ratio!

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
SAVE = ROOT / "outputs" / "ALBAII_CFD_NOFEEDBACK_YES_BEND"

###############################################################################
# Parameters to pass for the calculations                                     #
###############################################################################

    
lattice_file   = 'ring_a2.mat' #Read ALBA II lattice ring_a2.mat or THERING.mat to read the ALBA one
lattice_folder = 'lattices' #Important quan treballis amb aquests!
results        =  SAVE  #A1 for the ALBA lattice and A2 for the ALBAII lattice and CFDA2
direction      = 'h' #v: vertical h: horizontal (SI NOMÉS ES FA EL CÀLCUL D'UNA)
step_exp       =  5
step           =  10**(-step_exp)
read_numerical =  True
dispersion     =  False  #Important ja que sino tot petaria amb la cromaticitat! calcular les matrius amb dispersió.
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
# Calculating the numerical dORMdq if required and saving them
###############################################################################

if read_numerical == False:
    #I add kick angle variable to perform the numerical ORM calculation
    #IMPORTANT, add ind_cor[sub_direction] for ALBA2
    for i in ind["cor"]["h"]: ring[i].KickAngle = np.array([0,0])
    numerical_ORM = numerical.dORM_dq_pure(ring, ind["bpm"], ind["cor"]["h"], ind["CFD"], step, "h")
    np.save(os.path.join(results,prefix +"h_numdORM_dq"),numerical_ORM)
    numerical_ORM = numerical.dORM_dq_pure(ring, ind["bpm"], ind["cor"]["v"], ind["CFD"], step, "v")
    np.save(os.path.join(results,prefix +"v_numdORM_dq"),numerical_ORM)


dORMH = np.load(os.path.join(results,prefix + "h_numdORM_dq.npy"))
dORMV = np.load(os.path.join(results,prefix + "v_numdORM_dq.npy"))

###############################################################################
#Calculating dORM with thick elements and assessing validity
###############################################################################

#Energy response without sextupoles!

dRij_dEnergy = numerical.quickdORMdEnergy(ring, ind)


cORM = AnaORM.AnaORM(ring,"h", ind)
cORM.assign_optics()
cORM.bpm.broadcasters(0, 3)
cORM.cor.broadcasters(1, 3)
cORM.CFD.broadcasters(2, 3)

denergy = cORM.ddip_denergy(cORM.dip)


thickh = (cORM.dRij_dqk_thick23_master(cORM.bpm, cORM.cor, cORM.CFD) 
          + 0*cORM.dRij_dqk_thick23_disp(cORM.bpm, cORM.cor, cORM.CFD)
          + cORM.dRij_dk_energy_term(cORM.bpm, cORM.cor, cORM.dip, dRij_dEnergy["h"], denergy)) 





cORM = AnaORM.AnaORM(ring,"v", ind)
cORM.assign_optics()
cORM.bpm.broadcasters(0, 3)
cORM.cor.broadcasters(1, 3)
cORM.dip.broadcasters(2, 3)

thickv = (cORM.dRij_dqk_thick23_master(cORM.bpm, cORM.cor, cORM.dip)
          + cORM.dRij_dk_energy_term(cORM.bpm, cORM.cor, cORM.dip, dRij_dEnergy["v"], denergy)
          - cORM.dRij_dk_fringe(cORM.bpm, cORM.cor, cORM.dip))

aa = cORM.dRij_dk_fringe(cORM.bpm, cORM.cor, cORM.dip)

##########################################################
# Comparisons
##########################################################
dORMV = np.transpose(dORMV, (2,1,0))
dORMH = np.transpose(dORMH, (2,1,0))


##TESTS###
dORMV = np.transpose(dORMV, (2,1,0))
dORMH = np.transpose(dORMH, (2,1,0))
thickv = np.transpose(thickv, (2,1,0))
thickh = np.transpose(thickh, (2,1,0))

plot_utils.plot_both_Zeus(dORMV, dORMH, thickv, thickh)




