#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 22:49:20 2025

@author: deumenec
Code to calculate the ORM Jacobian in normal quadrupoles in the ALBAI lattice,
it allows to benchmark formulas in a controled enviroment to provide some peace of mind.
"""

import os
from pathlib import Path
import numpy as np
import at

from ACalORAT import numerical
from ACalORAT import read
from ACalORAT import AnaORM
from ACalORAT import plot_utils

ROOT = Path(__file__).resolve().parent.parent
SAVE = ROOT / "outputs" / "ALBAI_ORM"

###############################################################################
# Parameters to pass for the calculations
###############################################################################

    
lattice_file   = 'THERING.mat' #Read ALBA II lattice ring_a2.mat or THERING.mat to read the ALBA one
lattice_folder = 'lattices' #Important quan treballis amb aquests!
results        =  SAVE / 'A1' #A1 for the ALBA lattice and A2 for the ALBAII lattice and CFDA2
direction      = 'v' #v: vertical h: horizontal (SI NOMÉS ES FA EL CÀLCUL D'UNA)
step           =  1e-5
read_numerical =  True   #To calculate the jacobians numerically
dispersion     =  True  #Important ja que sino tot petaria amb la cromaticitat! calcular les matrius amb dispersió.
lin_all        =  False #To turn off higher order multipoles
max_ind        =  2     #cutoff index in polynomB
RF_corr        =  False
calc_dq        =  True

    
###############################################################################
# Reading the lattice parameters
###############################################################################

ring, ind = read.ALBA(ROOT  / "data" / "THERING.mat")

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
    if calc_dq:
        numerical_ORM = numerical.dORM_dq(ring, ind["bpm"], ind["cor"]["v"], ind["quad"], step, "v")
        np.save(os.path.join(results,prefix +"v_numdORM_dq"),numerical_ORM)
        
        for i in ind["cor"]["h"]: ring[i].KickAngle = np.array([0,0])
        numerical_ORM = numerical.dORM_dq(ring, ind["bpm"], ind["cor"]["h"], ind["quad"], step, "h")
        np.save(os.path.join(results,prefix + "h_numdORM_dq"),numerical_ORM)
        

dORMV = np.load(os.path.join(results,prefix + "v_numdORM_dq.npy"))
dORMH = np.load(os.path.join(results,prefix + "h_numdORM_dq.npy"))

###############################################################################
#Calculating dORM with thick elements and assessing validity
###############################################################################

#time1 = time.perf_counter()
###### Example calculating the dORM_dq with thin and thick elements!
cORM = AnaORM.AnaORM(ring,"v" ,ind)
cORM.assign_optics()
cORM.bpm.broadcasters(1, 3)
cORM.cor.broadcasters(2, 3)
cORM.quad.broadcasters(0, 3)
thickv = cORM.dRij_dqk_thick23(cORM.bpm, cORM.cor, cORM.quad)
##########################################################

###### Example calculating the dORM_dq with thin and thick elements!
cORM = AnaORM.AnaORM(ring,"h", ind)
cORM.assign_optics()
cORM.dip.correct_entrance()#Corrects optics entrance at dipoles
#cORM.quad.correct_strength()#Acounts for the fact that 
cORM.bpm.broadcasters(1, 4)
cORM.cor.broadcasters(2, 4)
cORM.quad.broadcasters(0, 4)
cORM.dip.broadcasters(3, 4)

thickh = np.sum(cORM.dRij_dqk_thick23(cORM.bpm, cORM.cor, cORM.quad),axis=3 ) #+ cORM.dRij_dqk_thick23_disp(cORM.bpm, cORM.cor, cORM.quad, cORM.dip)
##########################################################

plot_utils.plot_both_Zeus(dORMV, dORMH, thickv, thickh)

cORM = AnaORM.AnaORM(ring,"v", ind)
cORM.assign_optics()
cORM.bpm.broadcasters(0, 2)
cORM.cor.broadcasters(1, 2)
aaORMV = cORM.Rab_thick2_(cORM.bpm, cORM.cor)

aaORMV2 = cORM.Rab_thick2_disp(cORM.bpm, cORM.cor)

sORMV = numerical.dORM_dq_semi(ring, ind, 1e-5, "v")
sORMH = numerical.dORM_dq_semi(ring, ind, 1e-5, "h")

plot_utils.plot_both_Zeus(dORMV, dORMH, sORMV, sORMH)
"""
RespV = at.latticetools.OrbitResponseMatrix(ring, "v", ind["bpm"], ind["cor"]["v"])
RespV.build_tracking() # FIX: build_tracking returns None
aanumORMV = RespV.response
"""

