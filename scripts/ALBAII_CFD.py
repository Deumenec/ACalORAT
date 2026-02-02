#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 15:35:00 2026

@author: deumenec

Main code to calculate parameters for the derivative of the ORM
"""

import os
import numpy as np
from pathlib import Path
import at
from ACalORAT import numerical
from ACalORAT import read
from ACalORAT import AnaORM
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

SAVE = ROOT / "outputs" / "ALBAII_CFD"

os.chdir(ROOT)
###############################################################################
# Parameters to pass for the calculations
###############################################################################

    
lattice_file   = 'ring_a2.mat' #Read ALBA II lattice ring_a2.mat or THERING.mat to read the ALBA one
results        = 'A2' #A1 for the ALBA lattice and A2 for the ALBAII lattice and CFDA2
step           =  1e-4

p              ={"read_numerical" :  True,
                 "dispersion"     :  True,  #Important ja que sino tot petaria amb la cromaticitat! calcular les matrius amb dispersió.
                 "lin_all"        :  False,  #To turn off higher order multipoles
                 "max_ind"        :  2,      #Cutoff index in polynomB, simplifies the ring for certain calculations
                 "RF_corr"        :  False,
                 "calc_dq"        :  False,
                 "calc_dCFD"      :  True}

    
###############################################################################
# Reading the lattice parameters
###############################################################################

ring, ind_bpm, ind_cor, ind_quad, ind_dip, ind_RF, ind_sex = read.ALBAII(ROOT  / "data" / "ring_a2.mat")

###############################################################################
# Configuration of path name for the different options used
###############################################################################

if p["lin_all"] == True:
    linname = "l"+ str(p["max_ind"])+"_"  
if p["lin_all"] == False:
    linname = "nl_"
    
if p["dispersion"] == True:
    dsname = "d_"    
if p["dispersion"] == False:
    dsname = "nd_" 

prefix = linname + dsname

if p["dispersion"] == False:
    ring.disable_6d()

if p["lin_all"] == True: #DESACTIVA TOTS ELS Sextupols i ordres superiors 
    for element in filter(at.checkattr("PolynomB"), ring):
        #print(element.FamName)
        i=p["max_ind"]
        while(i !=0):
            if len(element.PolynomB)>i:
                element.PolynomB[i] = 0
                i +=1
            else:
                i=0

for ind in ind_cor["h"]: ring[ind].KickAngle = np.array([0,0])
for ind in ind_cor["v"]: ring[ind].KickAngle = np.array([0,0])

###############################################################################
# Calculating the numerical dORMdq if required and saving them
###############################################################################

if p["read_numerical"] == False:
    #I add kick angle variable to perform the numerical ORM calculation
    #IMPORTANT, add ind_cor[sub_direction] for ALBA2
    if p["calc_dq"]:
        numerical_ORM = numerical.dORM_dq(ring, ind_bpm, ind_cor["v"], ind_quad, step, "v")
        np.save(os.path.join(results,prefix +"v_numdORM_dq"),numerical_ORM)
        for ind in ind_cor["h"]: ring[ind].KickAngle = np.array([0,0])
        numerical_ORM = numerical.dORM_dq(ring, ind_bpm, ind_cor["h"], ind_quad, step, "h")
        np.save(os.path.join(results,prefix + "h_numdORM_dq"),numerical_ORM)

pathCFD = "cor_SVD"
#Prompt to calculate numerically the derivative of the response matrix with respect to CFDº


num_dORM_dqH, num_dORM_dqV, dFreq_dCFD, dKicksH_dCFD, dKicksV_dCFD,  = numerical.dORM_dCFD(ring, ind_bpm, ind_cor, ind_dip, ind_RF, step, "h",multithread=True, method="Cor_SVD", num =1) #In ALBAII all dipoles are CFD!

np.save(SAVE /pathCFD /"num_dORM_dqH",num_dORM_dqH)
np.save(SAVE /pathCFD /"num_dORM_dqV",num_dORM_dqV)
np.save(SAVE /pathCFD /"dFreq_dCFD",dFreq_dCFD)
np.save(SAVE /pathCFD /"dKicksH_dCFD",dKicksH_dCFD)
np.save(SAVE /pathCFD /"dKicksV_dCFD",dKicksV_dCFD)




###### Calculating the dORM_dCFD with thin and thick elements!
cORM = AnaORM.AnaORM(ring,"h" ,ind_bpm, ind_cor["h"], ind_quad, ind_dip, np.array([]))
cORM.assign_optics()
cORM.quad.correct_strength()
cORM.bpm.broadcasters(1, 3)
cORM.cor.broadcasters(2, 3)
cORM.dip.broadcasters(0, 3)
ana_dORM_dCFDH0 = ( cORM.dRij_dqk_thick23(cORM.bpm, cORM.cor, cORM.dip) #Term corresponding to the quadrupole in CFD
                   + 0)

##########################################################

cORM = AnaORM.AnaORM(ring,"v" ,ind_bpm, ind_cor["v"], ind_quad, ind_dip, np.array([]))
cORM.assign_optics()
cORM.quad.correct_strength()
cORM.bpm.broadcasters(1, 3)
cORM.cor.broadcasters(2, 3)
cORM.dip.broadcasters(0, 3)
ana_dORM_dCFDV0 = ( cORM.dRij_dqk_thick23(cORM.bpm, cORM.cor, cORM.dip) #Term corresponding to the quadrupole in CFD
                   + 0)




