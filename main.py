#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 22:49:20 2025

@author: deumenec
Main code to do everyfg!
"""

import os
import numpy as np
import at
import time
import copy

import numerical
import read
import AnaORM
import plot_utils
import math_utils
import matplotlib.pyplot as plt

#os.chdir('Z:\Projectes\AlbaThick') #Set my working directory!
#os.chdir('/Users/deumenec/Documents/Uni/9é semestre/ALBA/Teoria/AlbaThick') #Set my working directory!

###############################################################################
# Parameters to pass for the calculations
###############################################################################

    
lattice_file   = 'ring_a2.mat' #Read ALBA II lattice ring_a2.mat or THERING.mat to read the ALBA one
lattice_folder = 'lattices'
results        = 'A2'  #A1 for the ALBA lattice and A2 for the ALBAII lattice
direction      = 'v' #v: vertical h: horizontal (SI NOMÉS ES FA EL CÀLCUL D'UNA)
step_exp       =  7
step           =  10**(-step_exp)
read_numerical =  True
dispersion     =  True  #Important ja que sino tot petaria amb la cromaticitat! calcular les matrius amb dispersió.
lin_all        =  False  #To turn off higher order multipoles
max_ind        =  2     #cutoff index in polynomB


###############################################################################
# Reading the lattice parameters
###############################################################################

lattice_path = os.path.join(lattice_folder, lattice_file)
ring, ind_bpm, ind_cor, ind_quad, ind_dip = read.ALBAII(lattice_path)

###############################################################################
# Configuration of path name for the different options used
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

###############################################################################
# Calculating the numerical dORMs if required and saving them
###############################################################################

if read_numerical == False:
    #I add kick angle variable to perform the numerical ORM calculation
    #IMPORTANT, add ind_cor[sub_direction] for ALBA2
    sub_direction = "v"
    for ind in ind_cor[sub_direction]: ring[ind].KickAngle = np.array([0,0])
    numerical_ORM = numerical.dORM_dq(ring, ind_bpm, ind_cor[sub_direction], ind_quad, step, sub_direction)
    np.save(os.path.join(results,prefix + sub_direction+ "_numdORM_dq"),numerical_ORM)
    #The other direction
    sub_direction = "h"
    for ind in ind_cor[sub_direction]: ring[ind].KickAngle = np.array([0,0])
    numerical_ORM = numerical.dORM_dq(ring, ind_bpm, ind_cor[sub_direction], ind_quad, step, sub_direction)
    np.save(os.path.join(results,prefix + sub_direction+ "_numdORM_dq"),numerical_ORM)
    
###############################################################################
# Loading saved numerical dORMs to perform comparisons
###############################################################################
    
dORMV = np.load(os.path.join(results,prefix + "v_numdORM_dq.npy"))
dORMH = np.load(os.path.join(results,prefix + "h_numdORM_dq.npy"))


#time1 = time.perf_counter()
###### Example calculating the dORM_dq with thin and thick elements!
cORM = AnaORM.AnaORM(ring,"v" ,ind_bpm, ind_cor["v"], ind_quad, ind_dip, np.array([]))
cORM.assign_optics()
cORM.bpm.broadcasters(1, 3)
cORM.cor.broadcasters(2, 3)
cORM.quad.broadcasters(0, 3)
thickv = cORM.dRij_dqk_thick23(cORM.bpm, cORM.cor, cORM.quad)
vdRij_dqk = cORM.dRij_dqk_thin(cORM.bpm, cORM.cor, cORM.quad)
##########################################################

###### Example calculating the dORM_dq with thin and thick elements!
cORM = AnaORM.AnaORM(ring,"h" ,ind_bpm, ind_cor["h"], ind_quad, ind_dip, np.array([]))
cORM.assign_optics()
cORM.bpm.broadcasters(1, 3)
cORM.cor.broadcasters(2, 3)
cORM.quad.broadcasters(0, 3)
thickh = cORM.dRij_dqk_thick23(cORM.bpm, cORM.cor, cORM.quad)
hdRij_dqk = cORM.dRij_dqk_thin(cORM.bpm, cORM.cor, cORM.quad)
##########################################################
#time2 = time.perf_counter()
#print(time2-time1)

#plot_utils.plot_both(dORMV, dORMH, vdRij_dqk, hdRij_dqk)
#plot_utils.plot_both_Zeus(dORMV, dORMH, vdRij_dqk, hdRij_dqk)

#plot_utils.plot_both(dORMV, dORMH, thickv, thickh)
#plot_utils.plot_both_Zeus(dORMV, dORMH, thickv, thickh)

###########Dispersion test in bpms######################
cORM.bpm.broadcasters(0, 2)
cORM.dip.correct_entrance()
cORM.dip.broadcasters(1, 2)
bpmdispls = cORM.disp_i(cORM.bpm, cORM.dip)
#math_utils.listPlot([cORM.bpm.dispersion ,bpmdispls ], ["bpmDisp", "bpmDispcalc"],title="bpm Dispersions" ,savename = "disps")
########################################################

###########Dispersion test in cors######################
cORM.cor.broadcasters(0, 2)
cORM.dip.broadcasters(1, 2)
cordispls = cORM.disp_i(cORM.cor, cORM.dip)
#math_utils.listPlot([cORM.cor.dispersion ,cordispls ], ["bpmDisp", "bpmDispcalc"],title="bpm Dispersions" ,savename = "disps")
########################################################

###########Dispersion test in dips######################
cORM.dip1 = copy.deepcopy(cORM.dip)
cORM.dip.broadcasters(0, 2)
cORM.dip1.broadcasters(1, 2)
dipdispls = cORM.disp_i(cORM.dip, cORM.dip1)
math_utils.listPlot([cORM.dip.dispersion ,dipdispls ], ["bpmDisp", "bpmDispcalc"],title="bpm Dispersions" ,savename = "disps")
########################################################

ring.disable_6d()
mcf = ring.mcf*ring.circumference
mcf1=  cORM.MCF(cORM.dip, cORM.dip1)

ring.enable_6d()
print(mcf, mcf1)


