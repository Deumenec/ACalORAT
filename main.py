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
lattice_folder = 'lattices' #Important quan treballis amb aquests!
results        = 'A2'  #A1 for the ALBA lattice and A2 for the ALBAII lattice and CFDA2
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

for ind in ind_cor["h"]: ring[ind].KickAngle = np.array([0,0])
for ind in ind_cor["v"]: ring[ind].KickAngle = np.array([0,0])
###############################################################################
# Calculating the numerical dORMdCFD if required and saving them
###############################################################################

if read_numerical == False:
    #I add kick angle variable to perform the numerical ORM calculation
    #IMPORTANT, add ind_cor[sub_direction] for ALBA2
    sub_direction = "v"
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
original_orbit = at.find_orbit6(ring, refpts=ind_bpm)[1]
print("hiii")
(ring[ind_dip[100]]).PolynomB[0]+=0.00000001
uncorrected_orbit = at.find_orbit6(ring, refpts=ind_bpm)[1]
correction, final_orbit = numerical.kick_cor(ring , ind_bpm, ind_cor, 0.0000000001, original_orbit)
oo= np.array([i[0] for i in original_orbit])
uo= np.array([i[0] for i in uncorrected_orbit])
co= np.array([i[0] for i in final_orbit])
math_utils.listPlot([oo, uo, co], ["original","uncorrected" ,"corrected"],"Kicker Orbit correction", "orbit_correction")
plt.show()