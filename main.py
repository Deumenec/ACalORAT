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
results        = 'A2' #A1 for the ALBA lattice and A2 for the ALBAII lattice and CFDA2
direction      = 'v' #v: vertical h: horizontal (SI NOMÉS ES FA EL CÀLCUL D'UNA)
step_exp       =  7
step           =  10**(-step_exp)
read_numerical =  True
dispersion     =  True  #Important ja que sino tot petaria amb la cromaticitat! calcular les matrius amb dispersió.
lin_all        =  False  #To turn off higher order multipoles
max_ind        =  2     #cutoff index in polynomB
RF_corr        =  False
calc_dq        =  False
calc_dCFD      =  True

if RF_corr:
    results += "RFs"
    
###############################################################################
# Reading the lattice parameters
###############################################################################

lattice_path = os.path.join(lattice_folder, lattice_file)
ring, ind_bpm, ind_cor, ind_quad, ind_dip, ind_RF = read.ALBAII(lattice_path)

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
# Calculating the numerical dORMdq if required and saving them
###############################################################################

if read_numerical == False:
    #I add kick angle variable to perform the numerical ORM calculation
    #IMPORTANT, add ind_cor[sub_direction] for ALBA2
    if calc_dq:
        numerical_ORM = numerical.dORM_dq(ring, ind_bpm, ind_cor["v"], ind_quad, step, "v")
        np.save(os.path.join(results,prefix +"v_numdORM_dq"),numerical_ORM)
        for ind in ind_cor["h"]: ring[ind].KickAngle = np.array([0,0])
        numerical_ORM = numerical.dORM_dq(ring, ind_bpm, ind_cor["h"], ind_quad, step, "h")
        np.save(os.path.join(results,prefix + "h_numdORM_dq"),numerical_ORM)
    if calc_dCFD:
        a=0

h_numerical_dORM_dCFD = numerical.dORM_dCFD(ring, ind_bpm, ind_cor, ind_dip, ind_RF, step, "h", num = 5) #In ALBAII all dipoles are CFD!


dORMV = np.load(os.path.join(results,prefix + "v_numdORM_dq.npy"))
dORMH = np.load(os.path.join(results,prefix + "h_numdORM_dq.npy"))

#dORMH_CFD = np.load(os.path.join(results,prefix + "h_numdORM_dCFD.npy"))

#dORM_dCFD =  numerical.dORM_dCFD(ring, ind_bpm, ind_cor["h"], ind_dip, ind_RF, step, "h") #In ALBAII all dipoles are indeed CFD


###############################################################################
#Calculating dORM with thick elements and assessing validity
###############################################################################

#time1 = time.perf_counter()
###### Example calculating the dORM_dq with thin and thick elements!
cORM = AnaORM.AnaORM(ring,"v" ,ind_bpm, ind_cor["v"], ind_quad, ind_dip, np.array([]))
cORM.assign_optics()
cORM.quad.correct_strength()
cORM.bpm.broadcasters(1, 3)
cORM.cor.broadcasters(2, 3)
cORM.quad.broadcasters(0, 3)
thickv = cORM.dRij_dqk_thick23(cORM.bpm, cORM.cor, cORM.quad)
##########################################################

###### Example calculating the dORM_dq with thin and thick elements!
cORM = AnaORM.AnaORM(ring,"h" ,ind_bpm, ind_cor["h"], ind_quad, ind_dip, np.array([]))
cORM.assign_optics()
cORM.dip.correct_entrance()#Corrects optics entrance at dipoles
#cORM.quad.correct_strength()#Acounts for the fact that 
cORM.bpm.broadcasters(1, 4)
cORM.cor.broadcasters(2, 4)
cORM.quad.broadcasters(0, 4)
cORM.dip.broadcasters(3, 4)

thickh = np.sum(cORM.dRij_dqk_thick23(cORM.bpm, cORM.cor, cORM.quad),axis=3 ) + cORM.dRij_dqk_thick23_disp(cORM.bpm, cORM.cor, cORM.quad, cORM.dip)
##########################################################
#time2 = time.perf_counter()
#print(time2-time1)

#plot_utils.plot_both_Zeus(dORMV, dORMH, thickv, thickh)



#Dispersion derivative test!!!


def dispersion(ring):
    #Calculates dispersion in bpms
    all_optics = at.get_optics(ring, refpts = ind_bpm)
    return np.array([i[0] for i in all_optics[2]["dispersion"]])

###########Dispersion test in bpms######################
cORM = AnaORM.AnaORM(ring,"h" ,ind_bpm, ind_cor["h"], ind_quad, ind_dip, np.array([]))
cORM.assign_optics()
cORM.dip.correct_entrance() #Already correcting for the hef
cORM.bpm.broadcasters(0, 2)
cORM.dip.broadcasters(1, 2)
disp0 = cORM.ni_sum(cORM.bpm, cORM.dip)
dispReal = dispersion(ring)
########################################################

plt.plot(disp0)
plt.plot(dispReal)



"""
###############################################################################
#Tests regarding kicker response to CFD activation
###############################################################################

original_orbit = at.find_orbit(ring, refpts=ind_bpm)[1]
(ring[ind_dip[30]]).PolynomB[0]+=0.00000001
uncorrected_orbit = at.find_orbit(ring, refpts=ind_bpm)[1]
correction1, final_orbit1 = numerical.kick_cor(ring , ind_bpm, ind_cor, 0.0000000001, original_orbit)

(ring[ind_dip[30]]).PolynomB[0]-=0.00000002
#Per aquest mètode sembla bastant beneficios trobar la jacobiana amb punts a tots dos costats de la ORM.
#Test for the linearity of the response with respect to kickangles
uncorrected_orbit2 = at.find_orbit(ring, refpts=ind_bpm)[1]
correction2, final_orbit2 = numerical.kick_cor(ring , ind_bpm, ind_cor, 0.0000000001, original_orbit)

oo= np.array([i[0] for i in original_orbit])
uo= np.array([i[0] for i in uncorrected_orbit])
co1= np.array([i[0] for i in final_orbit1])
co2= np.array([i[0] for i in final_orbit2])

math_utils.listPlot([oo, uo, co1, co2], ["original","uncorrected" ,"corrected+", "corrected-"],"Kicker Orbit correction", "orbit_correction")
plt.show()
"""


