# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 12:37:59 2026

@author: dhuerta

Calculating the derivative of the Orbit Response Matrix numerically with dispersion
but without sextupoles in the ALBAII lattice and testing analytical formulas.
"""

import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import at
import copy 
from ACalORAT import numerical
from ACalORAT import read
from ACalORAT import AnaORM
from ACalORAT import plot_utils

ROOT = Path(__file__).resolve().parent.parent

SAVE = ROOT / "outputs" / "ALBAII_CFD_no_sext"

os.chdir(ROOT)
if not os.path.exists(SAVE):
    os.mkdir(SAVE)

###############################################################################
# Parameters to pass for the calculations
###############################################################################

    
lattice_file   = 'ring_a2.mat' #Read ALBA II lattice ring_a2.mat or THERING.mat to read the ALBA one
results        = 'A2' #A1 for the ALBA lattice and A2 for the ALBAII lattice and CFDA2
step           =  1e-4

p              ={"lin_all"        :  True,  #To turn off higher order multipoles
                 "max_ind"        :  2,      #Cutoff index in polynomB, simplifies the ring for certain calculations
                 "calculate"      :  False}

    
###############################################################################
# Reading the lattice parameters
###############################################################################

ring, ind = read.ALBAII(ROOT  / "data" / "ring_a2.mat")


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

#Adding kickangles to the correctors in ALBAII
for i in ind["cor"]["h"]: ring[i].KickAngle = np.array([0,0])
for i in ind["cor"]["v"]: ring[i].KickAngle = np.array([0,0])

###############################################################################
# Calculating the numerical dORMdCFD if required and saving them
###############################################################################

pathCFD = "Cor_SVD"
#Prompt to calculate numerically the derivative of the response matrix with respect to CFDº

if  p["calculate"]:
    if not os.path.exists(SAVE / pathCFD):
        os.mkdir(SAVE / pathCFD)
    num_dORM_dqH, num_dORM_dqV, dFreq_dCFD, dKicksH_dCFD, dKicksV_dCFD, x_sex = numerical.dORM_dCFD(ring, ind, step ,multithread=True, method="Cor_SVD", num = 10) #In ALBAII all dipoles are CFD!
    np.save(SAVE /pathCFD /"num_dORM_dqH",num_dORM_dqH)
    np.save(SAVE /pathCFD /"num_dORM_dqV",num_dORM_dqV)
    np.save(SAVE /pathCFD /"dFreq_dCFD",dFreq_dCFD)
    np.save(SAVE /pathCFD /"dKicksH_dCFD",dKicksH_dCFD)
    np.save(SAVE /pathCFD /"dKicksV_dCFD",dKicksV_dCFD)
    np.save(SAVE /pathCFD /"x_sex",x_sex)

else:
    try:
        num_dORM_dqH = np.load(SAVE /pathCFD /"num_dORM_dqH.npy")
        num_dORM_dqV = np.load(SAVE /pathCFD /"num_dORM_dqV.npy")
        dFreq_dCFD   = np.load(SAVE /pathCFD /"dFreq_dCFD.npy")
        dKicksH_dCFD = np.load(SAVE /pathCFD /"dKicksH_dCFD.npy")
        dKicksV_dCFD = np.load(SAVE /pathCFD /"dKicksV_dCFD.npy")
        x_sex        = np.load(SAVE /pathCFD /"x_sex.npy")
    except:
        raise ImportError("No hi ha respostes calculades")

optics = at.get_optics(ring, refpts = range(len(ring)))[2]

#Kicktests:
#Tests if the kicks performed by the numerical calculation respect the dispersion sum condition

corx_disp = optics["dispersion"][ind["cor"]["h"]][:,0]
cory_disp = optics["dispersion"][ind["cor"]["v"]][:,0]



feedx = np.sum(corx_disp*dKicksH_dCFD, axis = 1)
feedy = np.sum(cory_disp*dKicksV_dCFD, axis = 1)
feedxm = np.sum(dKicksH_dCFD, axis = 1)
feedym = np.sum(dKicksV_dCFD, axis = 1)


cORM = AnaORM.AnaORM(ring,"v", ind) #Automatically uses the vertical correctors!
cORM.assign_optics()
#Propagate optics along the dip to get the exit

cORM.dip.correct_strength() #Correct entrance and effective force for the calculations in the CFD
cORM.bpm.broadcasters(1, 3)
cORM.cor.broadcasters(2, 3)
cORM.dip.broadcasters(0, 3)

#Terms for the energy perturbation:    
cORM.bpm2 = copy.deepcopy(cORM.bpm)
cORM.cor2 = copy.deepcopy(cORM.cor)
cORM.dip2 = copy.deepcopy(cORM.dip)
cORM.dip2.average()
cORM.add_element("allQuad", ind["all_quad"], "h")
cORM.add_element("corH", ind["cor"]["h"], "h")
cORM.bpm2.broadcasters(0, 4)
cORM.cor2.broadcasters(1, 4)
cORM.corH.broadcasters(1, 4)
cORM.dip2.broadcasters(2, 4)
cORM.allQuad.broadcasters(3,4)

ana_dORM_dCFDV0 = (cORM.dRij_dqk_thick23(cORM.bpm, cORM.cor, cORM.dip)         #Term corresponding to the quadrupole in CFD with a factor due to the extra change due to change in effective quadrupole because of the change in dipole moment
                  )# +cORM.dRij_dCFD_energy(cORM.bpm2, cORM.cor2, cORM.dip2, cORM.allQuad))    #Term for energy change
                   
                   
                   
                   
plot_utils.plot_both_Zeus(num_dORM_dqV, num_dORM_dqV,ana_dORM_dCFDV0[0:10] , ana_dORM_dCFDV0[0:10]) #THE CHANGE IN ENERGY IS REALLY NOISY but who cares ):
#Cool plot with the kicker activation!
#plot_utils.rainbow_plot(dKicksH_dCFD[:], SAVE / "plots" , NAME = "rainbow_kicks.pdf" )
CFD_RATIO = [ring[i].BendingAngle / ring[i].Length*ring[i].EntranceAngle for i in ind["CFD"] ]
plt.plot(CFD_RATIO[0:10])
