#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 22:49:20 2025

@author: deumenec
Main code to do everyfg!
"""

import os
from pathlib import Path
import numpy as np
import at
import copy

from ACalORAT import numerical
from ACalORAT import read
from ACalORAT import AnaORM
from ACalORAT import plot_utils




ROOT = Path(__file__).resolve().parent.parent.parent
SAVE = ROOT / "outputs" / "ALBAII_ORM"

###############################################################################
# Parameters to pass for the calculations
###############################################################################

    
lattice_file   = 'ring_a2.mat' #Read ALBA II lattice ring_a2.mat or THERING.mat to read the ALBA one
lattice_folder = 'lattices' #Important quan treballis amb aquests!
results        =  SAVE / 'A2' #A1 for the ALBA lattice and A2 for the ALBAII lattice and CFDA2
direction      = 'v' #v: vertical h: horizontal (SI NOMÉS ES FA EL CÀLCUL D'UNA)
step_exp       =  4
step           =  10**(-step_exp)
read_numerical =  True
dispersion     =  False  #Important ja que sino tot petaria amb la cromaticitat! calcular les matrius amb dispersió.
lin_all        =  True  #To turn off higher order multipoles
max_ind        =  2     #cutoff index in polynomB
RF_corr        =  False
calc_dq        =  True

    
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


spl = 1 #Number of times the quadrupoles are split

def split_el(ring, i, num):
    """Given a ring element i, the ring is modified to have that element 
    split num times handling frontier and length"""
    split_el = copy.deepcopy(ring[i])
    if hasattr(split_el,"Length"):
        split_el.Length = split_el.Length/num
    else:
        raise TypeError(f"Intentant dividir l'element {i} prim")
    ring.pop(i)
    for j in range(num): ring.insert(i, split_el, copy_elements=True )
    return 


def split_fam(ring, ind, num, ind0):
    """"Split a family of elements in a ring each one in a certain number returns a dict """
    i = 0
    ind_c = copy.copy(ind)
    split_dict = {}
    while i<len(ind):
        split_dict.update({ind_c[i]:np.array(range(ind_c[i], ind_c[i]+num))})
        split_el(ring, ind_c[i], num)
        ind_c += num-1
        for fam in ind0:
            if type(fam) == np.ndarray: 
                fam += num-1
        i += 1
    return split_dict


split_dict = split_fam(ring, ind["quad"], spl, ind)

ind = read.find_ind_ALBAII(ring)


#time1 = time.perf_counter()
###### Example calculating the dORM_dq with thin and thick elements!
cORM = AnaORM.AnaORM(ring,"v" ,ind)
cORM.assign_optics()
cORM.bpm.broadcasters(1, 3)
cORM.cor.broadcasters(2, 3)
cORM.quad.broadcasters(0, 3)

thickv = cORM.dRij_dqk_thick23(cORM.bpm, cORM.cor, cORM.quad)

thickv_bo = np.zeros((97,176, 176))
for i in range(97):
    thickv_bo[i] = np.sum(thickv[i*spl:(i+1)*spl], axis = 0)
##########################################################

###### Example calculating the dORM_dq with thin and thick elements!
cORM = AnaORM.AnaORM(ring,"h", ind)
cORM.assign_optics()
cORM.dip.correct_entrance()#Corrects optics entrance at dipoles
#cORM.quad.correct_strength()#Acounts for the fact that 
cORM.bpm.broadcasters(1, 4)
cORM.cor.broadcasters(2, 4)
cORM.quad.broadcasters(0, 4)
#cORM.dip.broadcasters(3, 4)

thickh = np.sum(cORM.dRij_dqk_thick23(cORM.bpm, cORM.cor, cORM.quad),axis=3 ) #+ cORM.dRij_dqk_thick23_disp(cORM.bpm, cORM.cor, cORM.quad, cORM.dip)

thickh_bo = np.zeros((97,176, 176))
for i in range(97):
    thickh_bo[i] = np.sum(thickh[i*spl:(i+1)*spl], axis = 0)
    
##########################################################

plot_utils.plot_both_Zeus(dORMV[0:-1], dORMH, thickv_bo[0:-1], thickh_bo)


