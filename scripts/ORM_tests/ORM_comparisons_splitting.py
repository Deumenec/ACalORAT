#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 12:28:40 2026

@author: deumenec

Comparison between the simple ORM calculated in ALBA and ALBAII
"""

from pathlib import Path
import numpy as np
import at
import matplotlib.pyplot as plt
import copy 

from ACalORAT import numerical
from ACalORAT import read
from ACalORAT import AnaORM
from ACalORAT import plot_utils
from ACalORAT import ana_utils
from ACalORAT import math_utils

CALC = True
spl = 10 #Number of times the correctors are split

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
    

ROOT = Path(__file__).resolve().parent.parent.parent
SAVE = ROOT / "outputs" /"ORM" /"ORMs_split"

ring0, ind0 = read.ALBA(ROOT  / "data" / "THERING.mat")
ring1, ind1 = read.ALBAII(ROOT  / "data" / "ring_a2.mat")

#Applying the divide method
       
#ADDING kickangles to the ALBAII ring
for i in ind1["cor"]["h"]: ring1[i].KickAngle = np.array([0,0])
for i in ind1["cor"]["v"]: ring1[i].KickAngle = np.array([0,0])

split_dict = split_fam(ring1, ind1["cor"]["v"], spl, ind1)

ind11 = read.find_ind_ALBAII(ring1)

if CALC:
    ORM_AII_ana = ana_utils.ORM(ring1, ind11, "v")
    #Save after calculating
    np.save(SAVE / "ORM_AII_ana", ORM_AII_ana)
    ORM_AI_ana = np.load(SAVE /"ORM_AI_ana.npy")
    ORM_AI_num = np.load(SAVE /"ORM_AI_num.npy")
    ORM_AII_num = np.load(SAVE /"ORM_AII_num.npy")
    
else:
    try:
        ORM_AI_ana = np.load(SAVE /"ORM_AI_ana.npy")
        ORM_AII_ana = np.load(SAVE /"ORM_AII_ana.npy")
        ORM_AI_num = np.load(SAVE /"ORM_AI_num.npy")
        ORM_AII_num = np.load(SAVE /"ORM_AII_num.npy")

        
    except:
        raise ImportError("No hi ha respostes calculades")

c_ana = np.zeros( (len(ind1["bpm"]), len(ind1["cor"]["h"])) )
for i, el in enumerate(c_ana):
    c_ana[:,i] = np.sum(ORM_AII_ana[:,i*spl:(i+1)*spl], axis = 1)/spl

error1 = math_utils.normalized_RMSE(ORM_AI_ana, ORM_AI_num,1)
error2 = math_utils.normalized_RMSE(c_ana, ORM_AII_num,1)

plot_utils.plot_double(error1, error2, "ALBAI", "ALBAII", yaxis = "% RMSD")

    
