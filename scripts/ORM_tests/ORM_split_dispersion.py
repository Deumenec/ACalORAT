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

#Comprovem com la formula per la mitjana de la dispersió funciona perfecte 

CALC = True
spl = 100 #Number of times the correctors are split

def split_el(ring, i, num):
    """Given a ring element i, the ring is modified to have that element 
    split num times handling frontier and length"""
    split_el = copy.deepcopy(ring[i])
    if hasattr(split_el, "BendingAngle"): 
        split_el.BendingAngle = split_el.BendingAngle/num
    if hasattr(split_el,"Length"):
        split_el.Length = split_el.Length/num
    else:
        raise TypeError(f"Intentant dividir l'element {i} prim")
    ring.pop(i)
    for j in range(num): ring.insert(i, copy.deepcopy(split_el), copy_elements=True)
    
    if hasattr(ring[i], "EntranceAngle"):
        for j in range(num-1): ring[i+j+1].EntranceAngle=0
    
    if hasattr(ring[i], "ExitAngle"):
        for j in range(num-1): ring[i+j].ExitAngle=0

        
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
SAVE = ROOT / "outputs" / "ORMs_split"

ring0, ind0 = read.ALBA(ROOT  / "data" / "THERING.mat")
ring1, ind1 = read.ALBAII(ROOT  / "data" / "ring_a2.mat")
ring2 = copy.deepcopy(ring1)
#Applying the divide method
       
#ADDING kickangles to the ALBAII ring
for i in ind1["cor"]["h"]: ring1[i].KickAngle = np.array([0,0])
for i in ind1["cor"]["v"]: ring1[i].KickAngle = np.array([0,0])

split_dict = split_fam(ring1, ind1["dip"], spl, ind1)

ind11 = read.find_ind_ALBAII(ring1)

split_optics = at.get_optics(ring1)
cORM = AnaORM.AnaORM(ring1, "h", ind11)
cORM0 = AnaORM.AnaORM(ring2, "h", ind1)
cORM0.assign_optics()
cORM.assign_optics()
cORM0.dip.average()
ana_disp = cORM0.dip.avDispersion

split_disp = cORM.dip.dispersion
av_dispersion = np.zeros(len(ind1["dip"]))
for i, el in enumerate(av_dispersion):
    av_dispersion[i] = np.average(split_disp[i*spl:(i+1)*spl])


plt.plot(ana_disp , color = "blue")
plt.plot(av_dispersion, color = "red")
plt.show()
