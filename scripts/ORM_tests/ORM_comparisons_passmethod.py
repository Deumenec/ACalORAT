#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:28:40 2026

@author: deumenec

Trying to change the pass-method to see if that is the source of all errors.
We observe that this is not the solution as it doens't fix anyting.
"""


from pathlib import Path
import numpy as np
import at
import matplotlib.pyplot as plt

from ACalORAT import numerical
from ACalORAT import read
from ACalORAT import AnaORM
from ACalORAT import plot_utils
from ACalORAT import ana_utils
from ACalORAT import math_utils

CALC = True

ROOT = Path(__file__).resolve().parent.parent.parent
SAVE = ROOT / "outputs" /"ORM" /"ORMs_pass_method_test"

ring0, ind0 = read.ALBA(ROOT  / "data" / "THERING.mat")
ring1, ind1 = read.ALBAII(ROOT  / "data" / "ring_a2.mat")

ring0.disable_6d()
ring1.disable_6d()

cutoff = 2
for element in ring0:
    if hasattr(element, "PolynomB") and len(element.PolynomB)>cutoff:
        for i , val in enumerate(element.PolynomB):
            if i>=cutoff: element.PolynomB[i] =0
for element in ring1:
    if hasattr(element, "PolynomB") and len(element.PolynomB)>cutoff:
        for i , val in enumerate(element.PolynomB):
            if i>=cutoff: element.PolynomB[i] =0
            
#ADDING kickangles to the ALBAII ring
for i in ind1["cor"]["h"]: 
    ring1[i] = at.Corrector("CORR", ring1[i].Length, [0,0])

for i in ind1["cor"]["v"]: 
    ring1[i] = at.Corrector("CORR", ring1[i].Length, [0,0])


if CALC:
    ORM_AI_ana  = ana_utils.ORM(ring0, ind0, "v")
    ORM_AII_ana = ana_utils.ORM(ring1, ind1, "v")
    ORM_AI_num  = numerical.ORM(ring0, ind0, "v")
    ORM_AII_num = numerical.ORM(ring1, ind1, "v")
    #Save after calculating
    np.save(SAVE / "ORM_AI_ana", ORM_AI_ana)
    np.save(SAVE / "ORM_AII_ana", ORM_AII_ana)
    np.save(SAVE / "ORM_AI_num", ORM_AI_num)
    np.save(SAVE / "ORM_AII_num", ORM_AII_num)
    
else:
    try:
        ORM_AI_ana = np.load(SAVE /"ORM_AI_ana.npy")
        ORM_AII_ana = np.load(SAVE /"ORM_AII_ana.npy")
        ORM_AI_num = np.load(SAVE /"ORM_AI_num.npy")
        ORM_AII_num = np.load(SAVE /"ORM_AII_num.npy")

        
    except:
        raise ImportError("No hi ha respostes calculades")

error1 = math_utils.normalized_RMSE(ORM_AI_ana, ORM_AI_num,1)
error2 = math_utils.normalized_RMSE(ORM_AII_ana, ORM_AII_num,1)

plot_utils.plot_double(error1, error2, "ALBAI", "ALBAII", yaxis = "% RMSD")

    
