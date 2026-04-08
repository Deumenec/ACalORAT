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

from ACalORAT import numerical
from ACalORAT import read
from ACalORAT import AnaORM
from ACalORAT import plot_utils
from ACalORAT import ana_utils
from ACalORAT import math_utils

CALC = True

ROOT = Path(__file__).resolve().parent.parent.parent
SAVE = ROOT / "outputs" /"ORM" /"ORMs"

ring0, ind0 = read.ALBA(ROOT  / "data" / "THERING.mat")
ring1, ind1 = read.ALBAII(ROOT  / "data" / "ring_a2.mat")

#ADDING kickangles to the ALBAII ring
for i in ind1["cor"]["h"]: ring1[i].KickAngle = np.array([0,0])
for i in ind1["cor"]["v"]: ring1[i].KickAngle = np.array([0,0])


d = "v" # v or h
if CALC:
    ORM_AI_ana  = ana_utils.ORM(ring0, ind0, d)
    ORM_AII_ana = ana_utils.ORM(ring1, ind1, d)
    ORM_AI_num  = numerical.ORM(ring0, ind0, d)
    ORM_AII_num = numerical.ORM(ring1, ind1, d)
    #Save after calculating
    np.save(SAVE / (d+ "ORM_AI_ana"), ORM_AI_ana)
    np.save(SAVE / (d+ "ORM_AII_ana"), ORM_AII_ana)
    np.save(SAVE / (d+ "ORM_AI_num"), ORM_AI_num)
    np.save(SAVE / (d+ "ORM_AII_num"), ORM_AII_num)
    
else:
    try:
        ORM_AI_ana = np.load(SAVE  /d+"ORM_AI_ana.npy")
        ORM_AII_ana = np.load(SAVE /d+"ORM_AII_ana.npy")
        ORM_AI_num = np.load(SAVE  /d+"ORM_AI_num.npy")
        ORM_AII_num = np.load(SAVE /d+"ORM_AII_num.npy")

        
    except:
        raise ImportError("No hi ha respostes calculades")

error1 = math_utils.normalized_RMSE(ORM_AI_ana, ORM_AI_num,1)
error2 = math_utils.normalized_RMSE(ORM_AII_ana, ORM_AII_num,1)

plot_utils.plot_double(error1, error2, "ALBAI", "ALBAII", yaxis = "% RMSD")

    
