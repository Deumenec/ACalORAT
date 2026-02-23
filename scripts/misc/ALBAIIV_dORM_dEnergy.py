# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 10:09:17 2026

@author: dhuerta
Comparing computed jacobians for ALBA and ALBAII 
"""


from pathlib import Path
import numpy as np
import at
import matplotlib.pyplot as plt

from ACalORAT import numerical
from ACalORAT import read
from ACalORAT import AnaORM
from ACalORAT import plot_utils

CALC = False

ROOT = Path(__file__).resolve().parent.parent.parent
SAVE = ROOT / "outputs" / "ALBAIIdEnergy"

ring, ind = read.ALBAII(ROOT  / "data" / "ring_a2.mat")


for i in ind["cor"]["h"]: ring[i].KickAngle = np.array([0,0])
for i in ind["cor"]["v"]: ring[i].KickAngle = np.array([0,0])

if CALC:
    dORM_exact = numerical.dORMdEnergy(ring, ind, step = 0.1)
    dORM_num0  = numerical.quickdORMdEnergy(ring, ind, step = 0.1)
        np.save(SAVE / "dORM_exact", dORM_exact)
    np.save(SAVE / "dORM_num0", dORM_num0)
else:
    try:
        dORM_exact = np.load(SAVE /"dORM_exact.npy")
        dORM_num0  = np.load(SAVE /"dORM_num0.npy")
        #dORM_num0  = np.load(SAVE /"dORM_num0.npy")
    except:
        raise ImportError("No hi ha respostes calculades")
        
plot_utils.compare(dORM_exact, dORM_num0)

cORM = AnaORM.AnaORM(ring, "v", ind)
cORM.assign_optics()
#np.save(SAVE / "dORM_num0", dORM_exact)    

dEnergy = cORM.all_optics[2]["closed_orbit"][:, 4]
plt.plot(dEnergy)
plt.show()