# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 10:09:17 2026

@author: dhuerta
Comparing computed jacobians for ALBA and ALBAII 
"""


import os
from pathlib import Path
import numpy as np
import at

from ACalORAT import numerical
from ACalORAT import read
from ACalORAT import AnaORM
from ACalORAT import plot_utils

ROOT = Path(__file__).resolve().parent.parent.parent

SAVE = ROOT / "outputs"  # / "ALBAII_CFD_no_sext"

dORMV_sex =  np.load(SAVE / "ALBAII_CFD" / "Cor_SVD"/"num_dORM_dqV.npy")
dORMV_n_sex =  np.load(SAVE / "ALBAII_CFD_no_sext" / "Cor_SVD"/"num_dORM_dqV.npy")

plot_utils.plot_both_Zeus(dORMV_sex, dORMV_sex, dORMV_n_sex, dORMV_n_sex)