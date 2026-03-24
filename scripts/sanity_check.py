#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:58:01 2026

@author: deumenec
Dispersion derivative sanity check
"""

import os
from pathlib import Path
import numpy as np
import at
import at
import copy

from ACalORAT import read

step = 1e-6

ROOT = Path(__file__).resolve().parent.parent
SAVE = ROOT / "outputs" / "ALBAII_CFD_NOFEEDBACK_ONLY_BEND"

ring, ind = read.ALBAII(ROOT  / "data" / "ring_a2.mat")


orbit0 = at.find_orbit6(ring)[0]
ring[ind["CFD"][0]].BendingAngle += (step)
orbit1 = at.find_orbit6(ring)[0]
aaBend = (orbit0-orbit1)/(step*ring[ind["CFD"][0]].Length)

