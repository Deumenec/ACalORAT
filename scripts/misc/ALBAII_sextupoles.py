#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:02:24 2026

@author: deumenec
Comparacions de la derivada de l'energia numèrica i analítica
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

def dORMdEnergy(ring, ind, step = 0.1):
    """ Returns the derivative of the orbit response matrix with respect to energy"""
    ring2 = copy.deepcopy(ring)
    mcf_val = get_mcf(ring2)
    ring_freq = ring.get_rf_frequency()
    
    ring2.set_cavity(Frequency=ring_freq + step)
    
    RespV = at.latticetools.OrbitResponseMatrix(ring2, "v", ind["bpm"], ind["cor"]["v"])
    RespV.build_tracking() # FIX: build_tracking returns None
    ORMVp = RespV.response

    ring2.set_cavity(Frequency=ring_freq - step)
    
    RespV = at.latticetools.OrbitResponseMatrix(ring2, "v", ind["bpm"], ind["cor"]["v"])
    RespV.build_tracking() # FIX: build_tracking returns None
    ORMVn = RespV.response
    
    return (ORMVp - ORMVn)/2*step*(mcf_val*ring_freq)
    
def quickdORMdEnergy(ring, ind, step=0.1):
    """
    Computes dR_vertical / d_delta. 
    Note: Frequency shift is applied to the ring, but we measure the VERTICAL matrix.
    """
    ring2 = copy.deepcopy(ring)
    mcf_val = get_mcf(ring2)
    ring_freq = ring.get_rf_frequency()
    
    # Calculate the total energy shift changing frequency via the definition:
    # delta = -(1/alpha_c) * (df/f). Total swing is 2 * df.
    delta_total = (2 * step) / (mcf_val * ring_freq)
    
    # +step
    ring2.set_cavity(Frequency=ring_freq + step)
    cORM_p = AnaORM.AnaORM(ring2, "v", ind)
    cORM_p.assign_optics()
    cORM_p.bpm.broadcasters(0, 2)
    cORM_p.cor.broadcasters(1, 2)
    ORMVp = cORM_p.Rab_thick2_(cORM_p.bpm, cORM_p.cor)
    
    # -step
    ring2.set_cavity(Frequency=ring_freq - step)
    cORM_n = AnaORM.AnaORM(ring2, "v", ind)
    cORM_n.assign_optics()
    cORM_n.bpm.broadcasters(0, 2)
    cORM_n.cor.broadcasters(1, 2)
    ORMVn = cORM_n.Rab_thick2_(cORM_n.bpm, cORM_n.cor)
    
    # Return derivative: (Matrix difference) / (Energy change)
    # The minus sign accounts for the inverse relation between f and delta
    return (ORMVp - ORMVn) / delta_total

