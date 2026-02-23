#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 12:43:48 2026

@author: deumenec

General methods used to compute simple cases
"""

from . import AnaORM

def ORM(ring, ind, direction):
    """
    Calculates the Orbit Response Matrix Using analytical formulas from AnaORM
    """
    cORM = AnaORM.AnaORM(ring, direction, ind)
    cORM.assign_optics()
    cORM.bpm.broadcasters(0, 2)
    cORM.cor.broadcasters(1, 2)
    if direction == "v":
        return cORM.Rab_thick2_(cORM.bpm, cORM.cor)
    elif direction == "h":
        return cORM.Rab_thick2_(cORM.bpm, cORM.cor) + cORM.Rab_thick2_disp(cORM.bpm, cORM.cor)