#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 12:43:51 2025

@author: deumenec
"""

import at
import numpy as np
import re #Per les regular expresions


def ALBA(path):
    """Reads the ALBAI lattice from the file and scans for the indices of
    bpms, correctors, quadrupoles and 
    """
    ring = at.load_mat(path)

    ind_bpm  = np.array(at.get_refpts(ring, 'BPM'))  #Indices for the BPM
    ind_cor     = { "v": np.array(at.get_refpts(ring, 'COR')), 
                    "h": np.array(at.get_refpts(ring, 'COR'))}
    ind_quad =  np.array(at.get_refpts(ring, lambda el: el.FamName.startswith('QV') or el.FamName.startswith('QH')))
    ind_dip =  np.array(at.get_refpts(ring, lambda el: el.FamName.startswith('BEND')))
    ind_RF = []
    ind_sex = np.array([])
    ind = {"bpm": ind_bpm, "cor": ind_cor, "quad": ind_quad, "dip": ind_dip, "RF": ind_RF, "sex": ind_sex}
    return ring, ind
    
def ALBAII(path):
    """Reads the ALBAII lattice from the file and scans for the indices of
    bpms, correctors, quadrupoles and 
    """
    ring = at.load_mat(path, use = "ring")

    ordsV = re.compile('^COR$|^SH[1-7][1-4]?$|^SV[246]');
    ordsH = re.compile('^COR$|^SV[1-7][1-4]?$');

    ind_bpm = at.get_refpts(ring, lambda el: el.FamName.startswith('BPM') and not el.FamName.startswith('BPM_')) #BPMs bons sense l'element nou
    #ind_bpm     = np.array([i[0]-1 for i in mat["bpmlist"]])
    ind_cor     = { "v": at.get_refpts(ring, lambda el: ordsV.search(el.FamName)), 
                    "h": at.get_refpts(ring, lambda el: ordsH.search(el.FamName))}
    ind_quad    = np.array(at.get_refpts(ring, lambda el: el.FamName.startswith('LIUQ') 
                                                       or el.FamName.startswith('LIDQ')
                                                       or  el.FamName.startswith('LQ') 
                                                       or el.FamName.startswith('MQ') 
                                                       or el.FamName.startswith('SQ')))
    ind_sex    = np.array(at.get_refpts(ring, lambda el: el.FamName.startswith('SH') 
                                                       or el.FamName.startswith('SV')))
    ind_dip    = at.get_refpts(ring, lambda el: el.FamName.startswith('QD') or el.FamName.startswith('QF'))
    ind_RF     = np.array([1467, 1468, 2209])#1467 és del 3r harmonic!
    ind_all_quad = np.sort(np.concatenate((ind_quad,ind_dip)))
    ind = {"bpm": ind_bpm, "cor": ind_cor, "quad": ind_quad, "dip": ind_dip,"CFD": ind_dip , "RF": ind_RF, "sex": ind_sex, "all_quad":ind_all_quad}
    return ring, ind

def find_ind_ALBAII(ring):
    """From a modified lattice of ALBAII indices for elements are found."""
    ordsV = re.compile('^COR$|^SH[1-7][1-4]?$|^SV[246]');
    ordsH = re.compile('^COR$|^SV[1-7][1-4]?$');

    ind_bpm = at.get_refpts(ring, lambda el: el.FamName.startswith('BPM') and not el.FamName.startswith('BPM_')) #BPMs bons sense l'element nou
    #ind_bpm     = np.array([i[0]-1 for i in mat["bpmlist"]])
    ind_cor     = { "v": at.get_refpts(ring, lambda el: ordsV.search(el.FamName)), 
                    "h": at.get_refpts(ring, lambda el: ordsH.search(el.FamName))}
    ind_quad    = np.array(at.get_refpts(ring, lambda el: el.FamName.startswith('LIUQ') 
                                                       or el.FamName.startswith('LIDQ')
                                                       or  el.FamName.startswith('LQ') 
                                                       or el.FamName.startswith('MQ') 
                                                       or el.FamName.startswith('SQ')))
    ind_sex    = np.array(at.get_refpts(ring, lambda el: el.FamName.startswith('SH') 
                                                       or el.FamName.startswith('SV')))
    ind_dip    = at.get_refpts(ring, lambda el: el.FamName.startswith('QD') or el.FamName.startswith('QF'))
    ind_RF     = np.array([1467, 1468, 2209])#1467 és del 3r harmonic!
    ind_all_quad = np.sort(np.concatenate((ind_quad,ind_dip)))
    ind = {"bpm": ind_bpm, "cor": ind_cor, "quad": ind_quad, "dip": ind_dip,"CFD": ind_dip , "RF": ind_RF, "sex": ind_sex, "all_quad":ind_all_quad}
    return ind
    