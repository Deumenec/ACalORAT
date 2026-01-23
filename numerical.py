#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 22:49:20 2025

@author: deumenec
"""

import at
import copy
import numpy as np
from joblib import Parallel, delayed
import tqdm

dir_dict = {"h": 0, "v": 1}

def get_mcf(ring):
    if (ring.is_6d==True):
        ring.disable_6d()
        mcf = ring.mcf
        ring.enable_6d()
        return mcf
    return ring.mcf

def ORM(ring, direction, ind_bpm, ind_cor):
    """ ORM numerically to check if the corrections we want to apply are worthwile!
    """
    Resp = at.latticetools.OrbitResponseMatrix(ring,direction, ind_bpm, ind_cor) #class for computing the ORM
    Resp.build_tracking()
    return Resp.response

#Functions to calculate the dORMdq numerically.
def compute_single_quad(ring, quad, ORM, direction, step, ind_bpm, ind_cor):
    #Make a deep copy of the ring so threads don't interfere
    local_ring = copy.deepcopy(ring)

    #Change the quadrupole strength
    local_ring[quad].PolynomB[1] += step

    #Compute the new ORM
    Resp_local = at.latticetools.OrbitResponseMatrix(
        local_ring, direction, ind_bpm, ind_cor)
    
    Resp_local.build_tracking()

    return (Resp_local.response - ORM) / step

def dORM_dq(ring, ind_bpm, ind_cor, ind_quad, step, direction):
    """
    Parameters
    ----------
    ring : at.lattice
        Ring for which the matrix is calculated
    ind_bpm : array
        indices of the BPMs for the ORM matrix
    ind_cor : array
        indices of the correctors for the ORM matrix
    ind_quad : array
        indices 
    dimension : char
        "v" for the vertical dimension and "h"
    Returns
    -------
    num_dORM_dq: np.array 
        The dORM_dq rank 3 tensor with indices dORM_dq[quadrupole][bpm][corrector]
    """
    num_dORM_dq = np.zeros([len(ind_quad), len(ind_bpm), len(ind_cor)])
    Resp = at.latticetools.OrbitResponseMatrix(ring,direction, ind_bpm, ind_cor) #class for computing the ORM
    Resp.build_tracking()
    ORM = Resp.response
    
    num_dORM_dq = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_single_quad) (ring, quad, ORM, direction, step, ind_bpm, ind_cor) for quad in ind_quad)
    num_dORM_dq = np.array(num_dORM_dq)
    return num_dORM_dq


def applyCorrections(ring, ind_cor, kicks, direction):
    """Given a vector of changes in correctors, they are applyied to a ring 
    """
    dir_dict = {"h": 0, "v": 1}
    dir_ind = dir_dict[direction]
    for i, cor in enumerate(ind_cor):
        ((ring[cor]).KickAngle)[dir_ind] += kicks[i] 
    return

def rms(vec):
    """Calculates the RMS of a given vector."""
    return np.sqrt(np.sum(vec*vec))
    
def kick_cor(ring , ind_bpm, ind_cor, threshold, original_orbit):
    """
    Tweek corrector kickangles to minimize errors at bpms with l2 norm matching
    the ring orbit with the original one, it does not apply any energy correction
    """
    max_steps = 4
    
    #Total kicks performed
    t_kicks = {"h":np.zeros(len(ind_cor["h"])), "v": np.zeros(len(ind_cor["v"]))}
    
    #Kicks for current iteration
    kicks = {"h":np.zeros(len(ind_cor["h"])), "v": np.zeros(len(ind_cor["v"]))}
    
    #Finds orbit at the beginig
    orbit = at.find_orbit6(ring, refpts=ind_bpm)[1]
    dxs = np.array([orbit[i][0] -original_orbit[i][0] for i in range(len(orbit))])
    dys = np.array([orbit[i][2] -original_orbit[i][2] for i in range(len(orbit))])
    difference = rms(dxs)+rms(dys)
    if difference<threshold:
        #print("No correction was needed")
        return t_kicks, orbit 
    
    while max_steps >1:
        
        max_steps -=1
        Resp = at.latticetools.OrbitResponseMatrix(ring,"h", ind_bpm, ind_cor["h"])
        Resp.build_tracking()
        ORMH = Resp.response
        Resp = at.latticetools.OrbitResponseMatrix(ring,"v", ind_bpm, ind_cor["v"])
        Resp.build_tracking()
        ORMV = Resp.response
        
        #Numerically inverse the ORM
        kicks["h"], *_ =np.linalg.lstsq(ORMH, dxs)
        kicks["v"], *_ =np.linalg.lstsq(ORMV, dys)
        
        #Aplico com el kick contrari al que està causant el desplaçament observat per corregir-lo
        kicks["h"] = -kicks["h"]
        kicks["v"] = -kicks["v"]
        
        t_kicks["h"] = t_kicks["h"]+kicks["h"]
        t_kicks["v"] = t_kicks["v"]+kicks["v"]
        
        applyCorrections(ring, ind_cor["h"], kicks["h"], "h")
        applyCorrections(ring, ind_cor["v"], kicks["v"], "v")
        
        orbit = at.find_orbit6(ring, refpts=ind_bpm)[1]
        dxs = np.array([orbit[i][0] -original_orbit[i][0] for i in range(len(orbit))])
        dys = np.array([orbit[i][2] -original_orbit[i][2] for i in range(len(orbit))])
        
        difference = rms(dxs)+rms(dys)
        #print(difference)
        if difference<threshold:
            #Restore the ring after changing the kickangles
            applyCorrections(ring, ind_cor["h"],  -t_kicks["h"], "h")
            applyCorrections(ring, ind_cor["v"], -t_kicks["v"], "v")
            return t_kicks, orbit
        
    #print("Iteration did not converge")
    applyCorrections(ring, ind_cor["h"],  -t_kicks["h"], "h")
    applyCorrections(ring, ind_cor["v"], -t_kicks["v"], "v")
    return t_kicks, orbit

def kick_RF_cor(ring , ind_bpm, ind_cor, threshold, original_orbit, dispersion):
    """
    Uses SVD to correct in bpms using kickers. Afterwards, it changes the RF frequency
    To make the sum of the kicks applyed be zero.
    At least two iterations are needed for the method to work
    """
    #Hardcoded 
    max_steps = 4
    
    #Total kicks performed
    t_kicks = {"h":np.zeros(len(ind_cor["h"])), "v": np.zeros(len(ind_cor["v"]))}
    
    #Kicks for current iteration
    kicks = {"h":np.zeros(len(ind_cor["h"])), "v": np.zeros(len(ind_cor["v"]))}
    
    #Finds orbit at the beginig
    orbit = at.find_orbit6(ring, refpts=ind_bpm)[1]
    dxs = np.array([orbit[i][0] -original_orbit[i][0] for i in range(len(orbit))])
    dys = np.array([orbit[i][2] -original_orbit[i][2] for i in range(len(orbit))])
    difference = rms(dxs)+rms(dys)
    if difference<threshold:
        print("No correction was needed")
        return t_kicks, orbit #The original one as expected
    
    while max_steps >1:
        max_steps -=1
        Resp = at.latticetools.OrbitResponseMatrix(ring,"h", ind_bpm, ind_cor["h"])
        Resp.build_tracking()
        ORMH = Resp.response + [dispersion["h"][ind_bpm]]
        Resp = at.latticetools.OrbitResponseMatrix(ring,"v", ind_bpm, ind_cor["v"])
        Resp.build_tracking()
        ORMV = Resp.response + [dispersion["h"][ind_bpm]]
        #Numerically inverse the ORM
        kicks["h"], *_ =np.linalg.lstsq(ORMH, dxs)
        kicks["v"], *_ =np.linalg.lstsq(ORMV, dys)
        #Aplico com el kick contrari al que està causant el desplaçament observat per corregir-lo
        kicks["h"] = -kicks["h"]
        kicks["v"] = -kicks["v"]
        
        t_kicks["h"] = t_kicks["h"]+kicks["h"]
        t_kicks["v"] = t_kicks["v"]+kicks["v"]
        
        applyCorrections(ring, ind_cor["h"], kicks["h"], "h")
        applyCorrections(ring, ind_cor["v"], kicks["v"], "v")
        
        #After applying the correction for the correctors, it does the RF correction
        #With the simple formula I deduced:
        
        sum_kicks= np.sum(kicks["h"])
        ring_freq = ring.get_rf_frequency()
        mcf       = get_mcf(ring)
        quotient, *_ =np.linalg.lstsq(ORMH, dispersion["h"])
        
        d_ring_freq = -sum_kicks*mcf*ring_freq/np.sum(quotient)
        
        ring.set_rf_frequency(ring, ring_freq+d_ring_freq)
        
        orbit = at.find_orbit6(ring, refpts=ind_bpm)[1]
        dxs = np.array([orbit[i][0] -original_orbit[i][0] for i in range(len(orbit))])
        dys = np.array([orbit[i][2] -original_orbit[i][2] for i in range(len(orbit))])
        
        difference = rms(dxs)+rms(dys)
        #print(difference)
        if difference<threshold:
            #Restore the ring after changing the kickangles
            applyCorrections(ring, ind_cor["h"],  -t_kicks["h"], "h")
            applyCorrections(ring, ind_cor["v"], -t_kicks["v"], "v")
            return t_kicks, orbit
        
    print("Iteration did not converge")
    applyCorrections(ring, ind_cor["h"],  -t_kicks["h"], "h")
    applyCorrections(ring, ind_cor["v"], -t_kicks["v"], "v")
    return t_kicks, orbit


def compute_single_CFD(ring, CFD, ORM, direction, step, ind_bpm, ind_cor, closed_orbit, dispersion):
    #Make a deep copy of the ring
    local_ring = copy.deepcopy(ring)
    #Change the CFD strength ratio(quadrupole and KickAngle)
    B0 = local_ring[CFD].BendingAngle/ local_ring[CFD].Length
    ratio = B0/local_ring[CFD].PolynomB[1]
    
    #Activating the CFD
    local_ring[CFD].PolynomB[1] += step
    local_ring[CFD].PolynomB[0] += step*ratio
    
    #Make the coresponding live-feedback corrections to kickers and RF-freq
    diff = 1
    while diff >= 1e-9:
        kick_RF_cor(ring , ind_bpm, ind_cor, 1e-9, closed_orbit, dispersion)
    
    #Compute the new ORM and this partial deriative
    Resp_local = at.latticetools.OrbitResponseMatrix(
        local_ring, direction, ind_bpm, ind_cor)
    
    Resp_local.build_tracking()
    
    return (Resp_local.response - ORM) / step

def dORM_dCFD(ring, ind_bpm, ind_cor, ind_CFD, ind_RF, step, direction):
    """
    Parameters
    ----------
    ring : at.lattice
        Ring for which the matrix is calculated
    ind_bpm : array
        indices of the BPMs for the ORM matrix
    ind_cor : array
        indices of the correctors for the ORM matrix
    ind_quad : array
        indices 
    dimension : char
        "v" for the vertical dimension and "h"
    Returns
    -------
    num_dORM_dq: np.array 
        The dORM_dq rank 3 tensor with indices dORM_dq[quadrupole][bpm][corrector]
    """

    num_dORM_dq = np.zeros([len(ind_CFD), len(ind_bpm), len(ind_cor)])
    closed_orbit = at.find_orbit(ring, refpts=range(len(ring)))[1] #Compute the closed orbit along all elements in the ring

    optics = at.get_optics(ring, refpts=range(len(ring)))[2]
    dispersion = {"h": np.array( [ optics["dispersion"][i][0] for i in range(len(ring)) ] ),
                  "v": np.array( [ optics["dispersion"][i][2] for i in range(len(ring)) ] )}
    
    Resp = at.latticetools.OrbitResponseMatrix(ring,direction, ind_bpm, ind_cor[direction]) #class for computing the ORM in the original direction
    print("hii")
    ORM = Resp.build_tracking()
    for i, CFD in enumerate(ind_CFD[:5]):
        print(i)
        num_dORM_dq[i] = compute_single_CFD(ring, CFD, ORM, direction, step, ind_bpm, ind_cor, closed_orbit, dispersion)
    """
    num_dORM_dq = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_single_CFD) (ring, CFD, ORM, direction, step, ind_bpm, ind_cor, closed_orbit) for CFD in ind_CFD)
    num_dORM_dq = np.array(num_dORM_dq)
    """
    return num_dORM_dq



    