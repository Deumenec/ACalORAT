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
import matplotlib.pyplot as plt

dir_dict = {"h": 0, "v": 1}

#GLOBAL VARIABLE TO ACTIVATE THE VERBOSE MODE
v = True

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
    
    num_dORM_dq = Parallel(n_jobs=-3, verbose=10)(
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
    #Here SUPER IMPORTANT, remember to select only bpms from the original closed orbit!
    dxs = np.array([orbit[i][0] -original_orbit[ind_bpm[i]][0] for i in range(len(orbit))])
    dys = np.array([orbit[i][2] -original_orbit[ind_bpm[i]][2] for i in range(len(orbit))])
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
        dxs = np.array([orbit[i][0] -original_orbit[ind_bpm[i]][0] for i in range(len(orbit))])
        dys = np.array([orbit[i][2] -original_orbit[ind_bpm[i]][2] for i in range(len(orbit))])
        
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

def Cor_SVD_cor(ring , ind, threshold, original_orbit):
    """
    Uses SVD to correct orbit ONLY WITH correctors.
    Afterwards, it changes the RF frequency to cancel "corrector drift".
    And loops until stability or max_steps
    """
    
    max_steps = 10
    
    # Total accumulated changes
    t_kicks = {"h":np.zeros(len(ind["cor"]["h"])), "v": np.zeros(len(ind["cor"]["v"]))}
    t_d_ring_freq = 0.0 # Make sure it's a float
    
    # Kicks for current iteration
    kicks = {"h":np.zeros(len(ind["cor"]["h"])), "v": np.zeros(len(ind["cor"]["v"]))}
    
    # 1. Check Initial State
    orbit = at.find_orbit6(ring, refpts=ind["bpm"])[1]
    dxs = np.array([orbit[i][0] - original_orbit[i][0] for i in range(len(orbit))])
    dys = np.array([orbit[i][2] - original_orbit[i][2] for i in range(len(orbit))])
    
    difference = rms(dxs) + rms(dys)
    if difference < threshold:
        if v: print("No correction was needed")
        return t_kicks, t_d_ring_freq, orbit 
    
    # 2. Iteration Loop
    while max_steps > 0:
        max_steps -= 1
        
        # Calculate Optics and ORM
        optics = at.get_optics(ring, refpts=range(len(ring)))[2]
        
        dispersion = { #Dispersion in bpms
            "h": np.array([optics["dispersion"][i][0] for i in range(len(ring))])[ind["bpm"]],
            "v": np.array([optics["dispersion"][i][2] for i in range(len(ring))])[ind["bpm"]]
        }
        dispersion_cor = { #Dispersion in correctors
            "h": np.array([optics["dispersion"][i][0] for i in range(len(ring))])[ind["cor"]["h"]],
            "v": np.array([optics["dispersion"][i][2] for i in range(len(ring))])[ind["cor"]["v"]]
        }
        
        Resp = at.latticetools.OrbitResponseMatrix(ring, "h", ind["bpm"], ind["cor"]["h"])
        Resp.build_tracking()
        ORMH = Resp.response
        
        Resp = at.latticetools.OrbitResponseMatrix(ring, "v", ind["bpm"], ind["cor"]["v"])
        Resp.build_tracking()
        ORMV = Resp.response
        
        # SVD for Orbit
        kicks["h"], *_ = np.linalg.lstsq(ORMH, dxs, rcond=None)
        kicks["v"], *_ = np.linalg.lstsq(ORMV, dys, rcond=None)
        
        # Apply Negative Feedback
        kicks["h"] = -kicks["h"]
        kicks["v"] = -kicks["v"]
        
        # ACCUMULATE KICKS
        t_kicks["h"] = t_kicks["h"] + kicks["h"]
        t_kicks["v"] = t_kicks["v"] + kicks["v"]
        
        applyCorrections(ring, ind["cor"]["h"], kicks["h"], "h")
        applyCorrections(ring, ind["cor"]["v"], kicks["v"], "v")
        
        # Frequency Correction Logic
        
        sum_kicks = np.sum(kicks["h"]*dispersion_cor["h"])
        
        ring_freq = ring.get_rf_frequency()
        mcf_val = get_mcf(ring) 
        
        quotient, *_ = np.linalg.lstsq(ORMH, dispersion["h"], rcond=None)
        
        # Formula for the change in 
        d_ring_freq = -sum_kicks * mcf_val * ring_freq / (np.sum(dispersion_cor["h"]*quotient))
        
        if v: print(f"Step {10-max_steps} | d_freq: {d_ring_freq} | Sum Kicks: {sum_kicks}")
        
        # Apply Frequency Change
        new_ring_freq = (ring_freq + d_ring_freq)
        ring.set_cavity(Frequency = new_ring_freq)
        
        # ACCUMULATE FREQUENCY
        t_d_ring_freq += d_ring_freq 
        
        # Apply corrector change as well to compensate the frequency change 
        kicks2 = quotient*d_ring_freq/(mcf_val*new_ring_freq)
        applyCorrections(ring, ind["cor"]["h"], kicks2 , "h")
        t_kicks["h"] +=  kicks2
        
        # Check Convergence
        orbit = at.find_orbit6(ring, refpts=ind["bpm"])[1]
        dxs = np.array([orbit[i][0] - original_orbit[i][0] for i in range(len(orbit))])
        dys = np.array([orbit[i][2] - original_orbit[i][2] for i in range(len(orbit))])
        print(t_kicks)
        difference = rms(dxs) + rms(dys)
        if v: print("Diff: ", difference)
        
        if difference < threshold:
            if v: print("Iteration finished, Total Delta Freq:", t_d_ring_freq)
            return t_kicks, t_d_ring_freq, orbit

    if v: print("Iteration did not converge")
    #Retorna igualment les correccions que ha fet a la màquina.
    return t_kicks, t_d_ring_freq, orbit


def Full_SVD_cor(ring, ind_bpm, ind_cor, threshold, original_orbit):
    """
    Uses Extended SVD to correct orbit WITH correctors and RF simultaneously.
    It builds an augmented matrix [ORM_h | Dispersion_h] to solve for 
    both kicks and momentum deviation (delta) in one step.
    """
    if v: print("hii")
    max_steps = 10
    
    # Acumulated changes
    t_kicks = {"h": np.zeros(len(ind_cor["h"])), "v": np.zeros(len(ind_cor["v"]))}
    t_d_ring_freq = 0.0
    
    # Initial Orbit check
    orbit = at.find_orbit6(ring, refpts=ind_bpm)[1]
    # Difference from target orbit
    dxs = np.array([orbit[i][0] - original_orbit[i][0] for i in range(len(orbit))])
    dys = np.array([orbit[i][2] - original_orbit[i][2] for i in range(len(orbit))])
    
    difference = rms(dxs) + rms(dys)
    if difference < threshold:
        if v: print("No correction was needed")
        return t_kicks, t_d_ring_freq, orbit

    # --- Iteration Loop ---
    while max_steps > 0:
        max_steps -= 1
        
        # 1. Update Physics (Optics + ORM)
        optics = at.get_optics(ring, refpts=range(len(ring)))[2]
        
        # Get Dispersion at BPM locations
        disp_h = np.array([optics["dispersion"][i][0] for i in range(len(ring))])[ind_bpm]
        
        # 2. Build Response Matrices
        Resp_H = at.latticetools.OrbitResponseMatrix(ring, "h", ind_bpm, ind_cor["h"])
        Resp_H.build_tracking()
        ORMH = Resp_H.response
        Resp_V = at.latticetools.OrbitResponseMatrix(ring, "v", ind_bpm, ind_cor["v"])
        Resp_V.build_tracking()
        ORMV = Resp_V.response

        # We append Dispersion as the last column of the matrix ORMH
        # Matrix shape becomes (N_BPMs, N_Cors + 1)
        # Reshape dispersion to be a column vector (N, 1)
        col_disp = disp_h.reshape(-1, 1)
        M_aug = np.hstack([ORMH, col_disp])
        
        # Horizontal Solve (Returns kicks AND energy deviation)
        sol_h, *_ = np.linalg.lstsq(M_aug, dxs, rcond=None)
        
        # Vertical Solve (Standard SVD)
        kicks_v, *_ = np.linalg.lstsq(ORMV, dys, rcond=None)

        # 5. Extract Results
        # The last element of sol_h is the required dp/p (delta)
        # The rest are the corrector kicks
        kicks_h = sol_h[:-1]
        delta_val = sol_h[-1]

        # 6. Negative Feedback (Invert the sign to correct)
        kicks_h = -kicks_h
        kicks_v = -kicks_v
        delta_val = -delta_val 

        # 7. Convert Delta to Frequency
        # df = - f_rf * mcf * delta
        ring_freq = ring.get_rf_frequency()
        mcf_val = get_mcf(ring)
        d_freq = - ring_freq * mcf_val * delta_val

        # 8. Apply Corrections
        
        # Accumulate
        t_kicks["h"] += kicks_h
        t_kicks["v"] += kicks_v
        t_d_ring_freq += d_freq
        sum_kicks = np.sum(t_kicks["h"])
        if v: print(f"Step {10-max_steps} | d_freq_total: {t_d_ring_freq} | Sum Kicks: {sum_kicks}")

        # Apply to Ring
        applyCorrections(ring, ind_cor["h"], kicks_h, "h")
        applyCorrections(ring, ind_cor["v"], kicks_v, "v")
        
        new_ring_freq = ring_freq + d_freq
        ring.set_cavity(Frequency=new_ring_freq)

        # 9. Check Convergence
        orbit = at.find_orbit6(ring, refpts=ind_bpm)[1]
        dxs = np.array([orbit[i][0] - original_orbit[i][0] for i in range(len(orbit))])
        dys = np.array([orbit[i][2] - original_orbit[i][2] for i in range(len(orbit))])
        
        difference = rms(dxs) + rms(dys)
        if v: print(f"  Orbit Diff: {difference:.2e}")
        
        if difference < threshold:
            if v: print("Convergence Reached.")
            return t_kicks, t_d_ring_freq, orbit

    # End of Loop
    if v: print("Iteration did not converge within max steps")
    return t_kicks, t_d_ring_freq, orbit

def compute_single_CFD(ring, CFD, ORMH, ORMV, step, ind, closed_orbit, method):
    
    # Deep copy of the ring
    local_ring = copy.deepcopy(ring)
    
    element = local_ring[CFD]
    
    # Calculate B0 (Curvature/Field) from Geometry
    # Ensure Length is not zero to avoid errors (though CFDs have length)
    length = element.Length if element.Length != 0 else 1.0 
    B0 = element.BendingAngle / length
    
    # Calculate Ratio (Dipole / Quadrupole)
    val_quad = element.PolynomB[1]
    
    if val_quad != 0:
        ratio = B0 / val_quad
    else:
        # Fallback if it's purely a dipole
        ratio = 0.0 

    # Apply Perturbations
    local_ring[CFD].PolynomB[1] += step
    local_ring[CFD].PolynomB[0] += step * ratio
    
    if v: print(f"Analysing CFD {CFD} | Step: {step:.1e} | Ratio: {ratio:.4f} | dPolyB0: {step*ratio:.1e}")
    
    # --- FEEDBACK LOOP ---
    t_kicks = {"h": np.zeros(len(ind["cor"]["h"])), "v": np.zeros(len(ind["cor"]["v"]))}
    t_df = 0.0

    if method == "Cor_SVD":
        t_kicks, t_df, _ = Cor_SVD_cor(local_ring , ind, 1e-9, closed_orbit) 
        
    elif method == "Full_SVD":
        t_kicks, t_df, _ = Full_SVD_cor(local_ring , ind, 1e-9, closed_orbit) 
      
    # --- COMPUTE NEW RESPONSE ---
    # Observem que encara que hagi trobat l'estat "corregit" de l'anell, cal tornar
    # a calcular la matriu de resposta perqué aquesta pot haver canviat a l'última iteració
    
    Resp_localH = at.latticetools.OrbitResponseMatrix(
        local_ring, "h", ind["bpm"], ind["cor"]["h"])
    Resp_localV = at.latticetools.OrbitResponseMatrix(
        local_ring, "v", ind["bpm"], ind["cor"]["v"])
    
    Resp_localH.build_tracking()
    Resp_localV.build_tracking()

    new_orbit = at.find_orbit(ring, refpts = range(len(ring)))[1]
    x_sex = np.array([i[0] for i in new_orbit])[ind["sex"]]
    # --- RETURN DICTIONARY ---
    # We divide by 'step' to get the derivative
    return {
        "dH": (Resp_localH.response - ORMH) / step,
        "dV": (Resp_localV.response - ORMV) / step,
        "dFreq": t_df / step,
        "dKicks_h": t_kicks["h"] / step,
        "dKicks_v": t_kicks["v"] / step,
        "dsex"     : x_sex /step
    }

def dORM_dCFD(ring, ind ,step, num=None, multithread=False, method="Cor_SVD"):
    """
    Returns:
        dORM_H, dORM_V, dFreq_vec, dKicksH_mat, dKicksV_mat
        ind: dictionary containing all the indices
    """

    # 1. Initialize Storage Arrays
    # If num is specified, we limit the output size
    n_calcs = len(ind["CFD"]) if num is None else min(len(ind["CFD"]), num)
    target_cfds = ind["CFD"][:n_calcs]

    # Initialize Base Response Matrices to get shapes
    print("Calculating Base ORMs...")
    RespH = at.latticetools.OrbitResponseMatrix(ring, "h", ind["bpm"], ind["cor"]["h"])
    RespH.build_tracking() # FIX: build_tracking returns None, modifies in place
    ORMH = RespH.response 

    RespV = at.latticetools.OrbitResponseMatrix(ring, "v", ind["bpm"], ind["cor"]["v"])
    RespV.build_tracking() # FIX: build_tracking returns None
    ORMV = RespV.response
    closed_orbit = at.find_orbit6(ring, refpts=ind["bpm"])[1] 
    
    # 2. Allocate Result Arrays
    # Shapes: [N_CFD, N_BPM, N_COR]
    num_dORM_dqH = np.zeros((n_calcs, *ORMH.shape))
    num_dORM_dqV = np.zeros((n_calcs, *ORMV.shape))
    
    # Shapes: [N_CFD] (Scalar frequency change per CFD)
    dFreq_dCFD = np.zeros(n_calcs)
    
    # Shapes: [N_CFD, N_COR] (Vector kick change per CFD)
    dKicksH_dCFD = np.zeros((n_calcs, len(ind["cor"]["h"])))
    dKicksV_dCFD = np.zeros((n_calcs, len(ind["cor"]["v"])))
    
    #Orbit at sextupoles for tracking results
    x_sex = np.zeros((n_calcs, len(ind["sex"])))

    
    # 3. Execution (Single vs Multi-thread)
    results = []

    if not multithread:
        for i, CFD in enumerate(target_cfds):
            print(f"Processing {i+1}/{n_calcs} (CFD index {CFD})")
            res = compute_single_CFD(ring, CFD, ORMH, ORMV, step, ind, closed_orbit, method)
            results.append(res)
            
    else:
        results = Parallel(n_jobs=10, verbose=10)(
            delayed(compute_single_CFD)(
                ring, CFD, ORMH, ORMV, step, ind, closed_orbit, method
            ) for CFD in target_cfds
        )
    
    # 4. Unpack Results
    # Whether we ran sequential or parallel, 'results' is now a list of dictionaries
    for i, res in enumerate(results):
        num_dORM_dqH[i] = res["dH"]
        num_dORM_dqV[i] = res["dV"]
        dFreq_dCFD[i]   = res["dFreq"]
        dKicksH_dCFD[i] = res["dKicks_h"]
        dKicksV_dCFD[i] = res["dKicks_v"]
        x_sex[i]        = res["dsex"]
      
    print("Calculation Finished.")
    return num_dORM_dqH, num_dORM_dqV, dFreq_dCFD, dKicksH_dCFD, dKicksV_dCFD, x_sex


    