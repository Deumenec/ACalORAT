#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 22:49:20 2025

@author: dhuerta
"""

import at
import copy
import numpy as np
from joblib import Parallel, delayed

from . import AnaORM
from . import ana_utils

dir_dict = {"h": 0, "v": 1}

#GLOBAL VARIABLE TO ACTIVATE THE VERBOSE MODE
v = True
d = False

def get_mcf(ring):
    if (ring.is_6d==True):
        ring.disable_6d()
        mcf = ring.mcf
        ring.enable_6d()
        return mcf
    return ring.mcf

def ORM(ring, ind, direction = "v"):
    Resp = at.latticetools.OrbitResponseMatrix(ring, direction, ind["bpm"], ind["cor"][direction], steerdelta= 1e-6)
    Resp.build_tracking(tol=1e-12, max_iterations=150) 
    return Resp.response


def compute_single_quad(ring, quad, ORM, direction, step, ind_bpm, ind_cor):
    """
    Computing the derivative of the ORM with respect to changing a quadrupole only
    """
    #Make a deep copy of the ring so threads don't interfere
    local_ring = copy.deepcopy(ring)

    #Change the quadrupole strength
    local_ring[quad].PolynomB[1] += step

    #Compute the new ORM
    Resp_local = at.latticetools.OrbitResponseMatrix(
        local_ring, direction, ind_bpm, ind_cor, steerdelta = 5e-7)
    
    resp1 = Resp_local.build_tracking(tol=1e-13, max_iterations=150)
    
    local_ring[quad].PolynomB[1] += -2*step

    #Compute the new ORM
    Resp_local = at.latticetools.OrbitResponseMatrix(
        local_ring, direction, ind_bpm, ind_cor, steerdelta = 5e-7)
    
    resp2 = Resp_local.build_tracking(tol=1e-13, max_iterations=150)


    return (resp1 - resp2) / (2*step)

def compute_single_CFD_pure(ring, quad, ORM, direction, step, ind_bpm, ind_cor):
    """
    Computing the derivative of the Orbit response matrix with respect to changing
    only the quadrupole and the dipole in a CFD acordingly.
    
    Using symetric finite differences
    """
    
    #Make a deep copy of the ring so threads don't interfere
    local_ring = copy.deepcopy(ring)
    elem = local_ring[quad]
    
    #Change the quadrupole strength AND DIPOLE STRENGTH
    base_K = elem.PolynomB[1] 
    
    if base_K != 0 and hasattr(elem, "BendingAngle"):
        ratio = elem.BendingAngle / (elem.Length * base_K)
    else:
        ratio = 0
    elem.PolynomB[1] += step
    elem.PolynomB[0] += step * ratio

    #Compute the new ORM
    Resp_local = at.latticetools.OrbitResponseMatrix(
        local_ring, direction, ind_bpm, ind_cor, steerdelta = 5e-7)
    
    resp1 = Resp_local.build_tracking(tol=1e-13, max_iterations=150)
    
    elem.PolynomB[1] += -2*step
    elem.PolynomB[0] += -2*step * ratio

    #Compute the new ORM
    Resp_local = at.latticetools.OrbitResponseMatrix(
        local_ring, direction, ind_bpm, ind_cor, steerdelta = 5e-7)
    
    resp2 = Resp_local.build_tracking(tol=1e-13, max_iterations=150)


    return (resp1 - resp2) / (2*step)

def compute_single_dip_pure(ring, quad, direction, step, ind_bpm, ind_cor):
    """
    Computing the derivative of the Orbit response matrix with respect to changing
    only the dipole component of one element.
    
    Using symetric finite differences
    """
    #Make a deep copy of the ring so threads don't interfere
    local_ring = copy.deepcopy(ring)
    elem = local_ring[quad]
    
    elem.PolynomB[0] += step

    #Compute the new ORM
    Resp_local = at.latticetools.OrbitResponseMatrix(
        local_ring, direction, ind_bpm, ind_cor, steerdelta = 5e-7)
    
    resp1 = Resp_local.build_tracking(tol=1e-13, max_iterations=150)
    optics1 = at.get_optics(local_ring, refpts=range(len(local_ring)))
    energy1 = optics1[2]["closed_orbit"][:, 4]
    disp1   = optics1[1]["dispersion"][ind_bpm, 0]

    #Steping down the local ring for the second derivative
    elem.PolynomB[0] += -2*step

    #Compute the new ORM
    Resp_local = at.latticetools.OrbitResponseMatrix(
        local_ring, direction, ind_bpm, ind_cor, steerdelta = 5e-7)

    resp2 = Resp_local.build_tracking(tol=1e-13, max_iterations=150)
    optics2 = at.get_optics(local_ring, refpts=range(len(local_ring)))
    energy2 = optics2[2]["closed_orbit"][:, 4]
    disp2   = optics2[1]["dispersion"][ind_bpm, 0]

    result = {"num_dORM_dq"  : (resp1 - resp2) / (2*step),
              "energy"       : np.average((energy1 - energy2))/(2*step),
              "dispersion"   : (disp1 - disp2) / (2*step)}

    return result

def single_snum_quad(ring, quad, ORM, direction, step, ind):
    """
    Calculates the ORM semi-numerially by just calculating the ORM at different
    configurations using the analytical formulas.
    """
    
    #Make a deep copy of the ring so threads don't interfere
    local_ring = copy.deepcopy(ring)

    #Change the quadrupole strength
    local_ring[quad].PolynomB[1] += step

    #Compute the new ORM
    resp1 = ana_utils.ORM(local_ring, ind, direction)
    
    local_ring[quad].PolynomB[1] += -2*step

    #Compute the new ORM
    resp2 = ana_utils.ORM(local_ring, ind, direction)


    return (resp1 - resp2) / (2*step)
    
def compute_average_dispersion(ring, ind, all_disp):
    """ Calculates the average dispersion in a set of ring elements.
    In the horizontal direction
    Handles thick elements, thin elements, and zero-focusing correctors/drifts safely.
    """
    avDispersion = np.zeros(len(ind))
    dispersion = all_disp[ind, 0]
    dispersionp = all_disp[ind, 1]
    
    for i, idx in enumerate(ind):
        el = ring[idx]
        length = getattr(el, "Length", 0.0)
        bend = getattr(el, "BendingAngle", 0.0)
        k = getattr(el, "K", 0.0)
        
        # If it's a thin element (length = 0), average dispersion is just point dispersion
        if length == 0:
            avDispersion[i] = dispersion[i]
            continue
        
        k_total = k + (bend/length)**2
        
        # Prevent 0/0 division (NaNs) for standard correctors/drifts
        if abs(k_total) < 1e-12:
            # Analytical limit of integration when k -> 0
            avDispersion[i] = dispersion[i] + dispersionp[i] * length / 2.0 + bend * length / 6.0
        else:
            k_c = complex(k_total)
            phi = np.sqrt(k_c) * length
            term1 = dispersion[i] * np.sin(phi) / phi
            term2 = dispersionp[i] / (k_c * length) * (1 - np.cos(phi))
            term3 = bend / (length * k_c) * (1 - np.sin(phi) / phi)
            avDispersion[i] = np.real(term1 + term2 + term3)
            
    return avDispersion
    
    
    
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
    Resp = at.latticetools.OrbitResponseMatrix(ring,direction, ind_bpm, ind_cor, steerdelta = 1e-6) #class for computing the ORM
    Resp.build_tracking(tol=1e-12, max_iterations=100)
    ORM = Resp.response
    
    num_dORM_dq = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_single_quad) (ring, quad, ORM, direction, step, ind_bpm, ind_cor) for quad in ind_quad)
    num_dORM_dq = np.array(num_dORM_dq)
    return num_dORM_dq

def dORM_dq_pure(ring, ind_bpm, ind_cor, ind_quad, step, direction):
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
        With respect to changing both the quadrupolar and the dipolar component acordingly
        and with no active feedback correction.
    """
    num_dORM_dq = np.zeros([len(ind_quad), len(ind_bpm), len(ind_cor)])
    Resp = at.latticetools.OrbitResponseMatrix(ring,direction, ind_bpm, ind_cor, steerdelta = 1e-6) #class for computing the ORM
    Resp.build_tracking(tol=1e-12, max_iterations=100)
    ORM = Resp.response
    
    num_dORM_dq = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_single_quad) (ring, quad, ORM, direction, step, ind_bpm, ind_cor) for quad in ind_quad)
    num_dORM_dq = np.array(num_dORM_dq)
    return num_dORM_dq

def dORM_dbend(ring, ind_bpm, ind_cor, ind_quad, step, direction):
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
    dict:
        num_dORM_dq: np.array 
            The dORM_dq rank 3 tensor with indices dORM_dq[quadrupole][bpm][corrector]
            With respect to changing ONLY the dipole component of given magnets
        energy: np.array
    
    """
    num_dORM_dq = np.zeros([len(ind_quad), len(ind_bpm), len(ind_cor)])
    energy      = np.zeros([len(ind_quad)])
    dispersion  = np.zeros([len(ind_quad), len(ind_bpm)])
    results = Parallel(n_jobs=-3, verbose=10)(
        delayed(compute_single_dip_pure) (ring, quad, direction, step, ind_bpm, ind_cor) for quad in ind_quad)
    for i, res in enumerate(results):
        energy[i]      = res["energy"]
        num_dORM_dq[i] = res["num_dORM_dq"]
        dispersion[i]  = res["dispersion"]

    return num_dORM_dq, energy, dispersion

def dORM_dq_semi(ring, ind, step, direction):
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
    print("hii")
    num_dORM_dq = np.zeros([len(ind["quad"]), len(ind["bpm"]), len(ind["cor"][direction])])
    Resp = at.latticetools.OrbitResponseMatrix(ring,direction, ind["bpm"], ind["cor"][direction], steerdelta = 1e-5) #class for computing the ORM
    Resp.build_tracking()
    ORM = Resp.response
    
    num_dORM_dq = Parallel(n_jobs=-1, verbose=10)(
        delayed(single_snum_quad) (ring, quad, ORM, direction, step, ind) for quad in ind["quad"])
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
    return np.sqrt(np.mean(vec*vec))

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

def Cor_SVD_cor(ring, ind, threshold, original_orbit, ORMH, ORMV, recalc_step=12):
    """
    Corrects orbit using Average Dispersion for frequency feedback.
    """
    max_steps = 30
    local_ORMH = ORMH
    local_ORMV = ORMV
    
    t_kicks = {"h": np.zeros(len(ind["cor"]["h"])), "v": np.zeros(len(ind["cor"]["v"]))}
    t_d_ring_freq = 0.0
    
    # Initial Check
    optics_data = at.get_optics(ring, refpts=range(len(ring)))
    orbit = optics_data[2]["closed_orbit"]
    
    # Extract Arrays for Average Calc
    # optics_data[2] is a record array. We need columns 'dispersion' (idx 0) and 'dispersion' derivative?
    # AT usually returns [D, D', ...] in 'dispersion'. 
    # Let's assume standard AT structure: optics_data[2]['dispersion'] is (N, 4) -> [Dx, D'x, Dy, D'y]
    all_disp = optics_data[2]['dispersion']
    
    # Initial difference check...
    bpm_orbit = orbit[ind["bpm"]]
    dxs = np.array([bpm_orbit[i][0] - original_orbit[ind["bpm"][i]][0] for i in range(len(bpm_orbit))])
    dys = np.array([bpm_orbit[i][2] - original_orbit[ind["bpm"][i]][2] for i in range(len(bpm_orbit))])
    
    
    if (rms(dxs) + rms(dys)) < threshold:
        print("No correction needed")
        return t_kicks, t_d_ring_freq, orbit

    for step in range(1, max_steps + 1):
        
        # --- A. RECALCULATE ORM IF NEEDED ---
        if step == recalc_step:
            print(f"Step {step}: Recalculating ORM...")
            # ... (Same as before) ...
            Resp_H = at.latticetools.OrbitResponseMatrix(ring, "h", ind["bpm"], ind["cor"]["h"])
            Resp_V = at.latticetools.OrbitResponseMatrix(ring, "v", ind["bpm"], ind["cor"]["v"])
            Resp_H.build_tracking()
            Resp_V.build_tracking()
            local_ORMH = Resp_H.response
            local_ORMV = Resp_V.response

        # --- B. COMPUTE AVERAGE DISPERSION ---
        # We need D and D' for all elements to compute the average
        # Extract full arrays first
        Dx_all  = all_disp[:, 0]
        
        # Compute Average Dispersion for Horizontal Correctors
        avg_disp_h_cor = compute_average_dispersion(
            ring, ind["cor"]["h"], all_disp
        )
        
        disp_h_bpm = Dx_all[ind["bpm"]]

        # --- C. SVD CORRECTION ---
        kicks_h, *_ = np.linalg.lstsq(local_ORMH, dxs, rcond=None)
        kicks_v, *_ = np.linalg.lstsq(local_ORMV, dys, rcond=None)
        
        kicks_h = -kicks_h
        kicks_v = -kicks_v
        
        t_kicks["h"] += kicks_h
        t_kicks["v"] += kicks_v
        
        applyCorrections(ring, ind["cor"]["h"], kicks_h, "h")
        applyCorrections(ring, ind["cor"]["v"], kicks_v, "v")

        # --- D. FREQUENCY CORRECTION  ---
        
        # 1. Path Lengthening = Sum( Kick * Average_Dispersion )
        # This is physically rigorous: integral(D(s) * x''(s) ds)
        sum_kicks = np.sum(kicks_h * avg_disp_h_cor)
        
        # 2. Dispersion Drive Pattern
        quotient, *_ = np.linalg.lstsq(local_ORMH, disp_h_bpm, rcond=1e-8)
        
        ring_freq = ring.get_rf_frequency()
        mcf_val = get_mcf(ring)
        
        # 3. Denominator: Sum( Average_Dispersion * Corrector_Strength_for_Dispersion_Pattern )
        # We use average dispersion here too for consistency
        denom = np.sum(avg_disp_h_cor * quotient)
        
        if abs(denom) < 1e-15:
            d_ring_freq = 0.0
        else:
            d_ring_freq = -sum_kicks * mcf_val * ring_freq / denom
            
        new_ring_freq = ring_freq + d_ring_freq
        ring.set_cavity(Frequency=new_ring_freq)
        #Store total change
        t_d_ring_freq += d_ring_freq

        # Compensate
        kicks2 = quotient * d_ring_freq / (mcf_val * new_ring_freq)
        applyCorrections(ring, ind["cor"]["h"], kicks2, "h")
        t_kicks["h"] += kicks2
        
        # --- E. UPDATE OPTICS & CHECK ---
        ##########################
        # Testing without changing the optics!
        ##########################
        #optics_data = at.get_optics(ring, refpts=range(len(ring)))
        orbit = at.find_orbit6(ring, refpts=range(len(ring)))[1]
        #all_disp = optics_data[2]['dispersion'] # Update for next loop
        
        bpm_orbit = orbit[ind["bpm"]]
        dxs = np.array([bpm_orbit[i][0] - original_orbit[ind["bpm"][i]][0] for i in range(len(bpm_orbit))])
        dys = np.array([bpm_orbit[i][2] - original_orbit[ind["bpm"][i]][2] for i in range(len(bpm_orbit))])
        
        diff = rms(dxs) + rms(dys)
        if v: print(f"Step {step} | dFreq: {t_d_ring_freq} | Sum Kicks: {sum_kicks} | diff: {diff} ")

        
        if diff < threshold:
            
            return t_kicks, t_d_ring_freq, orbit
    print("iteration did not converge")
    return t_kicks, t_d_ring_freq, orbit


def compute_single_CFD(ring, CFD, ORMH, ORMV, step, ind, closed_orbit, method):
    """
    Computes derivatives using Central Difference (f(x+h) - f(x-h)) / 2h.
    """
    print("HOLAAAAAA")
    # GEOMETRY
    base_element = ring[CFD]
    length = base_element.Length if getattr(base_element, 'Length', 0) != 0 else 1.0
    
    # Bending
    angle = getattr(base_element, 'BendingAngle', 0.0)
    B0 = angle / length
    
    # Calculate Ratio (Dipole / Quadrupole) to maintain CFD geometry
    val_quad = base_element.PolynomB[1]
    
    if val_quad != 0:
        ratio = B0 / val_quad
    else:
        ratio = 0.0 

    # --- 2. INTERNAL HELPER FOR SIMULATION ---
    def _get_state(perturbation_sign):
        """
        Runs the simulation for a specific perturbation direction (+1 or -1).
        """
        local_ring = copy.deepcopy(ring)
        
        # Apply Perturbation (+step or -step)
        delta = step * perturbation_sign
        local_ring[CFD].PolynomB[1] += delta
        local_ring[CFD].PolynomB[0] += delta * ratio
        
        # --- FEEDBACK LOOP (CORRECTION) ---
        t_kicks = {"h": np.zeros(len(ind["cor"]["h"])), 
                   "v": np.zeros(len(ind["cor"]["v"]))}
        t_df = 0.0

        if method == "Cor_SVD":
            t_kicks, t_df, _   = Cor_SVD_cor(local_ring, ind, 1e-12, closed_orbit, ORMH, ORMV) 
        elif method == "Full_SVD":
            t_kicks, t_df, _   = Full_SVD_cor(local_ring, ind, 1e-12, closed_orbit, ORMH, ORMV) 
        
        # --- COMPUTE RESPONSE MATRIX ---
        Resp_localH = at.latticetools.OrbitResponseMatrix(
            local_ring, "h", ind["bpm"], ind["cor"]["h"])
        Resp_localV = at.latticetools.OrbitResponseMatrix(
            local_ring, "v", ind["bpm"], ind["cor"]["v"])
        
        Resp_localH.build_tracking()
        Resp_localV.build_tracking()

        # --- COMPUTE ORBIT FOR SEXTUPOLES ---

        new_orbit = at.find_orbit(local_ring, refpts=range(len(local_ring)))[1]
        x_sex = np.array([i[0] for i in new_orbit])[ind["sex"]]
        x_sexp= np.array([i[1] for i in new_orbit])[ind["sex"]]
        energy = np.average(new_orbit[:,4])
        
        new_optics = at.get_optics(ring, refpts=range(len(ring)))[2]
        
        bpm_disp = np.array([i[0] for i in new_optics["dispersion"]])[ind["sex"]]
        
        disp = new_optics["dispersion"]
        avdisp = compute_average_dispersion(local_ring, ind["cor"]["h"],disp )
        cond = np.sum(t_kicks["h"] * avdisp)
        print(f"AFTER CORRECTION, THE ENERGY CHANGE IN CORRECTORS IS: {cond}")
        return {
            "H": Resp_localH.response,
            "V": Resp_localV.response,
            "freq": t_df,
            "kicks_h": t_kicks["h"],
            "kicks_v": t_kicks["v"],
            "sex": x_sex,
            "sexp": x_sexp,
            "denergy" : energy,
            "bpm_disp": bpm_disp
        }
    
    # Computing perturved states
    state_plus = _get_state(perturbation_sign=+1)
    
    state_minus = _get_state(perturbation_sign=-1)

    # Computing all the derivatives:
        
    denominator = 2 * step

    return {
        "dH":       (state_plus["H"] - state_minus["H"]) / denominator,
        "dV":       (state_plus["V"] - state_minus["V"]) / denominator,
        "dFreq":    (state_plus["freq"] - state_minus["freq"]) / denominator,
        "dKicks_h": (state_plus["kicks_h"] - state_minus["kicks_h"]) / denominator,
        "dKicks_v": (state_plus["kicks_v"] - state_minus["kicks_v"]) / denominator,
        "dsex":     {"0": (state_plus["sex"] - state_minus["sex"]) / denominator, 
                     "1": (state_plus["sexp"] - state_minus["sexp"]) / denominator},
        "denergy": (state_plus["denergy"] - state_minus["denergy"]) / denominator,
        "ddisp":   (state_plus["bpm_disp"] - state_minus["bpm_disp"]) / denominator
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
    np.save("ORMH", ORMH)
    np.save("ORMV", ORMH)

    closed_orbit = at.find_orbit6(ring, refpts=range(len(ring)))[1] 
        
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
    dx_sex = np.zeros((n_calcs, len(ind["sex"])))
    energy = np.zeros(n_calcs)
    dni_dCFD =  np.zeros((n_calcs, len(ind["bpm"])))
    
    # 3. Execution (Single or Multi-thread)
    results = []

    if not multithread:
        for i, CFD in enumerate(target_cfds):
            print(f"Processing {i+1}/{n_calcs} (CFD index {CFD})")
            res = compute_single_CFD(ring, CFD, ORMH, ORMV, step, ind, closed_orbit, method)
            results.append(res)
            
    else:
        results = Parallel(n_jobs=-2, verbose=6)(
            delayed(compute_single_CFD)(
                ring, CFD, ORMH, ORMV, step, ind, closed_orbit, method
            ) for CFD in target_cfds
        )
    
    # Unpacking results
    for i, res in enumerate(results):
        num_dORM_dqH[i] = res["dH"]
        num_dORM_dqV[i] = res["dV"]
        dFreq_dCFD[i]   = res["dFreq"]
        dKicksH_dCFD[i] = res["dKicks_h"]
        dKicksV_dCFD[i] = res["dKicks_v"]
        x_sex[i]        = res["dsex"]["0"]
        dx_sex[i]       = res["dsex"]["1"]
        energy[i]       = res["denergy"]
        dni_dCFD[i]     = res["ddisp"]
      
    print("Calculation Finished.")
    return num_dORM_dqH, num_dORM_dqV, dFreq_dCFD, dKicksH_dCFD, dKicksV_dCFD, x_sex, dx_sex,energy, dni_dCFD


    
def dORMdEnergy(ring, ind, step = 0.1):
    """ Returns the derivative of the orbit response matrix with respect to energy"""
    ring2 = copy.deepcopy(ring)
    mcf_val = get_mcf(ring2)
    ring_freq = ring.get_rf_frequency()
    
    delta_total = (2 * step) / (mcf_val * ring_freq)
    
    ring2.set_cavity(Frequency=ring_freq + step)
    
    RespV = at.latticetools.OrbitResponseMatrix(ring2, "v", ind["bpm"], ind["cor"]["v"], steerdelta=1e-6)
    RespV.build_tracking(tol=1e-12, max_iterations=100) # FIX: build_tracking returns None
    ORMVp = RespV.response

    ring2.set_cavity(Frequency=ring_freq - step)
    
    RespV = at.latticetools.OrbitResponseMatrix(ring2, "v", ind["bpm"], ind["cor"]["v"], steerdelta=1e-6)
    RespV.build_tracking(tol=1e-12, max_iterations=100) # FIX: build_tracking returns None
    ORMVn = RespV.response
    
    return (ORMVp - ORMVn)/delta_total
    
def quickdORMdEnergy(ring, ind ,step=0.0001, c_disp =1):
    """
    Returns a dictionary containing the derivatives of the ORM to energy:
    v the vertical derivative
    h the horizontal derivative 
    
    #TODO: validate also the horizontal response!
    """
    ring2 = copy.deepcopy(ring)
    mcf_val = get_mcf(ring2)
    ring_freq = ring.get_rf_frequency()
    
    # Calculate the total energy shift changing frequency via the definition:
    # delta = -(1/alpha_c) * (df/f). 
    delta_total = (2 * step) / (mcf_val * ring_freq)
    
    # +step
    ring2.set_cavity(Frequency=ring_freq + step)
    #do one in each dimension!
    
    cORM_pv = AnaORM.AnaORM(ring2, "v",ind) 
    cORM_pv.assign_optics()
    cORM_pv.bpm.broadcasters(0, 2)
    cORM_pv.cor.broadcasters(1, 2)
    ORMVpv = cORM_pv.Rab_thick2_(cORM_pv.bpm, cORM_pv.cor)
    
    cORM_ph = AnaORM.AnaORM(ring2, "h",ind) 
    cORM_ph.assign_optics()
    cORM_ph.bpm.broadcasters(0, 2)
    cORM_ph.cor.broadcasters(1, 2)
    ORMVph = cORM_ph.Rab_thick2_(cORM_ph.bpm, cORM_ph.cor) + c_disp*cORM_ph.Rab_thick2_disp(cORM_ph.bpm, cORM_ph.cor)
    
    # -step
    ring2.set_cavity(Frequency=ring_freq - step)
    
    cORM_nv = AnaORM.AnaORM(ring2, "v",ind) 
    cORM_nv.assign_optics()
    cORM_nv.bpm.broadcasters(0, 2)
    cORM_nv.cor.broadcasters(1, 2)
    ORMVnv = cORM_nv.Rab_thick2_(cORM_nv.bpm, cORM_nv.cor)
    
    cORM_nh = AnaORM.AnaORM(ring2, "h",ind) 
    cORM_nh.assign_optics()
    cORM_nh.bpm.broadcasters(0, 2)
    cORM_nh.cor.broadcasters(1, 2)
    ORMVnh = cORM_nh.Rab_thick2_(cORM_nh.bpm, cORM_nh.cor) + c_disp*cORM_nh.Rab_thick2_disp(cORM_nh.bpm, cORM_nh.cor)
    
    return {"h": (ORMVnh - ORMVph) / delta_total, 
            "v": (ORMVnv - ORMVpv) / delta_total}

def quickdORMdEnergy_CFD(ring, ind ,step=1e-6, c_disp=1):
    """
    Returns a dictionary containing the derivatives of the ORM to energy:
    v the vertical derivative
    h the horizontal derivative 
    
    #TODO: validate also the horizontal response!
    """
    ring2 = copy.deepcopy(ring)
    mcf_val = get_mcf(ring2)
    ring_freq = ring.get_rf_frequency()
    
    delta_total = (2 * step) / (mcf_val * ring_freq)
    
    # +step
    ring2.set_cavity(Frequency=ring_freq + step)
    #do one in each dimension!
    
    cORM_pv = AnaORM.AnaORM(ring2, "v",ind) 
    cORM_pv.assign_optics()
    cORM_pv.bpm.broadcasters(0, 2)
    cORM_pv.CFD.broadcasters(1, 2)
    ORMVpv = cORM_pv.Rab_thick2_K(cORM_pv.bpm, cORM_pv.CFD)
    
    cORM_ph = AnaORM.AnaORM(ring2, "h",ind) 
    cORM_ph.assign_optics()
    cORM_ph.bpm.broadcasters(0, 2)
    cORM_ph.CFD.broadcasters(1, 2)
    ORMVph = cORM_ph.Rab_thick2_K(cORM_ph.bpm, cORM_ph.CFD) + c_disp *cORM_ph.Rab_thick2_disp(cORM_ph.bpm, cORM_ph.CFD)
    
    # -step
    ring2.set_cavity(Frequency=ring_freq - step)
    
    cORM_nv = AnaORM.AnaORM(ring2, "v",ind) 
    cORM_nv.assign_optics()
    cORM_nv.bpm.broadcasters(0, 2)
    cORM_nv.CFD.broadcasters(1, 2)
    ORMVnv = cORM_nv.Rab_thick2_K(cORM_nv.bpm, cORM_nv.CFD)
    
    cORM_nh = AnaORM.AnaORM(ring2, "h",ind) 
    cORM_nh.assign_optics()
    cORM_nh.bpm.broadcasters(0, 2)
    cORM_nh.CFD.broadcasters(1, 2)
    ORMVnh = cORM_nh.Rab_thick2_K(cORM_nh.bpm, cORM_nh.CFD) + c_disp*cORM_nh.Rab_thick2_disp(cORM_nh.bpm, cORM_nh.CFD)
    
    return {"h": (ORMVnh - ORMVph) / delta_total, 
            "v": (ORMVnv - ORMVpv) / delta_total}



def Full_SVD_cor(ring, ind, threshold, original_orbit, ORMH, ORMV, recalc_step=15):
    #Alternative method to calculate the derivative respecting the feedback condition
    if v: print("Starting Constrained Native RF SVD...")
    max_steps = 30
    
    ind_bpm = ind["bpm"]
    ind_cor = ind["cor"]
    cav_ind = at.get_refpts(ring, at.RFCavity)
    
    t_kicks = {"h": np.zeros(len(ind_cor["h"])), "v": np.zeros(len(ind_cor["v"]))}
    t_d_ring_freq = 0.0
    ring_freq = ring.get_rf_frequency()
    
    # Weight of the constraint. 
    # 1e4 forces the solver to respect the constraint above all else.
    W = 10000.0 

    for step in range(1, max_steps + 1):
        optics_data = at.get_optics(ring, refpts=range(len(ring)))
        orbit = optics_data[2]["closed_orbit"]
        all_disp = optics_data[2]["dispersion"]
        
        bpm_orbit = orbit[ind_bpm]
        dxs = np.array([bpm_orbit[i][0] - original_orbit[ind_bpm[i]][0] for i in range(len(bpm_orbit))])
        dys = np.array([bpm_orbit[i][2] - original_orbit[ind_bpm[i]][2] for i in range(len(bpm_orbit))])
        
        difference = rms(dxs) + rms(dys)
        if difference < threshold:
            if v: print(f"Convergence Reached at step {step-1}.")
            return t_kicks, t_d_ring_freq, orbit
        
        if step == recalc_step:
            Resp_H = at.latticetools.OrbitResponseMatrix(
                ring, "h", bpmrefs=ind_bpm, steerrefs=ind_cor["h"], cavrefs=cav_ind
            )
            Resp_H.build_tracking(tol=1e-12, max_iterations=100)
            ORMH = Resp_H.response
            
            Resp_V = at.latticetools.OrbitResponseMatrix(
                ring, "v", bpmrefs=ind_bpm, steerrefs=ind_cor["v"]
            )
            Resp_V.build_tracking(tol=1e-12, max_iterations=100)
            ORMV = Resp_V.response

        # --- APPLY THE DISPERSION CONSTRAINT ---
        # 1. Calculate average dispersion at correctors
        avg_disp_h_cor = compute_average_dispersion(ring, ind_cor["h"], all_disp)
        
        # 2. Build the constraint row: [eta_1, eta_2, ..., eta_n, 0]
        # The 0 at the end is for the RF cavity column (it is allowed to change energy)
        constraint_row = np.zeros(len(ind_cor["h"]) + 1)
        constraint_row[:-1] = avg_disp_h_cor * W
        
        # 3. Augment the ORMH matrix vertically
        M_aug = np.vstack((ORMH, constraint_row))
        
        # 4. Augment the target vector (dxs) with a 0 for the constraint target
        dxs_aug = np.append(dxs, 0.0)

        # --- SVD SOLVE ---
        sol_h, *_ = np.linalg.lstsq(M_aug, dxs_aug, rcond=None)
        kicks_v, *_ = np.linalg.lstsq(ORMV, dys, rcond=None)

        # Extract values
        kicks_h = -sol_h[:-1] # Correctors
        d_freq  = -sol_h[-1]  # RF Frequency
        
        # Optional: Print the constraint check to prove it works
        path_length_leak = np.sum(kicks_h * avg_disp_h_cor)
        if v: print(f"Step {step} | dFreq: {d_freq:.4e} | Path Leak: {path_length_leak:.2e} | diff: {difference:.2e}")

        # Apply
        t_kicks["h"] += kicks_h
        t_kicks["v"] += kicks_v
        t_d_ring_freq += d_freq

        applyCorrections(ring, ind_cor["h"], kicks_h, "h")
        applyCorrections(ring, ind_cor["v"], kicks_v, "v")
        
        new_ring_freq = ring_freq + d_freq
        ring.set_cavity(Frequency=new_ring_freq)
        ring_freq = new_ring_freq 

    if v: print("Iteration did not converge within max steps")
    return t_kicks, t_d_ring_freq, orbit