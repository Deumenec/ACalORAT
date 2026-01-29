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
import matplotlib.pyplot as plt

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

def Cor_SVD_cor(ring , ind_bpm, ind_cor, threshold, original_orbit):
    """
    Uses SVD to correct orbit ONLY WITH correctors.
    Afterwards, it changes the RF frequency to cancel "corrector drift"
    (Making the sum of kickangles zero)
    At least two iterations are needed for the method to work well
    threshold: accepted threshold in the orbit difference.
    """
    
    #TODO: Implement as an argument
    max_steps = 10
    
    #Total kicks performed
    t_kicks = {"h":np.zeros(len(ind_cor["h"])), "v": np.zeros(len(ind_cor["v"]))}
    t_d_ring_freq = 0
    
    #Kicks for current iteration
    kicks = {"h":np.zeros(len(ind_cor["h"])), "v": np.zeros(len(ind_cor["v"]))}
    
    #Finds orbit at the beginig
    orbit = at.find_orbit6(ring, refpts=ind_bpm)[1]
    dxs = np.array([orbit[i][0] -original_orbit[ind_bpm[i]][0] for i in range(len(orbit))])
    dys = np.array([orbit[i][2] -original_orbit[ind_bpm[i]][2] for i in range(len(orbit))])
    difference = rms(dxs)+rms(dys)
    if difference<threshold:
        print("No correction was needed")
        return t_kicks, orbit #The original one as expected
    
    #PREGUNTAR: Al canviar l'energia, canvia la matriu de resposta així que per ser exactes cal tornar a calcular-la cada cop encara que triga molt més així...
    while max_steps >0:
        max_steps -=1
        
        #Calculo dispersió i matriu de resposta a cada iteració ja que canvien al canviar l'energia 
        optics = at.get_optics(ring, refpts=range(len(ring)))[2]
        dispersion = {"h": np.array( [optics["dispersion"][i][0] for i in range(len(ring)) ])[ind_bpm],
                      "v": np.array( [ optics["dispersion"][i][2] for i in range(len(ring)) ])[ind_bpm] } #La vertical es casi 0 així que no la faig servir al final...
        
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
        #Em guardo el kick total que s'ha fet.
        t_kicks["h"] = t_kicks["h"]+kicks["h"]
        t_kicks["v"] = t_kicks["v"]+kicks["v"]
        
        applyCorrections(ring, ind_cor["h"], kicks["h"], "h")
        applyCorrections(ring, ind_cor["v"], kicks["v"], "v")
        
        
        #Aplico la formula deduïda per la correcció a l'energia
        sum_kicks= np.sum(kicks["h"])
        print("amb d_freq: ", t_d_ring_freq ," Drift_kick: ", sum_kicks) #Comprovar que la correcció cada cop sigui millor.
        ring_freq = ring.get_rf_frequency()
        mcf       = get_mcf(ring)
        quotient, *_ =np.linalg.lstsq(ORMH, dispersion["h"])
        #MOLT IMPORTANT: EL FACTOR (4) al denominador AQUÍ ÉS PER EL PERFIL D'AQUEST MÍNIM
        #Numericament sembla que millora molt la convergència agafant 4 perquè crec que la funció no és diferenciable a aquest punt. Això té tot el sentit del mon si es pensa que la funció al final es com una suma de variables aleatòries que depenen de l'energia així que té un perfil com abs (freq)**(1/4 )     
        d_ring_freq = -sum_kicks*mcf*ring_freq/(np.sum(quotient)*6)
        #Aplico el canvi a la freqüència i registro el canvi total fet a l'energia
        new_ring_freq = (ring_freq+d_ring_freq)
        ring.set_cavity(Voltage = None,  Frequency = new_ring_freq)
        t_d_ring_freq += d_ring_freq

        #Comparo l'orbita amb l'original abans de les correccions
        orbit = at.find_orbit6(ring, refpts=ind_bpm)[1]
        dxs = np.array([orbit[i][0] -original_orbit[ind_bpm[i]][0] for i in range(len(orbit))])
        dys = np.array([orbit[i][2] -original_orbit[ind_bpm[i]][2] for i in range(len(orbit))])
        difference = rms(dxs)+rms(dys)
        print("diff: ", difference)
        
        #Si la correcció ha estat satisfactoria, retorno l'anell a l'estat original i ja la tinc!
        if difference<threshold:
            """  #Per si cal desfer les correccions
            applyCorrections(ring, ind_cor["h"],  -t_kicks["h"], "h")
            applyCorrections(ring, ind_cor["v"], -t_kicks["v"], "v")
            new_ring_freq = ring.get_rf_frequency() -t_d_ring_freq
            ring.set_cavity(Frequency = new_ring_freq)
            """
            print("Iteració acabada, canvi en la frequencia de", t_d_ring_freq)
            return t_kicks, t_d_ring_freq, orbit
        plt.plot(t_kicks["h"], label = str(max_steps))
    #No puc saber quina és la correcció que ha fallat amb aquest mètode.
    print("Iteration did not converge")
    applyCorrections(ring, ind_cor["h"],  -t_kicks["h"], "h")
    applyCorrections(ring, ind_cor["v"], -t_kicks["v"], "v")
    return t_kicks, d_ring_freq, orbit




def Full_SVD_cor(ring , ind_bpm, ind_cor, threshold, original_orbit):
    """
    Uses SVD to correct orbit WITH correctors and RF at the same time (one extra column in the matrix).
    Afterwards, it changes the RF frequency to cancel "corrector drift"
    (Making the sum of kickangles zero) 
    At least two iterations are needed for the method to work well
    """

    max_steps = 10
    
    #Total kicks performed
    t_kicks = {"h":np.zeros(len(ind_cor["h"])), "v": np.zeros(len(ind_cor["v"]))}
    t_d_ring_freq = 0
    
    #Kicks for current iteration
    kicks = {"h":np.zeros(len(ind_cor["h"])), "v": np.zeros(len(ind_cor["v"]))}
    
    #Finds orbit at the beginig
    orbit = at.find_orbit6(ring, refpts=ind_bpm)[1]
    dxs = np.array([orbit[i][0] -original_orbit[ind_bpm[i]][0] for i in range(len(orbit))])
    dys = np.array([orbit[i][2] -original_orbit[ind_bpm[i]][2] for i in range(len(orbit))])
    difference = rms(dxs)+rms(dys)
    if difference<threshold:
        print("No correction was needed")
        return t_kicks, orbit #The original one as expected
    
    #PREGUNTAR: Al canviar l'energia, canvia la matriu de resposta així que per ser exactes cal tornar a calcular-la cada cop encara que triga molt més així...
    while max_steps >0:
        max_steps -=1
        
        #Calculo dispersió i matriu de resposta a cada iteració ja que canvien al canviar l'energia 
        optics = at.get_optics(ring, refpts=range(len(ring)))[2]
        dispersion = {"h": np.array( [optics["dispersion"][i][0] for i in range(len(ring)) ])[ind_bpm],
                      "v": np.array( [ optics["dispersion"][i][2] for i in range(len(ring)) ])[ind_bpm] } #La vertical es casi 0 així que no la faig servir al final...
        
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
        #Em guardo el kick total que s'ha fet.
        t_kicks["h"] = t_kicks["h"]+kicks["h"]
        t_kicks["v"] = t_kicks["v"]+kicks["v"]
        
        applyCorrections(ring, ind_cor["h"], kicks["h"], "h")
        applyCorrections(ring, ind_cor["v"], kicks["v"], "v")
        
        
        #Aplico la formula deduïda per la correcció a l'energia igualment a cada iteració per assegurar que es compleix la correcció, en qualsevol cas ara la convergència hauria de ser millor!
        sum_kicks= np.sum(kicks["h"])
        print("amb d_freq: ", t_d_ring_freq ," Drift_kick: ", sum_kicks) #Comprovar que la correcció cada cop sigui millor.
        ring_freq = ring.get_rf_frequency()
        mcf       = get_mcf(ring)
        quotient, *_ =np.linalg.lstsq(ORMH, dispersion["h"])
        #MOLT IMPORTANT: EL FACTOR (4) al denominador AQUÍ ÉS PER EL PERFIL D'AQUEST MÍNIM
        #Numericament sembla que millora molt la convergència agafant 4 perquè crec que la funció no és diferenciable a aquest punt. Això té tot el sentit del mon si es pensa que la funció al final es com una suma de variables aleatòries que depenen de l'energia així que té un perfil com abs (freq)**(1/4 )     
        d_ring_freq = -sum_kicks*mcf*ring_freq/(np.sum(quotient)*6)
        #Aplico el canvi a la freqüència i registro el canvi total fet a l'energia
        new_ring_freq = (ring_freq+d_ring_freq)
        ring.set_cavity(Voltage = None,  Frequency = new_ring_freq)
        t_d_ring_freq += d_ring_freq

        #Comparo l'orbita amb l'original abans de les correccions
        orbit = at.find_orbit6(ring, refpts=ind_bpm)[1]
        dxs = np.array([orbit[i][0] -original_orbit[ind_bpm[i]][0] for i in range(len(orbit))])
        dys = np.array([orbit[i][2] -original_orbit[ind_bpm[i]][2] for i in range(len(orbit))])
        difference = rms(dxs)+rms(dys)
        print("diff: ", difference)
        
        #Si la correcció ha estat satisfactoria, retorno l'anell a l'estat original i ja la tinc!
        if difference<threshold:
            """  #Per si cal desfer les correccions però la gràcia és que el ring surt d'aquí corregit
            applyCorrections(ring, ind_cor["h"],  -t_kicks["h"], "h")
            applyCorrections(ring, ind_cor["v"], -t_kicks["v"], "v")
            new_ring_freq = ring.get_rf_frequency() -t_d_ring_freq
            ring.set_cavity(Frequency = new_ring_freq)
            """
            print("Iteració acabada, canvi en la frequencia de", t_d_ring_freq)
            return t_kicks, t_d_ring_freq, orbit
        plt.plot(t_kicks["h"], label = str(max_steps))
    #No puc saber quina és la correcció que ha fallat amb aquest mètode.
    print("Iteration did not converge")
    applyCorrections(ring, ind_cor["h"],  -t_kicks["h"], "h")
    applyCorrections(ring, ind_cor["v"], -t_kicks["v"], "v")
    return t_kicks, d_ring_freq, orbit







def compute_single_CFD(ring, CFD, ORM, direction, step, ind_bpm, ind_cor, closed_orbit, method):
    
    #Make a deep copy of the ring
    local_ring = copy.deepcopy(ring)
    #Change the CFD strength ratio(quadrupole and KickAngle)
    B0 = local_ring[CFD].BendingAngle/ local_ring[CFD].Length
    ratio = B0/local_ring[CFD].PolynomB[1]
    
    #Activating the CFD
    local_ring[CFD].PolynomB[1] += step
    local_ring[CFD].PolynomB[0] += step*ratio
    
    print(f"S'està analitzant el CFD {CFD} i s'ha fet els canvis a B0 {step*ratio} B1 {step}")
    #Mètode iteratiu que efectua la correcció segons quin s'hagi triat a l'argument de la funció
    
    if method == "Cor_SVD":
        Cor_SVD_cor(local_ring , ind_bpm, ind_cor, 1e-13, closed_orbit) 
        
    if method == "Full_SVD":
        Full_SVD_cor(local_ring , ind_bpm, ind_cor, 1e-13, closed_orbit) 
        
    #Observem que encara que hagi trobat l'estat "corregit" de l'anell, cal tornar
    #a calcular la matriu de resposta perqué aquesta pot haver canviat a l'última iteració, de fet és la gràcia de fer tot això.
    Resp_local = at.latticetools.OrbitResponseMatrix(
        local_ring, direction, ind_bpm, ind_cor["h"])
    
    Resp_local.build_tracking()
    
    return (Resp_local.response - ORM) / step

def dORM_dCFD(ring, ind_bpm, ind_cor, ind_CFD, ind_RF, step, direction, num=5, multithread = False, method = "Full_SVD"):
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
    num : int
        allows to limit the total number of CFD calculated for quick tests only with the first ones
    multithread : Bool
        If set True, uses Parallel with all the cores in the CPU and calculates all the CFD responses
    method : str
        Method used to calculate the active feedback
        "Cor_SVD" : Uses only correctors to calculate the correction in each step
        "Full_SVD": Uses correctors and an extra column for dispersion and energy to compute the SVD
        
        
    Returns
    -------
    num_dORM_dq: np.array 
        The dORM_dq rank 3 tensor with indices dORM_dq[quadrupole][bpm][corrector]
    """

    num_dORM_dq = np.zeros([len(ind_CFD), len(ind_bpm), len(ind_cor["h"])])
    closed_orbit = at.find_orbit(ring, refpts=range(len(ring)))[1] #Compute the closed orbit along all elements in the ring

    
    Resp = at.latticetools.OrbitResponseMatrix(ring,direction, ind_bpm, ind_cor[direction]) #class for computing the ORM in the original direction
    print("hii")
    ORM = Resp.build_tracking()
    if not multithread:
        for i, CFD in enumerate(ind_CFD[:3]):
            print(i)
            num_dORM_dq[i] = compute_single_CFD(ring, CFD, ORM, direction, step, ind_bpm, ind_cor, closed_orbit, method)
    if multithread:
        num_dORM_dq = Parallel(n_jobs=-1, verbose=10)(
            delayed(compute_single_CFD) (ring, CFD, ORM, direction, step, ind_bpm, ind_cor, closed_orbit) for CFD in ind_CFD)
        num_dORM_dq = np.array(num_dORM_dq)
     
    return num_dORM_dq



    