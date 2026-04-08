#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 12:43:48 2026

@author: dhuerta

Validated standard methods using the defined functions in AnaORM to compute relevant quantities
"""

from . import AnaORM

def ORM(ring, ind, direction):
    """
    Calculates the Orbit Response Matrix analytically in correctors
    using the method for correctors without quadrupole component.

    Parameters
    ----------
    ring : at.lattice.lattice_object.Lattice
    
        ring used in the calculations
        
    ind : dict
    
        index dictionary set up for the AnaORM class
        
    direction : str
    
        "h" or "v" depending on the ORM calculated.

    Returns
    -------
    ORM : np.array
    
        Orbit response matrix with dimensions [bpm, correctors]

    """
    
    
    cORM = AnaORM.AnaORM(ring, direction, ind)
    cORM.assign_optics()
    cORM.bpm.broadcasters(0, 2)
    cORM.cor.broadcasters(1, 2)
    if direction == "v":
        return cORM.Rab_thick2_(cORM.bpm, cORM.cor)
    elif direction == "h":
        return cORM.Rab_thick2_(cORM.bpm, cORM.cor) + cORM.Rab_thick2_disp(cORM.bpm, cORM.cor)


def dORM_dpure_quad(ring, ind, direction):
    """
    Calculates the Jacobian of the ORM with respect to changing the strength of
    pure quadrupoles in the ring.


    Parameters
    ----------
    ring : at.lattice.lattice_object.Lattice
    
        ring used in the calculations
        
    ind : dict
    
        index dictionary set up for the AnaORM class
        
    direction : str
    
        "h" or "v" depending on the ORM calculated.

    Returns
    -------
    ORM : np.array
    
        Orbit response matrix with dimensions [bpm, correctors]

    """
    #TODO
    
    return 0

def dORM_dCFD_quad(ring, ind, direction):
    """
    Calculates the Jacobian of the ORM with respect to changing ONLY the strength of
    the quadrupole component in CFDs.


    Parameters
    ----------
    ring : at.lattice.lattice_object.Lattice
    
        ring used in the calculations
        
    ind : dict
    
        index dictionary set up for the AnaORM class
        
    direction : str
    
        "h" or "v" depending on the ORM calculated.

    Returns
    -------
    dORM : np.array
    
        Orbit response matrix with dimensions [quadrupole, bpm, correctors]

    """
    #TODO
    
    return 0
 
def ddisp_dquad(ring, ind, ind_quad):
    """
    Calculates the change in dispersion in bpms with respect to changing ONLY
    THE QUADRUPOLE component in either pure quads or CFDs (works for one at a time).

    """
    #TODO
    return 0