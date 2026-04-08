#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 19:29:25 2025

@author: dhuerta

Class with defined functions to compute ORMs and their derivatives
"""
    
import at
from . import numerical
from .Elements import Elements

from .physics.ORM import ORM
from .physics.dORM_dq    import dORM_dq
from .physics.dORM_dh    import dORM_dh
from .physics.Dispersion import Dispersion
from .physics.Chromatics import Chromatics


#dir_dict = {"h": 0, "v": 1} #For further reference, see the structure of the at.get_optics output

  
class AnaORM(ORM, dORM_dq, dORM_dh, Dispersion, Chromatics):
    """
    Method used for all analytical calculations. It handles all global ring 
    variables and optics in groups of elements, which it gets from the at.get_optics function. 
    
    For each element type, it contains an instance of the Elements Class with
    all relevant the information.

    Parameters
    ----------
    ring : at.lattice.lattice_object.Lattice
    
        lattice used
        
    direction : str
    
        "v" or "h", direction in which the calculations are performed.
        
    ind    : dict
    
        Indices for groups of elements in the ring, it is required to have:
            bpm, cor["v"], cor["h"], quad, dip and optionally sex and CFD 
    """
    
    def __init__(self, ring :at.lattice.lattice_object.Lattice, 
                 direction: str, 
                 ind: dict,
                 old_optics = False):
        """

        Parameters
        ----------
        ring : at.lattice.lattice_object.Lattice

        direction : str

            "v" or "h", direction in which the calculations are performed.

        ind : dict

            Indices for groups of elements in the ring, it is required to have:
                bpm, cor["v"], cor["h"], quad, dip and optionally sex and CFD
        
        old_optics : np.array, optional

            Full at.get_optics output, if given the optics are used instead of calculating them. The default is False.
        """
        self.ring     = ring
        self.mcf = numerical.get_mcf(ring)
        self.circumference = ring.circumference
        self.ind  = ind
        self.dir  = direction
        self.dir_ind = {"h": 0, "v": 1}[direction] #Index of the direction in at.get_optics in which the derivative is calculated
        if old_optics is False: self.all_optics = at.get_optics(ring, refpts=range(len(ring))) # get all the optics from the ring
        else: self.all_optics = old_optics
        self.tune = self.all_optics[1]["tune"][self.dir_ind]
        self.sgn  = -(-1)**self.dir_ind #sign associated with that direction for quadrupoles

    def assign_optics(self): 
        """
        Assigns the optics for all the elements in the ring.
        """
        self.bpm = Elements(self.ring, self.all_optics ,self.ind["bpm"], self.dir_ind, self.sgn)    
        self.cor = Elements(self.ring, self.all_optics ,self.ind["cor"][self.dir], self.dir_ind, self.sgn)
        self.quad= Elements(self.ring, self.all_optics ,self.ind["quad"], self.dir_ind,self.sgn)
        self.dip = Elements(self.ring, self.all_optics ,self.ind["dip"], self.dir_ind, self.sgn)
        if "CFD" in self.ind: self.CFD= Elements(self.ring, self.all_optics ,self.ind["CFD"], self.dir_ind,  self.sgn)
        if "sex" in self.ind: self.sex= Elements(self.ring, self.all_optics ,self.ind["sex"], self.dir_ind,  self.sgn)
    
    def add_element(self,name, ind, direction): 
        """
        ADDS a new element to the calculation method to perform different calculations if using non default indices.
        """
        setattr(self,name,Elements(self.ring, self.all_optics ,ind, {"h": 0, "v": 1}[direction], -(-1)**({"h": 0, "v": 1}[direction])))
    

    

    

    
    
    

