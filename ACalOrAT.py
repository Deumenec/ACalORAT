#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 19:29:25 2025

@author: deumenec
Class with defined functions to compute ORMs and their derivatives
"""
import numpy as np
import at

dir_dict = {"h": 0, "v": 1}

def broadcast_vector(v, axis, ndim):
    """
    v    : 1D vector to broadcast
    axis : axis along which v should vary
    ndim : number of dimensions of the target tensor
    """
    shape = [1] * ndim
    shape[axis] = -1            # -1 lets reshape infer v's length
    return v.reshape(shape)


class ACalOrAT:
    """Analitic Calculation of Orbit by AT
    Methods used for different calculations involving 

    Parameters
    ----------
    ring : at.lattice.lattice_object.Lattice
        lattice used
    direction : np.array
        "v" or "h", direction in which the calculations are performed
    ind_bpm : np.array
        Beam Position Monitors for the orbit response matrix
    ind_cor : np.array
        Correctors used for the ORM
    ind_quad : np.array
        Quadrupoles to take the derivatives
    ind_dip : np.array
        Indices for the dipoles in the matrix for dispersion calculations
    """
    def __init__(self, ring :at.lattice.lattice_object.Lattice, direction: str, ind_bpm: np.ndarray , ind_cor: np.ndarray ,ind_quad:np.ndarray, ind_dip: np.ndarray ):
        self.ring     = ring
        self.ind_bpm  = ind_bpm
        self.ind_quad = ind_quad
        self.ind_cor  = ind_cor
        self.ind_dip  = ind_dip

        self.dir_ind = dir_dict[direction] #Index of the direction in at.get_optics
        self.all_optics = at.get_optics(ring, refpts=range(len(ring))) # get all the optics from the ring
        self.tune = self.all_optics[1]["tune"][self.dir_ind]

    class Elements:
        """Group of elements of a certain type bpms, correctors, dipoles, or 
        quadrupoles containing all the important atributes from them with 
        their optics
        """
        def __init__(self, calc ,ind):
            """
            ind : list of indices of those elements, assigns the optics for each calculation
            """
            self.ind = ind
            self.beta = np.array([calc.all_Optics[2]["beta"][i][calc.dir_ind] for i in ind])
            self.alpha= np.array([calc.all_Optics[2]["alpha"][i][calc.dir_ind] for i in ind])
            self.gamma= (self.alpha*self.alpha+1)/self.beta
            self.mu   = np.array([calc.all_Optics[2]["mu"][i][calc.dir_ind] for i in ind])
            self.dispersion = np.array([calc.all_Optics[2]["dispersion"][i][calc.dir_ind] for i in ind])
            if hasattr(calc.ring[ind[0]], "BendingAngle"): 
                self.Bend = np.array([calc.ring[i].BendingAngle for i in ind])
                self.HasBend = True
            self.HasBend = False
            if hasattr(calc.ring[ind[0]], "Length"): 
                self.Length = np.array([calc.ring[i].Length for i in ind])
                self.HasLength = True
            self.HasLength = False
            if hasattr(calc.ring[ind[0]], "K"): 
                self.K = np.array([calc.ring[i].Length for i in ind])   
                self.HasK = True
            self.HasK = False
        def broadcasters(self, axis, ndim):
            """
            Prepares variables for broadcasting with numpy functions placing them
            in a np.array with the required dimensions and in the required broadcasting
            dimension.
            """
            self.betaB = broadcast_vector(self.beta, axis, ndim)
            self.alphaB = broadcast_vector(self.balpha, axis, ndim)
            self.gammaB = broadcast_vector(self.gamma, axis, ndim)
            self.muB = broadcast_vector(self.mu, axis, ndim)
            if (self.HasBend == True): self.BendB = broadcast_vector(self.Bend, axis, ndim)
            if (self.HasLength == True): self.BendB = broadcast_vector(self.Length, axis, ndim)
            if (self.HasK == True): self.BendB = broadcast_vector(self.K, axis, ndim)
            
    def assign_optics(self):
        self.bpm = self.Elements(self, self.ind_bpm)
        self.cor = self.Elements(self, self.ind_cor)
        self.quad= self.Elements(self, self.ind_quad)
        self.dip = self.Elements(self, self.ind_dip)
    
    def thin
