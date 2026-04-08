#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:11:27 2026

@author: dhuerta
"""

import numpy as np
import at

def broadcast_vector(v, axis, ndim):
    """
    Takes a numpy 1D vector and returns the numpy tensor [None, ..., :, ..., None]
    with the vector broadcasting in the specified axis.
    
    Parameters
    ----------
    v    : 1D vector to broadcast
    axis : axis along which v should vary
    ndim : number of dimensions of the target tensor
    """
    shape = [1] * ndim
    shape[axis] = -1
    return v.reshape(shape)


class Elements:
    r"""
    Data structure representing a homogenious group of accelerator elements.
    
    Each type of elements is represented by an instance of this class (for example
    bpms, quadrupoles, dipoles etc). It manages all relevant optical functions
    and attributes for vectorized analytical calculations.
    
    By default as in ATcollab, all optics are at the ENTRANCE of the element.
    
    Attributes
    ----------
    If broadcasting is applied, all attributes have another duplicate ending with "B" broadcasting in the necessary dimensions.

    beta : np.ndarray (complex)
    
        The betatron amplitude function ($\beta$) at the entrance of the elements.
        
    alpha : np.ndarray (complex)
    
        The Twiss $\alpha$ parameter at the entrance of the elements.
        
    gamma : np.ndarray (complex)
    
        The Twiss $\gamma$ parameter, derived from $\beta$ and $\alpha$.
        
    mu : np.ndarray
    
        The betatron phase advance ($\mu$) at the entrance of the elements.
        
    dispersion : np.ndarray (complex)
    
        The dispersion function ($\eta$) at the entrance of the elements.
        
    dispersionp : np.ndarray (complex)
    
        The derivative of the dispersion function ($\eta'$) at the entrance.
                                                   
    Length : np.ndarray
    
        The length of the elements (if applicable).
        
    Bend : np.ndarray
    
        The bending angle of the elements (if applicable).
        
    K : np.ndarray (complex)
    
        The normalized quadrupole gradient ($K$). The sign is adjusted depending on the dimension by sgn
        
    S : np.ndarray (complex)
    
        The normalized sextupole gradient. Sign convention is adjusted by `sgn`.
    """
    
    def __init__(self, ring, all_optics ,ind, dir_ind, sgn):
        """
        Initializes the Elements object by calculating the optics of the lattice

        Parameters
        ----------
        ring : list
    
        all_optics : tuple
        
            Output of at.get_optics whit the optics all the optics in the ring

        ind : np.ndarray or list
        
            Indices of the elements in the family inside the ring.
            
        dir_ind : int
        
            The transverse plane index (0 for horizontal, 1 for vertical).
            
        sgn : int
        
            -1 for horizontal and 1 for vertical, sign of the quadrupole strength
            in the concerning dimension
        """
        
        if len(ind) == 0: return
        
        self._ring= ring
        self._ind = ind
        self._sgn  = sgn
        self.beta = np.array([all_optics[2]["beta"][i][dir_ind] for i in ind], dtype= complex)
        self.alpha= np.array([all_optics[2]["alpha"][i][dir_ind] for i in ind], dtype= complex)
        self.gamma= (self.alpha*self.alpha+1)/self.beta
        self.mu   = np.array([all_optics[2]["mu"][i][dir_ind] for i in ind])
        self.dispersion = np.array([all_optics[2]["dispersion"][i][2*dir_ind] for i in ind], dtype= complex)
        self.dispersionp = np.array([all_optics[2]["dispersion"][i][2*dir_ind+1] for i in ind], dtype= complex)
        if hasattr(ring[ind[0]], "BendingAngle"):self.Bend = np.array([ring[i].BendingAngle for i in ind])
        if hasattr(ring[ind[0]], "Length"):      self.Length = np.array([ring[i].Length for i in ind])
        if (hasattr(ring[ind[0]], "PolynomB" ) and len(ring[ind[0]].PolynomB) >=2):  self.K = np.array([-sgn*ring[i].PolynomB[1] for i in ind], dtype= complex)
        if (hasattr(ring[ind[0]], "PolynomB" ) and len(ring[ind[0]].PolynomB) >=3):  self.S = np.array([-sgn*ring[i].PolynomB[2] for i in ind], dtype= complex)
        if hasattr(ring[ind[0]], "EntranceAngle"): self.EntranceAngle = np.array([ring[i].EntranceAngle for i in ind], dtype= complex)   
        if hasattr(ring[ind[0]], "ExitAngle"): self.ExitAngle = np.array([ring[i].ExitAngle for i in ind], dtype= complex)   
        
    def broadcasters(self, axis, ndim):
        """
        Prepares variables in Elements for broadcasting with numpy.
        
        It loops through all the attributes in the class which don't start 
        with _ and creates an array with the same name and the suffix "B" with
        the values broadcasting in the necessary direction.
        
        Parameters
        ----------
        axis : int
            The axis where the values are aligned
        ndim : int
            The total number of axis for the final calculation.
        
        Example: bpm.broadcasters(2, 3) adds among all the other optics: muB = mu[None, None, :]
        
        """
        self._bAxis = axis #Saving the broadcasting axis
        self._ndim  = ndim
        variables = [attr for attr in self.__dict__ if not attr.startswith('_')] #Crea broadcasters per totes les variables igual
        for var in variables:
            setattr(self, var+"B", broadcast_vector( getattr(self, var) , axis, ndim))
            
    def correct_entrance(self):
        """ Applies edge focusing to correct the entrance optics at the start 
        of a bending magnet due to the thin quadrupole caused by the non-zero 
        entrance angle.
        
        It corrects the alpha, the gamma, the derivative of dispersion 
        and in the horizontal direction it adds the extra focusing due to bending.
        
        Finally, IF the element is broadcasting in some dimension, it rebuilds
        the broadcasting.
        """
        #To prevent the correction from happening multiple times.
        if hasattr(self, "_corrected"): return
        setattr(self, "_corrected", True)
        
        #Entrance angle correction to betas
        if hasattr(self, "EntranceAngle"):
            self.alpha += self._sgn *self.beta*np.tan(self.EntranceAngle)*(self.Bend/self.Length)
            #Gamma at new optics
            self.gamma = (1 + self.alpha**2) / self.beta
            
            #Dispersion correction in the horizontal dimension
            if self._sgn ==-1: 
                self.dispersionp += np.tan(self.EntranceAngle)*(self.Bend/self.Length)*self.dispersion
                
        #strength correction for the horizontal dimension
        if self._sgn == -1 and hasattr(self, "Bend"):
            if not hasattr(self, "K"):
                self.K =0
            self.K +=  (self.Bend/self.Length)*(self.Bend/self.Length)
            
        if hasattr(self, "_bAxis"):
            self.broadcasters(self._bAxis, self._ndim)
        return
        
    def correct_strength(self):
        """ Calculates the closed orbit off-momentum and uses it to correct the actual effective
        strength of the quadrupoles... It can be applied before broadcasting but it is a really mince correction... 0.1%
        
        In principle also this momentum deviation is already in the same units as beam rigidity
        """
        #This term has an almost neglegible effect.
        co = at.find_orbit6(self._ring, self._ind)[1] 
        deviations = np.array([i[4] for i in co]) #The momentum deviation is in the 4rth component
        self.K = self.K/(1+deviations)
        
    def average(self):
        """Computes the average of the dispersion inside of elements in the 
        family by applying entrance angle, propagating and integrating inside.
        """
        #We start by considering the effect of the entrance fringe field:
        dispersionp = self.dispersionp
        if hasattr(self, "EntranceAngle"):
            dispersionp +=  -self.dispersion*np.tan(self.EntranceAngle)*(self.Bend/self.Length)
        if hasattr(self, "Bend"):
            k_total = self.K+(self.Bend/self.Length)**2 #Effective K value
            phi = np.sqrt(k_total)*self.Length
            self.avDispersion = self.dispersion *np.sin(phi)/phi + dispersionp/(k_total*self.Length)*(1-np.cos(phi))+self.Bend/(self.Length*k_total)*(1-np.sin(phi)/phi)
        if (not hasattr(self, "Bend")) and hasattr(self, "K"):
            k_total = self.K
            phi = np.sqrt(k_total)*self.Length
            self.avDispersion = self.dispersion + self.dispersionp*self.Length/2
        if (not hasattr(self, "Bend")) and not hasattr(self, "K"):
            self.avDispersion = self.dispersion + self.dispersionp*self.Length/2
        if hasattr(self, "_bAxis"):
            setattr(self, "avDispersion"+"B", broadcast_vector( self.avDispersion ,self._bAxis, self._ndim))