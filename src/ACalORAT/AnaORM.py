#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 19:29:25 2025

@author: deumenec
Class with defined functions to compute ORMs and their derivatives
"""
import numpy as np
import at
import copy

from . import numerical

dir_dict = {"h": 0, "v": 1}

def get_mcf(ring):
    if (ring.is_6d==True):
        ring.disable_6d()
        mcf = ring.mcf
        ring.enable_6d()
        return mcf
    return ring.mcf

def broadcast_vector(v, axis, ndim):
    """
    v    : 1D vector to broadcast
    axis : axis along which v should vary
    ndim : number of dimensions of the target tensor
    """
    shape = [1] * ndim
    shape[axis] = -1            # -1 lets reshape infer v's length
    return v.reshape(shape)


    
class Elements:
    """Group of elements of a certain type: bpms, correctors, dipoles, 
    quadrupoles or sextupoles containing all the important atributes from them with 
    their optics. By default all optics are before entrance. at the element
    """
    def __init__(self, ring, all_optics ,ind, dir_ind, sgn):
        """
        all_optics
        ind : np.array 
        list of only a certain class of elements elements, assigns the optics for each calculation
        """
        if len(ind) == 0: return
        self._ring= ring
        self._ind = ind
        self.beta = np.array([all_optics[2]["beta"][i][dir_ind] for i in ind], dtype= complex)
        self.alpha= np.array([all_optics[2]["alpha"][i][dir_ind] for i in ind], dtype= complex)
        self.gamma= (self.alpha*self.alpha+1)/self.beta
        self.mu   = np.array([all_optics[2]["mu"][i][dir_ind] for i in ind])
        self.dispersion = np.array([all_optics[2]["dispersion"][i][2*dir_ind] for i in ind], dtype= complex)
        self.dispersionp = np.array([all_optics[2]["dispersion"][i][2*dir_ind+1] for i in ind], dtype= complex)
        if hasattr(ring[ind[0]], "BendingAngle"):self.Bend = np.array([ring[i].BendingAngle for i in ind])
        if hasattr(ring[ind[0]], "Length"):      self.Length = np.array([ring[i].Length for i in ind])
        if hasattr(ring[ind[0]], "K"):           self.K = np.array([-sgn*ring[i].K for i in ind], dtype= complex)
        if hasattr(ring[ind[0]], "EntranceAngle"): self.EntranceAngle = np.array([ring[i].EntranceAngle for i in ind], dtype= complex)   
        if hasattr(ring[ind[0]], "ExitAngle"): self.ExitAngle = np.array([ring[i].ExitAngle for i in ind], dtype= complex)   
    def correct_entrance(self):
        """ To be called for dipoles to correct the optic functions inside of
        them after the fringe field and entrance angles. And adjust force.
        """

        self.alpha = self.alpha -self.beta*np.tan(self.EntranceAngle)*(self.Bend/self.Length)
        self.K = self.K + (self.Bend/self.Length)*(self.Bend/self.Length)
        
    def correct_strength(self):
        """ Calculates the closed orbit off-momentum and uses it to correct the actual effective
        strength of the quadrupoles... It can be applied before broadcasting but it is a really mince correction... 0.1%
        
        In principle also this momentum deviation is already in the same units as beam rigidity
        """
        co = at.find_orbit6(self._ring, self._ind)[1] 
        deviations = np.array([i[4] for i in co]) #The momentum deviation is in the 4rth component
        self.K = self.K/(1+deviations)
        
    def average(self):
        """Computes the average of the optical functions inside elements by slicing and propagating inside, it may be useful for
        some elements to be considered thin
        """
        k_total = self.K+(self.Bend/self.Length)**2 #Effective K value
        phi = np.sqrt(k_total)*self.Length
        self.avDispersion = self.dispersion *np.sin(phi)/phi + self.dispersionp/(k_total*self.Length)*(1-np.cos(phi))+self.Bend/(self.Length*k_total)*(1-np.sin(phi)/phi)
        
    def broadcasters(self, axis, ndim):
        """
        Prepares variables in Element for broadcasting with numpy functions placing them
        in a np.array with the required dimensions and in the required broadcasting
        dimension.
        
        Example: bpm.broadcasters(2, 3) -> muB = mu[None, None, :]
        
        With B indicating a broadcasting variable.
        """
        self._bAxis = axis #Saving the broadcasting axis
        variables = [attr for attr in self.__dict__ if not attr.startswith('_')] #Crea broadcasters per totes les variables igual
        for var in variables:
            setattr(self, var+"B", broadcast_vector( getattr(self, var) , axis, ndim))

class AnaORM:
    """Analitic Calculation of Orbit by AT
    Methods used for different calculations involving 

    Parameters
    ----------
    ring : at.lattice.lattice_object.Lattice
        lattice used
    direction : np.array
        "v" or "h", direction in which the calculations are performed
    ind      : dict
        Beam Position Monitors for the orbit response matrix
    ind_cor : np.array
        Correctors used for the ORM
    ind_quad : np.array
        Quadrupoles to take the derivatives
    ind_dip : np.array
        Indices for the dipoles in the matrix for dispersion calculations
    """
    def __init__(self, ring :at.lattice.lattice_object.Lattice, 
                 direction: str, 
                 ind: dict):
        self.ring     = ring
        self.mcf = get_mcf(ring)
        self.circumference = ring.circumference
        self.ind  = ind
        self.dir  = direction
        self.dir_ind = dir_dict[direction] #Index of the direction in at.get_optics in which the derivative is calculated
        self.all_optics = at.get_optics(ring, refpts=range(len(ring))) # get all the optics from the ring
        self.tune = self.all_optics[1]["tune"][self.dir_ind]
        self.sgn  = -(-1)**self.dir_ind #sign associated with that direction for quadrupoles

    def assign_optics(self): #Asigna l'òptica dels elements que hi ha en general a l'anell, però sempre es pot fer a mào
        self.bpm = Elements(self.ring, self.all_optics ,self.ind["bpm"], self.dir_ind, self.sgn)    
        self.cor = Elements(self.ring, self.all_optics ,self.ind["cor"][self.dir], self.dir_ind, self.sgn)
        self.quad= Elements(self.ring, self.all_optics ,self.ind["quad"], self.dir_ind,self.sgn)
        self.dip = Elements(self.ring, self.all_optics ,self.ind["dip"], self.dir_ind, self.sgn)
        if "CFD" in self.ind: self.CFD= Elements(self.ring, self.all_optics ,self.ind["CFD"], self.dir_ind,  self.sgn)
        
    def add_element(self,name, ind, direction): 
        #Important; this new element can be for a calculation set in a different direction. This will change the optics functions that
        #Are stored with it and the sign of its calculations, aleshores s'utilitzen aquestes direccions
        """ADDS a new element to the calculation method to perform different calculations"""
        setattr(self,name,Elements(self.ring, self.all_optics ,ind, dir_dict[direction], -(-1)**(dir_dict[direction])))
    #########################################################################
    # To use the functions here, broadcastig has to be defined on elements
    # By typing Element.broadcasters(axis, dim)
    #########################################################################
    
    def Cabn(self, Ea: Elements, Eb: Elements, n: int):
        return np.cos(n*np.abs(Ea.muB-Eb.muB)-n*np.pi*self.tune)

    def Sabn(self, Ea: Elements, Eb: Elements, n: int):
        """IMPORTANT, cal sumar una quantitat molt petita a la funció signe perqué
        comporti de la manera definida a l'informe!
        """
        return np.sign(Ea.muB-Eb.muB-0.0000001)*np.sin(n*np.abs(Ea.muB-Eb.muB)-n*np.pi*self.tune)
    
    def Rab_thin(self, Ea : Elements, Eb: Elements):
        return np.sqrt(Ea.betaB*Eb.betaB)/(2*np.sin(np.pi*self.tune))*self.Cabn(Ea, Eb, 1)
    
    def Tab_thin(self, Ea : Elements, Eb: Elements):
        return np.sqrt(Ea.betaB*Eb.betaB)/(2*np.sin(np.pi*self.tune))*self.Sabn(Ea, Eb, 1)
    
    def Rab_thick2_(self, Ea : Elements, Eb: Elements):
        """ Returns the ORM with thick correctors WITHOUT quadrupolar moment inside
        """
        Cij1 = self.Cabn(Ea, Eb, 1)
        Sij1 = self.Sabn(Ea, Eb, 1)
        
        Ijc = self.Ikc1_(Eb)
        Ijs = self.Iks1_(Eb)
        
        return np.real(np.sqrt(Ea.betaB*Eb.betaB)/(2*np.sin(np.pi*self.tune))*(Ijc*Cij1+Ijs*Sij1))
    
    def Rab_thick2_disp(self, Ea : Elements, Eb: Elements):
        """ Returns the dispersion term of the ORM with thick correctors WITHOUT quadrupolar moment inside and no dipole component
        """
        return np.real(-Ea.dispersionB*(Eb.dispersionB+1/2*Eb.dispersionpB*Eb.LengthB))/(self.mcf*self.circumference)
                 
    def Ik0(self, Ek: Elements):
        """Integral for element WITH quadrupolar moment """
        return (Ek.betaB+Ek.gammaB/Ek.KB)*Ek.LengthB/2+ (Ek.betaB-Ek.gammaB/Ek.KB)*(np.sin(2*np.sqrt(Ek.KB)*Ek.LengthB))/(4*np.sqrt(Ek.KB))+Ek.alphaB/(2*Ek.KB)*(np.cos(2*np.sqrt(Ek.KB)*Ek.LengthB)-1)

    def Ikc1_(self, Ek: Elements):
        """Integral for element WITHOUT quadrupolar moment """
        return 1 -Ek.alphaB*Ek.LengthB/(2*Ek.betaB)
        
    def Iks1_(self, Ek: Elements):
        """Integral for element WITHOUT quadrupolar moment """
        return Ek.LengthB/(2*Ek.betaB)


    def Ikc1(self, Ek: Elements):
        """Integral for element WITH quadrupolar moment """
        return np.sqrt(Ek.betaB/Ek.KB)*np.sin(Ek.LengthB*np.sqrt(Ek.KB))+(Ek.alphaB*(np.cos(Ek.LengthB*np.sqrt(Ek.KB))-1))/(Ek.KB*np.sqrt(Ek.betaB))
        
    def Iks1(self, Ek: Elements):
        """Integral for element WITH quadrupolar moment """
        return -(np.cos(Ek.LengthB*np.sqrt(Ek.KB))-1)/(Ek.KB*np.sqrt(Ek.betaB))  
    
    def Iks2(self, Ek: Elements): 
        """Integral term with quadrupole moment indide"""
        return 1/(2*Ek.KB)*(1-np.cos(2*np.sqrt(Ek.KB)*Ek.LengthB)+Ek.alphaB/(Ek.betaB)*(np.sin(2*np.sqrt(Ek.KB)*Ek.LengthB)/np.sqrt(Ek.KB)-2*Ek.LengthB))
        
    def Ikc2(self, Ek: Elements):
        """Integral term with quadrupole moment indide"""
        return self.Ik0(Ek) + (np.sin(2*np.sqrt(Ek.KB)*Ek.LengthB)/(2*np.sqrt(Ek.KB))-Ek.LengthB)/(Ek.KB*Ek.betaB)
    
    def Ijc1_q_L(self, Ej: Elements):
        """Integral term for elements without quadrupole moment inside divided by the length of the element"""
        return 1-(Ej.alphaB*Ej.LengthB)/(2*Ej.betaB)
        
    def Ijs1_q_L(self, Ej: Elements):
        """Integral term for elements without quadrupole moment inside divided by the length of the elmeent"""
        return Ej.LengthB/(2*Ej.betaB)
    
    def Ijc1_L(self, Ej: Elements):
        """Integral term for elements WITH quadrupole moment inside divided by the length of the element"""
        return np.real(np.sin(np.sqrt(Ej.KB)*Ej.LengthB)/(Ej.LengthB*np.sqrt(Ej.KB)) + Ej.alphaB*(np.cos(np.sqrt(Ej.KB)*Ej.Length)-1)/(Ej.KB*Ej.betaB*Ej.LengthB))
        
    def Ijs1_L(self, Ej: Elements):
        """Integral term for elements WITH quadrupole moment inside divided by the length of the elmeent"""
        return -(np.cos(np.sqrt(Ej.KB)*Ej.LengthB)-1)/(Ej.KB*Ej.betaB*Ej.LengthB)
    

    def dIjs1_dqk_(self, Ej: Elements, Ek: Elements):
        """Returns the dispersion derivative in position j with respect to thin quadrupoles k (in the horizontal direction)
        """
        #TODO: return extra term if the quadrupole index corresponds as well to a bending magnet
        
        #return case if the quadrupole has no bending component (basically for optics inside of )
        return 0
    def dIjc1_dqk_(self, Ej: Elements, Ek: Elements):
        """Returns the dispersion derivative in position j with respect to thin quadrupoles k
        """
        
    def dIjc1_dqk(self, Ej: Elements, Ek: Elements):
        """Returns the dispersion derivatative in j with respect to thik quadrupoles k 
        """
        
        
    def dRij_dqk_thin(self, Ei : Elements, Ej : Elements, Ek : Elements):
        """Considers all elements as thin, results can be greatlly improved by passing average
        optics computed with the average method instead of the entrance optics, but for thick
        elements one has to use the thick version of the formula.
        """
        Cij1 = self.Cabn(Ei, Ej, 1)
        Cik2 = self.Cabn(Ei, Ek, 2)
        Cjk2 = self.Cabn(Ej, Ek, 2)
        Sij1 = self.Sabn(Ei, Ej, 1)
        Sik2 = self.Sabn(Ei, Ek, 2)
        Sjk2 = self.Sabn(Ej, Ek, 2)
            
        cosTerm = Cij1 * ( Cik2 + Cjk2 + 2* np.cos(np.pi * self.tune)**2)
        sinTerm = Sij1 * ( Sik2 - Sjk2 + np.sin( 2*np.pi*self.tune)*(2*np.heaviside(Ei.muB-Ek.muB, 0)
                    -2*np.heaviside(Ej.muB-Ek.muB, 0)-np.sign(Ei.muB-Ej.muB)))
        
        ana_dORM_dq = self.sgn * (
        np.sqrt(Ei.betaB * Ej.betaB) * Ek.betaB * Ek.LengthB
        / (8 * np.sin(np.pi * self.tune)* np.sin(2 * np.pi * self.tune)) 
        * (cosTerm + sinTerm))
        return np.real(ana_dORM_dq)
    
    def dTij_dqk_thin(self, Ei : Elements, Ej : Elements, Ek : Elements):
        """Standalone method to compute it"""
        #TODO: write down
        
        return 0
    def dRij_dqk_thick3(self, Ei : Elements, Ej : Elements, Ek : Elements):
        """Computes the dRij_dqk asssuming only thick quadrupoles"""
        Cij1 = self.Cabn(Ei, Ej, 1)
        Cik2 = self.Cabn(Ei, Ek, 2)
        Cjk2 = self.Cabn(Ej, Ek, 2)
        Sij1 = self.Sabn(Ei, Ej, 1)
        Sik2 = self.Sabn(Ei, Ek, 2)
        Sjk2 = self.Sabn(Ej, Ek, 2)
        
        #Terms for the thick element formula        
        Ik0  = self.Ik0(Ek)
        Iks2 = self.Iks2(Ek)
        Ikc2 = self.Ikc2(Ek)
        SSik2= Ikc2*Sik2-Iks2*Cik2
        SSjk2= Ikc2*Sjk2-Iks2*Cjk2
        CCik2= Ikc2*Cik2+Iks2*Sik2
        CCjk2= Ikc2*Cjk2+Iks2*Sjk2
        
        cosTerm = Cij1 * ( CCik2 + CCjk2 + 2*Ik0 *np.cos(np.pi * self.tune)**2)
        sinTerm = Sij1 * ( SSik2 - SSjk2 + Ik0*np.sin(2*np.pi*self.tune)*(2*np.heaviside(Ei.muB-Ek.muB, 0)
            -2*np.heaviside(Ej.muB-Ek.muB, 0)-np.sign(Ei.muB-Ej.muB)))
        ana_dORM_dq = self.sgn * ( np.sqrt(Ei.betaB * Ej.betaB) 
         / (8 * np.sin(np.pi * self.tune)* np.sin(2 * np.pi * self.tune)) 
         * (cosTerm + sinTerm))
        return np.real(ana_dORM_dq) #Per assegurar que retorni un real bé
    
    
    def dRij_dfringes(self, Ei : Elements, Ej : Elements, Ek : Elements):
        """Calculates the entrance and exit dipole fringe field contribution to the ORM assuming thin correctors
        the input elements MUST HAVE NO ENTRANCE CORRECTIONS APPLIED BEFORE
        """
        # We first calculate the effective strength of the thin quadrupole, which is given by:
        # B * entrance angle ; where B = theta / length.
        #TODO: ESCRIURE BÉ, ÉS UN TERME IMPORTANT!
        Ek0 = copy.deepcopy(Ek) 
        
        Cij1 = self.Cabn(Ei, Ej, 1)
        Cik2 = self.Cabn(Ei, Ek, 2)
        Cjk2 = self.Cabn(Ej, Ek, 2)
        Sij1 = self.Sabn(Ei, Ej, 1)
        Sik2 = self.Sabn(Ei, Ek, 2)
        Sjk2 = self.Sabn(Ej, Ek, 2)
        
        
        cosTerm = Cij1 * ( Cik2 + Cjk2 + 2* np.cos(np.pi * self.tune)**2)
        sinTerm = Sij1 * ( Sik2 - Sjk2 + np.sin( 2*np.pi*self.tune)*(2*np.heaviside(Ei.muB-Ek.muB, 0)
                    -2*np.heaviside(Ej.muB-Ek.muB, 0)-np.sign(Ei.muB-Ej.muB)))
        
        ana_dORM_dq = self.sgn * (
        np.sqrt(Ei.betaB * Ej.betaB) * Ek.betaB * Ek.LengthB
        / (8 * np.sin(np.pi * self.tune)* np.sin(2 * np.pi * self.tune)) 
        * (cosTerm + sinTerm))
        return np.real(ana_dORM_dq)
    
    def dRij_dqk_thick23(self, Ei : Elements, Ej : Elements, Ek : Elements):
        """Computes the dRij_dqk asssuming thick correctors without quadrupolar component and thick quadrupoles"""
        Cij1 = self.Cabn(Ei, Ej, 1)
        Cik2 = self.Cabn(Ei, Ek, 2)
        Cjk2 = self.Cabn(Ej, Ek, 2)
        Sij1 = self.Sabn(Ei, Ej, 1)
        Sik2 = self.Sabn(Ei, Ek, 2)
        Sjk2 = self.Sabn(Ej, Ek, 2)
        
        #Terms for the thick quadrupole formula      
        
        Ik0  = self.Ik0(Ek)
        Iks2 = self.Iks2(Ek)
        Ikc2 = self.Ikc2(Ek)
        SSik2= Ikc2*Sik2-Iks2*Cik2
        SSjk2= Ikc2*Sjk2-Iks2*Cjk2
        CCik2= Ikc2*Cik2+Iks2*Sik2
        CCjk2= Ikc2*Cjk2+Iks2*Sjk2

        #Terms for thick correctors without quadrupole moment inside of them
        Ijc1_L = self.Ijc1_q_L(Ej)
        Ijs1_L = self.Ijs1_q_L(Ej)
        
        dRij_terms =  (Cij1 * ( CCik2 + CCjk2 + 2*Ik0 *np.cos(np.pi * self.tune)**2) + 
                       Sij1 * ( SSik2 - SSjk2 + Ik0*np.sin(2*np.pi*self.tune)*(2*np.heaviside(Ei.muB-Ek.muB, 0)
                           -2*np.heaviside(Ej.muB-Ek.muB, 0)-np.sign(Ei.muB-Ej.muB)))) 
        dTij_terms = (Sij1 * (CCik2 - CCjk2 + 2*Ik0 * np.cos(np.pi * self.tune)**2) + 
                      Cij1 * (-SSjk2 - SSik2 + Ik0 * np.sin(2*np.pi * self.tune) * (-2*np.heaviside(Ei.muB-Ek.muB, 0) 
                           + 2*np.heaviside(Ej.muB-Ek.muB, 0) + np.sign(Ei.muB-Ej.muB))))
        ana_dORM_dq = self.sgn * ( np.sqrt(Ei.betaB * Ej.betaB) 
         / (8 * np.sin(np.pi * self.tune)* np.sin(2 * np.pi * self.tune)) 
         * (Ijc1_L * dRij_terms + Ijs1_L * dTij_terms))
        
        return np.real(ana_dORM_dq) #Per assegurar que retorni un real bé
    
    def ni_sum(self, Ei:Elements, Ej: Elements):
        """ Calculates the dispersion in Ei generated by dipoles Ej in the ring
        """
        Ijc1 = self.Ijc1_L(Ej) 
        Ijs1 = self.Ijs1_L(Ej) 
        Rij_t = self.Rab_thin(Ei, Ej)
        Tij_t = self.Tab_thin(Ei, Ej)
        return np.sum(Ej.BendB*(Ijc1*Rij_t+Ijs1*Tij_t), axis = Ej._bAxis)
        
    def dni_dqk_sum(self, Ei : Elements, Ej: Elements, Ek: Elements):
        """ Derivative of the dispersions in Ei with respect to given quadrupole
        strengths Ek considering Ej dipoles in the ring... VALIDATED!
        """
    
        Cij1 = self.Cabn(Ei, Ej, 1)
        Cik2 = self.Cabn(Ei, Ek, 2)
        Cjk2 = self.Cabn(Ej, Ek, 2)
        Sij1 = self.Sabn(Ei, Ej, 1)
        Sik2 = self.Sabn(Ei, Ek, 2)
        Sjk2 = self.Sabn(Ej, Ek, 2)
        
        #Terms for the thick quadrupole formula      
        
        Ik0  = self.Ik0(Ek)
        Iks2 = self.Iks2(Ek)
        Ikc2 = self.Ikc2(Ek)
        SSik2= Ikc2*Sik2-Iks2*Cik2
        SSjk2= Ikc2*Sjk2-Iks2*Cjk2
        CCik2= Ikc2*Cik2+Iks2*Sik2
        CCjk2= Ikc2*Cjk2+Iks2*Sjk2

        #Terms for the dipoles, if they are CFD consider it as well through KB
        if (hasattr(Ej, "KB")):
            print("CFD detected")
            Ijc1 = self.Ijc1_L(Ej)
            Ijs1 = self.Ijs1_L(Ej)
        else:
            Ijc1 = self.Ijc1_q_L(Ej)
            Ijs1 = self.Ijs1_q_L(Ej)        

        
        dRij_terms =  (Cij1 * ( CCik2 + CCjk2 + 2*Ik0 *np.cos(np.pi * self.tune)**2) + 
                       Sij1 * ( SSik2 - SSjk2 + Ik0*np.sin(2*np.pi*self.tune)*(2*np.heaviside(Ei.muB-Ek.muB, 0)
                           -2*np.heaviside(Ej.muB-Ek.muB, 0)-np.sign(Ei.muB-Ej.muB)))) 
        dTij_terms = (Sij1 * ( CCik2 - CCjk2 + 2*Ik0 *np.cos(np.pi * self.tune)**2) + 
                      Cij1 * ( -SSjk2 - SSik2 + Ik0*np.sin(2*np.pi*self.tune)*(-2*np.heaviside(Ei.muB-Ek.muB, 0)
                           +2*np.heaviside(Ej.muB-Ek.muB, 0)+np.sign(Ei.muB-Ej.muB)))) 
        dni_dqk =  (np.abs(Ej.BendB)* np.sqrt(Ei.betaB * Ej.betaB) 
         / (8 * np.sin(np.pi * self.tune)* np.sin(2 * np.pi * self.tune)) 
         * (Ijc1 * dRij_terms + Ijs1 * dTij_terms))
        return np.sum(dni_dqk, axis = Ej._bAxis)
        

    def dRij_dqk_thick23_disp(self, Ei : Elements, Ej : Elements, Ek : Elements, El : Elements):
        """Computes the dRij_dqk dispersion term, which is relevant in the HORIZONTAL
        transverse dimension, neglecting the derivative with respect to the mcf
        Ei: BPMs,  Ej: correctors, Ek: quadrupoles, El: dipoles
        THIS formula only applies when quadrupoles (Ek) are not also bending magnets.
        """
        #TODO: No funciona x ALBA2
        dni_dqk = self.dni_dqk_sum(Ei, El, Ek)
        dnj_dqk = self.dni_dqk_sum(Ej, El, Ek)
        #Importantíssim contraure abans de juntar els tensors, sino la cosa peta bastant
        dispi = np.sum(Ei.dispersionB, axis = El._bAxis)
        dispj = np.sum((Ej.dispersionpB*Ej.LengthB/2+Ej.dispersionB) , axis = El._bAxis)
        
        #We can use the dispersions calculated before
        ana_dORM_dq_disp = (dni_dqk* dispj + dnj_dqk*dispi)/(self.mcf*self.circumference)
        
        return np.real(ana_dORM_dq_disp) #Per assegurar que retorni un real bé!
    
    def dxldqk(self, Ei: Elements, Ej: Elements, Ek: Elements, El: Elements):
        """
        Orbit displacement in sextupoles
        """
        #TODO: Importantísssssssima també!
    
        
    def dRij_dEnergy(self, Ei: Elements, Ej: Elements, Ek: Elements):
        """
        Calculates the response matrix to energy numerically, There is quite some error
        in the calculation as we lack quadrupoles, and other terms.
        
        #TODO: sembla que fins i tot en el cas sense sextupols fins i tot aquest mètode no dona bé.
        """
        # 1. Get the sensitivity tensor (Ni, Nj, Nk)
        dRij_dqk = self.dRij_dqk_thick23(Ei, Ej, Ek)
         
        return np.real(np.sum(dRij_dqk * (-Ek.KB), axis=0))
     
    def dRij_dCFD_energy(self, Ei: Elements, Ej: Elements, Ek: Elements):
        """
        Uses: and calculates the derivative of the energy with respect to a change in a given CFD
        Ei: bpms in horizontal
        Ej: corectors in horizontal
        Ek: CFD in horizontal
        dRijdEnergy: in axis 0 and 1
        """
        
        Ei.broadcasters(0, 3)   #n
        Ej.broadcasters(1, 3)   #m
        Ek.broadcasters(2, 3)   #k

        # 2. Extract Optics
        eta_n = Ei.dispersion        # (176,)
        eta_m = Ej.dispersion        # (176,)
        if not hasattr(Ek, 'avDispersion'):
            Ek.average()
        eta_k = Ek.avDispersion      # (208,)

        # 3. Compute 2D Matrices
        Rnm = self.Rab_thick2_(Ei, Ej)
        Rnk = self.Rab_thick2_(Ei, Ek)
        
        
        
        # 4. Energy Sensitivity Formula Logic
        R_inv = np.linalg.pinv(np.squeeze(Rnm))[:, None, :] 
        
        
        num = 1#np.sum(eta_m*R_inv*Rnk, axis = (0,1))
        denom = 1#np.sum(eta_m)

        # 5. Build d_delta/d_qk (BPM x CFD) -> (176 x 208)
        term1 = eta_k / (self.mcf * self.circumference)
        term2 = (num / denom)
        
        # Sensitivity d_delta/dqk (176, 208)
        d_delta_dqk = (term1) * (Ek.Bend / Ek.K)

        return d_delta_dqk
    
    
    def Rij_disp_term(self, Ei : Elements, Ej : Elements, Ed : Elements):
        """Calculates the dispersion caused at the entrance of the ith element
        caused by the El dipoles 
        #TODO: sure?
        """
        Ilc1 = self.Ikc1(Ed)
        Ils1 = self.Iks1(Ed)
        
        Cil1 = self.Cabn(Ei, Ed, 1)
        Sil1 = self.Sabn(Ei, Ed, 1)
        
        return np.sum(np.sqrt(Ei.betaB)/(2*np.sin(np.pi*self.tune))*Ed.BendB/Ed.LengthB * (Ilc1*Cil1+Ils1*Sil1), axis = Ed._bAxis)
    

    def aMCF(self, Em: Elements,En: Elements):
        """Calculates the mcf for a ring due to dipoles
        Diples in the first and second component analytically
        """
        Imc1 = self.Ikc1(Em)
        Ims1 = self.Iks1(Em)
        Inc1 = self.Ikc1(En)
        Ins1 = self.Iks1(En)        
        
        Cmn = self.Cabn(Em, En, 1)
        Smn = self.Sabn(Em, En, 1)
        
        return np.real(np.sum(1/(2*np.sin(np.pi*self.tune)) * np.sum(Em.BendB/Em.LengthB*En.BendB/En.LengthB* ( Cmn*(Imc1*Inc1-Ims1*Ins1) + Smn*(Ims1*Inc1+Imc1*Ins1)), axis =1), axis =0))
    
    def dMCFdq(self, Ek: Elements):
        """Derivative of the MCF with respect to one quadrupole strength
        """
        
        return -1/self.circumference*((Ek.dispersionB**2 + Ek.dispersionpB**2/Ek.KB)*Ek.LengthB/2 + (2*Ek.dispersionB*Ek.dispersionpB/(np.sqrt(Ek.KB))*np.sin(Ek.LengthB*np.sqrt(Ek.KB))**2+ (Ek.dispersionB**2-Ek.dispersionpB**2/Ek.KB)*np.sin(Ek.LengthB*np.sqrt(Ek.KB))*np.cos(Ek.LengthB*np.sqrt(Ek.KB)))/(2*np.sqrt(Ek.KB)))


    def nextdisp(self, Ei: Elements):
        """  Dispersion of the next element using the dispersion at the entrance of a quadrupole"""
        if hasattr(Ei, "Bend"):
            #En aquest cas considerem que la dispersió s'està originant també a l'interior de l'element que que 
            #L'angle d'entrada causa un "quadrupol prim" que canvia la derivada de la dispersió
            #TODO: pensar en els angles d'entrada i com la derivada de la dispersió canvia d'alguna manera amb un kick! 
            kickedDP = Ei.dispersionpB #-Ei.dispersionB*np.tan(Ei.EntranceAngleB)*(Ei.BendB/Ei.LengthB) #Focusing strength of entrance dipole
            return Ei.BendB/(Ei.LengthB*Ei.KB)+(Ei.dispersionB-Ei.BendB/(Ei.LengthB*Ei.KB))*np.cos(Ei.LengthB*np.sqrt(Ei.KB))+kickedDP*np.sin(Ei.LengthB*np.sqrt(Ei.KB))/np.sqrt(Ei.KB)
        return Ei.dispersionB*np.cos(Ei.LengthB*np.sqrt(Ei.KB))+Ei.dispersionpB*np.sin(Ei.LengthB*np.sqrt(Ei.KB))/np.sqrt(Ei.KB)

        
