#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 19:29:25 2025

@author: Domènec Huerta

Class with defined functions to compute ORMs and their derivatives
"""
import numpy as np
import at
import copy
from . import numerical
from .Elements import Elements

from .physics.ORM import ORM
from .physics.dORM_dquad    import dORM_dquad
from .physics.DispersionQuad import DispersionQuad


#dir_dict = {"h": 0, "v": 1} #For further reference, see the structure of the at.get_optics output

def extract_kicks(dkicks, ind_old, ind_new):
    """
    Used to get the kicks inside of sexupole correctors.
    
    Takes a Jacobian of kicks in the second dimension in a given set of elements and extracts the
    kicks in another set of elements by matching repeated elements, suposing the rest of kicks are zero!
    """
    dkicks = np.transpose(dkicks)
    new_dkicks = np.zeros(np.shape(dkicks))
    for i, eli in enumerate(ind_old):
        for j , elj in enumerate(ind_new):
            if elj > eli:
                break
            if elj == eli:
                new_dkicks[j] = dkicks[i]
                
    return np.transpose(new_dkicks)

  
class AnaORM(ORM, dORM_dquad, DispersionQuad):
    """Analitic Calculation of Orbit by AT
    Method used for all calculations on analytical calculation
    
    For each element type, it contains an instance of the Elements Class with
    all relevant information.
    #TODO: this is the most important class so explain well!
    Parameters
    ----------
    ring : at.lattice.lattice_object.Lattice
    
        lattice used
        
    direction : np.array
    
        "v" or "h", direction in which the calculations are performed
        
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
            DESCRIPTION.
        direction : str
            DESCRIPTION.
        ind : dict
            DESCRIPTION.
        old_optics : TYPE, optional
            DESCRIPTION. The default is False.

        """
        self.ring     = ring
        """pepsicolor"""
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
        #Important; this new element can be for a calculation set in a different direction. This will change the optics functions that
        #Are stored with it and the sign of its calculations, aleshores s'utilitzen aquestes direccions
        """ADDS a new element to the calculation method to perform different calculations"""
        setattr(self,name,Elements(self.ring, self.all_optics ,ind, {"h": 0, "v": 1}[direction], -(-1)**({"h": 0, "v": 1}[direction])))
    
    

    """
    #########################################################################
    ### Miscelaneous Closed Orbit Formulas
    #########################################################################
    """    
    def dRij_dEnergy(self, Ei: Elements, Ej: Elements, Ek1: Elements, Ek2: Elements):
        """
        Calculates the response matrix to energy numerically.
        Ek1: Pure quadrupoles
        Ek2: CFD 
        #TODO: sembla que fins i tot en el cas sense sextupols fins i tot aquest mètode no dona bé.
        """
        # 1. Get the sensitivity tensor (Ni, Nj, Nk)
        
        dRij_de = (np.sum(self.dRij_dqk_thick23(Ei, Ej, Ek1)*Ek1.KB , axis = 0)
                  + np.sum(self.dRij_dqk_thick23_master(Ei, Ej, Ek2)*Ek2.KB ,axis = 0))
         
        return np.real(dRij_de)
     
    def ddip_denergy(self, Ek: Elements):
        """
        Returns the change in energy when only changin one single CFD with the 
        change being relative to the change in quadrupole strength! 
        
        WITHOUT ACTVE CORRECTIONS
        
        Ek: CFD in horizontal dimension!
        """

        geometry = (Ek.Bend / (Ek.K))
        
        if not hasattr(Ek, 'avDispersion'):
            Ek.average()
        eta_k = np.squeeze(Ek.avDispersion)      
        
        term1 = (eta_k * geometry) / (self.mcf * self.circumference)
        
        return term1

    
    def dCFD_denergy(self, Ei: Elements, Ej: Elements, Ek: Elements):
        """
        Uses and calculates the derivative of the ENERGY with respect to a change in a given CFD WITH ACTIVE corrections!
        

        Ei: bpms in horizontal          0 
        Ej: corectors in horizontal     1
        Ek: CFD in horizontal           2
        """

        #TODO: Es pot reescriure aquest codi bé amb tensors pq sigui agnòstic a l'orde de les dimensions d'entrada.
        
        # 1. Extract Optics averages
        eta_n = np.squeeze(Ei.dispersion)        
        if not hasattr(Ej, 'avDispersion'):
            Ej.average()
        eta_m = np.squeeze(Ej.avDispersion)  
        if not hasattr(Ek, 'avDispersion'):
            Ek.average()
        eta_k = np.squeeze(Ek.avDispersion)      
        


        Rnm = np.squeeze(self.Rab_thick2_(Ei, Ej)   + self.Rab_thick2_disp(Ei, Ej)) # Shape: (N_BPM, M_COR)
        Rnk = np.squeeze(self.Rab_thick2_K(Ei, Ek)  + self.Rab_thick2_disp_K(Ei, Ek)) # Shape: (N_BPM, K_CFD)
        geometry = (Ek.Bend / (Ek.K))
        R_inv = np.linalg.pinv(Rnm)
        
        hcm0 = (R_inv @ Rnk) * geometry 
        
        term1 = (eta_k * geometry) / (self.mcf * self.circumference)
        
        term2 = (eta_m @ hcm0) / (eta_m @ R_inv @ eta_n)
        
        # Total shift 
        d_delta_dqk = term1 + term2
        
        return d_delta_dqk
    """
    #########################################################################
    ### Pure Dipole ORM Jacobian Formulas
    #########################################################################
    """
    def dRij_dk_energy_term(self, Ei: Elements, Ej: Elements, Ek: Elements, dRij_dEnergy, dEnergy):
        """
        Uses derivative with respect to energy and energy response matrix to calculate the term corresponding
        to the energy change in thick energy response

        Ei: bpms in horizontal          0 
        Ej: corectors in horizontal     1
        Ek: CFD in horizontal           2
        """
        
        term =     np.real(dEnergy[None, None,:] * dRij_dEnergy[: , :, None])

        return term
    
    def dRij_dbend_thick23_disp(self, Ei : Elements, Ej : Elements, Ek : Elements):
        """
        Computes the dRij_dbend dispersion term, which is relevant in the HORIZONTAL
        transverse dimension when changing the dipole component of a CFD. It is scaled for
        a proportional change to the quadrupole!
        
        #TODO: THIS METHOD DOES NOT WORK!
        Ei: BPMs,         1
        Ej: correctors,   2
        Ek: CFDs,         0
        """
        
        geometry = Ek.BendB / (Ek.KB)
        
        dni_dbend = Ek.LengthB * self.Rab_thick2_K(Ei, Ek) 
        dnj_dbend = Ek.LengthB * self.Rab_thick2_K(Ej, Ek)  
        
        Ej2 = Elements(self.ring, self.all_optics, Ej._ind - 1, self.dir_ind, self.sgn)
        Ej2.broadcasters(Ej._bAxis, Ej._ndim)
        
        dnj_dbend_prev = Ek.LengthB * self.Rab_thick2_K(Ej2, Ek)  
        
        ddnj_ds_dbend = (dnj_dbend - dnj_dbend_prev) / Ej2.LengthB
        
        if not hasattr(Ej, 'avDispersion'):
            Ej.average()

        ana_dORM_dbend_disp = (dni_dbend * Ej.avDispersionB + (dnj_dbend + ddnj_ds_dbend * Ej.LengthB / 2) * Ei.dispersionB ) / (self.mcf * self.circumference)
        

        
        return np.real(ana_dORM_dbend_disp * geometry)
        
        
    def dRij_dk_fringe(self, Ei:Elements, Ej:Elements, Ek:Elements):
        """
        Calculates the change in the ORM due to how changing a dipole causes a quadrupole to change in the fringes.
        By using the optics at the entrance of the dipole and the optics after the entrance.
        """
        ratio = Ek.Bend/(Ek.LengthB*Ek.KB)
        
        #Building the optics at the exit of the elements
        Ek_exit = Elements(self.ring, self.all_optics, Ek._ind+1, self.dir_ind, self.sgn)
        #Broadcasting in the same dimensions as before
        #TODO: CORRECT EXIT OPTICS BY USING ENTRANCE CORRECTION WITH OPOSITE SIGN.
        
        Ek_exit.broadcasters(Ek._bAxis, Ek._ndim)
        
        strength_entrance   = np.tan(Ek.EntranceAngleB)*ratio
        dRijdk_entrance     = self.dRij_dqk_thick2(Ei, Ej, Ek)
        
        #Here the sign is also positive!
        #SUUUPER DUPPER IMPORTANT!
        ##########################
        strenght_exit       = np.tan(Ek.ExitAngleB)*ratio
        dRijdk_exit         = self.dRij_dqk_thick2(Ei, Ej, Ek_exit)
        
        #Now we consider the proportional change in quadrupole strength!
        
        return np.real(strength_entrance * dRijdk_entrance + strenght_exit * dRijdk_exit)
        
    def dxldCFDk(self, Ei: Elements, Ej: Elements, Ek: Elements, El: Elements):
        """
        Orbit displacement in sextupoles ENTRANCE
            Class       Broadcast dim
        Ei: bpms        (0,4)    n
        Ej: cor         (1,4)    m
        Ek: CFD         (2,4)    k
        Es: sextupoles  (3,4)    l
        
        Returns:
            axis 0: CFD
            axis 1: sextupole
        """
    
        # 1. Extract Optics averages
        eta_n = np.squeeze(Ei.dispersion)        
        if not hasattr(Ej, 'avDispersion'):
            Ej.average()
        eta_m = np.squeeze(Ej.avDispersion)  
        if not hasattr(El, 'avDispersion'):
            El.average()
        eta_l = np.squeeze(El.dispersion)      
        
        
        Rnm = np.squeeze(self.Rab_thick2_(Ei, Ej)+ self.Rab_thick2_disp(Ei, Ej)) # Shape: (N_BPM, M_COR)
        Rnk = np.squeeze(self.Rab_thick2_K(Ei, Ek)+ self.Rab_thick2_disp_K(Ei, Ek)) # Shape: (N_BPM, K_CFD)
        Rlk = np.squeeze(self.Rab_thick2_K(El, Ek)+ self.Rab_thick2_disp_K(El, Ek)) # Shape: (L_sex, K_CFD)
        Rlk = Rlk.T
        Rlm = np.squeeze(self.Rab_thick2_(El, Ej) + self.Rab_thick2_disp(El, Ej) )
        Rlm = Rlm.T
        R_inv = np.linalg.pinv(Rnm)
        
        geometry = Ek.Bend / Ek.K
        
        term1 = Rlk
        
        term2 = Rlm@R_inv@Rnk
    
        num = (eta_l - Rlm@R_inv@eta_n)[:, None]
        d_e_non = ((eta_m@R_inv@Rnk)/(eta_m@R_inv@eta_n))[None, :]  
        
        term3 = num * d_e_non
        
        dxs = ((-term1 + term2 + term3) * geometry[None,:]).T
        
        return np.real(dxs)
    
    def dpxldCFDk(self, Ei: Elements, Ej: Elements, Ek: Elements, El: Elements):
        """
        Calculates the derivative of the orbit transverse derivative with 
        respect to CDF activation by SUPOSING THERE IS A DRIFT before the
        sextupole and using the numerical derivative inside of it
        
        Returns:
            axis 0: CFD
            axis 1: sextupole
        """
        
        El0 = Elements(self.ring, self.all_optics, El._ind-1, self.dir_ind, self.sgn)
        El0.broadcasters(El._bAxis, El._ndim)
        
        x1 = self.dxldCFDk(Ei, Ej, Ek, El)
        x0 = self.dxldCFDk(Ei, Ej, Ek, El0)
        
        return (x1-x0)/El0.Length[None, :]
    
    def average_dxsexdCFD(self, Ei: Elements, Ej: Elements, Ek: Elements, El: Elements, El0:Elements):
        """
        Orbit displacement in sextupoles AVERAGE
            Class       Broadcast dim
        Ei: bpms        (1,4)    n
        Ej: cor         (1,4)    m
        Ek: CFD         (2,4)    k
        El: sextupoles  (3,4)    l
        El0:drift sext  (3,4)    l
        
        Calculates the true average orbit displacement by sextupole by considering 
        """
        x = self.dxldCFDk(Ei, Ej, Ek, El)
        #Derivative calculated by considering the drift before!
        dx = (x-self.dxldCFDk(Ei, Ej, Ek, El0))/El0.Length
        
        
        
        return x + dx*El.Length
    
    
    
    def Rij_disp_term(self, Ei : Elements, Ej : Elements, Ed : Elements):
        """Calculates the dispersion caused at the entrance of the ith element
        caused by the El dipoles 
        """
        Ilc1 = self.Ikc1(Ed)
        Ils1 = self.Iks1(Ed)
        
        Cil1 = self.Cabn(Ei, Ed, 1)
        Sil1 = self.Sabn(Ei, Ed, 1)
        
        return np.sum(np.sqrt(Ei.betaB)/(2*np.sin(np.pi*self.tune))*Ed.BendB/Ed.LengthB * (Ilc1*Cil1+Ils1*Sil1), axis = Ed._bAxis)
    
    def dRi_dk_sex_term(self, Ei: Elements, Ej: Elements, Ek: Elements, El: Elements, i, j, k):
        """
        Ei: bpms
        Ej: correctors
        Ek: CFDs
        El: sextupoles
        
        Calculates the term of the jacobian of the ORM considering the effective quadrupole created in sextupoles due to closed orbit distortion
        """
        #General cosinus terms
        
        Cij1 = self.Cabn(Ei, Ej, 1)
        Sij1 = self.Sabn(Ei, Ej, 1)

        
        #Terms for thick sextupoles and mixing integrals
        Sil2 = self.Sabn(Ei, El, 2)
        Cil2 = self.Cabn(Ei, El, 2)
        Sjl2 = self.Sabn(Ej, El, 2)
        Cjl2 = self.Cabn(Ej, El, 2)
        Ilk0  = self.Ilk0(El, i, j, k)
        Ilkc2 = self.Ilkc2(El, i, j, k) 
        Ilks2 = self.Ilks2(El, i, j, k)
        
        SSilk2 = np.sum(Ilkc2*Sil2 - Ilks2*Cil2, axis = El._bAxis, keepdims=True)
        CCilk2 = np.sum(Ilkc2*Cil2 + Ilks2*Sil2, axis = El._bAxis, keepdims=True)
        DDilk2 = np.sum(Ilk0*np.heaviside(Ei.muB-El.muB, 0), axis = El._bAxis, keepdims=True)

        SSjlk2 = np.sum(Ilkc2*Sjl2 - Ilks2*Cjl2, axis = El._bAxis, keepdims=True)
        CCjlk2 = np.sum(Ilkc2*Cjl2 + Ilks2*Sjl2, axis = El._bAxis, keepdims=True)
        DDjlk2 = np.sum(Ilk0*np.heaviside(Ej.muB-El.muB, 0), axis = El._bAxis, keepdims=True)
        
        Iijk0 = np.sum(Ilk0, axis = El._bAxis, keepdims=True)
        
        #Terms for thick correctors without quadrupole moment inside of them
        Ijc1_L = self.Ikc1_(Ej)
        Ijs1_L = self.Iks1_(Ej)
    
        
        dRij_terms =  (Cij1 * ( CCilk2 + CCjlk2 + 2*Iijk0 *np.cos(np.pi * self.tune)**2) + 
                       Sij1 * ( SSilk2 - SSjlk2 
                     + np.sin(2*np.pi*self.tune)*(2*DDilk2 - 2*DDjlk2 - Iijk0*np.sign(Ei.muB-Ej.muB)))) 
        
        dTij_terms = (Sij1 * (CCilk2 - CCjlk2 + 2*Iijk0 * np.cos(np.pi * self.tune)**2) + 
                      Cij1 * (-SSjlk2 - SSilk2 
                     + np.sin(2*np.pi*self.tune)*(-2*DDilk2 + 2*DDjlk2 + Iijk0*np.sign(Ei.muB-Ej.muB)))) 

        
        ana_dORM_dq = self.sgn * ( np.sqrt(Ei.betaB * Ej.betaB) 
         / (8 * np.sin(np.pi * self.tune)* np.sin(2 * np.pi * self.tune)) 
         * (Ijc1_L * dRij_terms + Ijs1_L * dTij_terms))
        
        return np.squeeze( np.real(ana_dORM_dq) ) #Per assegurar que retorni un real bé
    
    def aMCF(self, Em: Elements,En: Elements):
        """Calculates the mcf for a ring due to dipoles analytically
        Diples in the first and second component 
        
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

        
