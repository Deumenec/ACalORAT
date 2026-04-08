import numpy as np
from ..Elements import Elements
from .BaseIntegrals import BaseIntegrals
from .dORM_dq import dORM_dq

class dORM_dh(BaseIntegrals):
    
    """
    Formulas for the Jacobian of the Orbit Response Matrix with respect to 
    changin only the B[0] component in a magnet, either a CFD or a dipole.
    """
    
    def dRij_dk_energy_term(self, Ei: Elements, Ej: Elements, Ek: Elements, dRij_dEnergy, dEnergy):
        """

        Uses the Jacobian of the ORM with respect to energy and the change in energy due 
        to a change in a given element to calculate the contribution of it to the Jacobian of the ORM.

        Ei: bpms in horizontal          0 

        Ej: corectors in horizontal     1

        Ek: CFD in horizontal           2
        """
        
        #TODO: Calculate analytically the dRij_dEnergy

        term = np.real(dEnergy[None, None,:] * dRij_dEnergy[: , :, None])

        return term
    
    def dRij_dbend_thick23_disp(self, Ei : Elements, Ej : Elements, Ek : Elements):
        """
        Computes the dRij_dbend dispersion term, which is relevant in the HORIZONTAL
        transverse dimension when changing the dipole component of a CFD. It is scaled for
        a proportional change to the quadrupole!
        
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
    
    def Rij_disp_term(self, Ei : Elements, Ej : Elements, Ed : Elements):
        """
        ??????? I don't know what this does!!!
        
        """
        Ilc1 = self.Ikc1(Ed)
        Ils1 = self.Iks1(Ed)
        
        Cil1 = self.Cabn(Ei, Ed, 1)
        Sil1 = self.Sabn(Ei, Ed, 1)
        
        return np.sum(np.sqrt(Ei.betaB)/(2*np.sin(np.pi*self.tune))*Ed.BendB/Ed.LengthB * (Ilc1*Cil1+Ils1*Sil1), axis = Ed._bAxis)
    
    def dRij_dk_fringe_term(self, Ei:Elements, Ej:Elements, Ek:Elements):
        """
        Calculates the change in the ORM due to how changing a dipole causes a quadrupole to change in the fringes.
        By using the optics at the entrance of the dipole and the optics after the entrance.
        """
        #TODO: Correcting optics to take the right ones for the finge quadrupoles.
        ratio = Ek.Bend/(Ek.LengthB*Ek.KB)
        
        
        #Building the optics at the exit of the elements
        Ek_exit = Elements(self.ring, self.all_optics, Ek._ind+1, self.dir_ind, self.sgn)
        #Broadcasting in the same dimensions as before
        Ek_exit.broadcasters(Ek._bAxis, Ek._ndim)
        
        strength_entrance   = np.tan(Ek.EntranceAngleB)*ratio
        dRijdk_entrance     = self.dRij_dqk_thick2(Ei, Ej, Ek)
        
        strenght_exit       = np.tan(Ek.ExitAngleB)*ratio
        dRijdk_exit         = self.dRij_dqk_thick2(Ei, Ej, Ek_exit)
        
        #Now we consider the proportional change in quadrupole strength!
        return np.real(strength_entrance * dRijdk_entrance + strenght_exit * dRijdk_exit)

    def dRi_dk_sex_term(self, Ei: Elements, Ej: Elements, Ek: Elements, El: Elements, i, j, k):
        """
        Ei: bpms

        Ej: correctors

        Ek: CFDs

        El: sextupoles
        
        Calculates the term on the Jacobian of the ORM with 
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
        
        return np.squeeze(np.real(ana_dORM_dq)) #Removing the auxiliary sextupole dimension
    
    