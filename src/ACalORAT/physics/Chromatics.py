import numpy as np
from ..Elements import Elements
from .BaseIntegrals import BaseIntegrals
from .dORM_dq import dORM_dq


class Chromatics(BaseIntegrals):
    """
    Formulas involving chromatic terms and closed orbit distortions.
    """
    
    def dRij_dEnergy_quad(self, Ei: Elements, Ej: Elements, Ek1: Elements, Ek2: Elements):
        """
        Calculates the response matrix to energy numerically.

        Ek1: Pure quadrupoles

        Ek2: CFD 
        """
        # 1. Get the sensitivity tensor (Ni, Nj, Nk)
        
        dRij_de = (np.sum(self.dRij_dqk_thick23(Ei, Ej, Ek1)*Ek1.KB , axis = 0)
                    + np.sum(self.dRij_dqk_thick23_master(Ei, Ej, Ek2)*Ek2.KB ,axis = 0))
            
        return np.real(dRij_de)
        
    def dRij_dEnergy_sex(self, Ei: Elements, Ej: Elements, Ek1: Elements, Ek2: Elements):
        """
        Calculates the response matrix to energy for sextupole correctors.

        Ek1: Pure sextupoles

        Ek2: CFD 
        """
        # 1. Get the sensitivity tensor (Ni, Nj, Nk)
        
        dRij_de = (np.sum(self.dRij_dqk_thick23(Ei, Ej, Ek1)*Ek1.KB , axis = 0)
                + np.sum(self.dRij_dqk_thick23_master(Ei, Ej, Ek2)*Ek2.KB ,axis = 0))
        
        return np.real(dRij_de)

    def ddip_denergy(self, Ek: Elements):
        """
        Returns the change in energy when only changin the bend component of
        a dipole (or CFD).
        It doesn't consider the effect of the feedback system, so it is only 
        the direct effect of changing the dipole component and it is not scaled 
        for a change in the quadrupole.

        Ek: CFD in horizontal dimension
        """
        
        if not hasattr(Ek, 'avDispersion'):
            Ek.average()
        eta_k = np.squeeze(Ek.avDispersion)      
        
        term1 = (eta_k) / (self.mcf * self.circumference)
        
        return term1

    def dxsex_ddip(self,Ei: Elements,  Ek: Elements):
        """
        Calculates the orbit displacement at an element Ei due to a change in
        PolynomB[0] in element Ek. with quadrupole moment inside!
        
        This is literally just the orbit response matrix!
        """
        Rik = self.Rab_thick2_K(Ei, Ek) + self.Rab_thick2_disp(Ei, Ek)

        return Rik
    
    def dxpsex_ddip(self,Ei: Elements,  Ek: Elements):
        """
        Calculates the orbit displacement at an element Ei due to a change in
        PolynomB[0] in element Ek. with quadrupole moment inside!
        
        This is literally just the orbit response matrix!
        """
        
        Ei0 = Elements(self.ring, self.all_optics, Ei._ind-1, self.dir_ind, self.sgn)
        Ei0.broadcasters(Ei._bAxis, Ei._ndim)
        
        x1 = self.dxsex_ddip(Ei, Ek)
        x0 = self.dxsex_ddip(Ei0, Ek)
        
        return (x1-x0)/Ei0.LengthB
    

    def dCFD_denergy(self, Ei: Elements, Ej: Elements, Ek: Elements):
        """
        Calculates the change in closed orbit energy caused by changing the 
        dipole component of a CFD sacaled for a proportional change in the quadrupole component.
        In a ring with an active feedback system where correctors are used to keep the orbit 
        fixed in the BPMs while the RF frequency avoids corrector drift.

        Ei: bpms in horizontal          0 

        Ej: corectors in horizontal     1

        Ek: CFD in horizontal           2
        """

        #TODO: Estaria bé reescriure aquest codi bé amb tensors pq sigui agnòstic a l'orde de les dimensions d'entrada.
        
        # 1. Extract Optics averages
        eta_n = np.squeeze(Ei.dispersion)        
        if not hasattr(Ej, 'avDispersion'):
            Ej.average()
        eta_m = np.squeeze(Ej.avDispersion)  
        if not hasattr(Ek, 'avDispersion'):
            Ek.average()
        eta_k = np.squeeze(Ek.avDispersion)      
        


        Rnm = np.squeeze(self.Rab_thick2_(Ei, Ej)   + self.Rab_thick2_disp(Ei, Ej)) # Shape: (N_BPM, M_COR)
        Rnk = np.squeeze(self.Rab_thick2_K(Ei, Ek)  + self.Rab_thick2_disp(Ei, Ek)) # Shape: (N_BPM, K_CFD)
        geometry = (Ek.Bend / (Ek.K))
        R_inv = np.linalg.pinv(Rnm)
        
        hcm0 = (R_inv @ Rnk) * geometry 
        
        term1 = (eta_k * geometry) / (self.mcf * self.circumference)
        
        term2 = (eta_m @ hcm0) / (eta_m @ R_inv @ eta_n)
        
        # Total shift 
        d_delta_dqk = term1 + term2
        
        return d_delta_dqk
    
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
        Rnk = np.squeeze(self.Rab_thick2_K(Ei, Ek)+ self.Rab_thick2_disp(Ei, Ek)) # Shape: (N_BPM, K_CFD)
        Rlk = np.squeeze(self.Rab_thick2_K(El, Ek)+ self.Rab_thick2_disp(El, Ek)) # Shape: (L_sex, K_CFD)
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