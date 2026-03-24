import numpy as np
from ..Elements import Elements
from .BaseIntegrals import BaseIntegrals

class dORM_dquad(BaseIntegrals):
    """
    Formulas for the Jacobian of the Orbit Response Matrix with respect to 
    changin quadrupole strength either in pure or CFDs
    """
    
    def dRij_dqk_thin(self, Ei : Elements, Ej : Elements, Ek : Elements):
        """
        Considers all elements as thin, results can be greatlly improved by passing average
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

    def dRij_dqk_thick2(self, Ei : Elements, Ej : Elements, Ek : Elements):
        """
        Computes the Jacobian of the ORM with respect to changing thin quadrupoles but thick correctors.
        """
        Cij1 = self.Cabn(Ei, Ej, 1)
        Cik2 = self.Cabn(Ei, Ek, 2)
        Cjk2 = self.Cabn(Ej, Ek, 2)
        Sij1 = self.Sabn(Ei, Ej, 1)
        Sik2 = self.Sabn(Ei, Ek, 2)
        Sjk2 = self.Sabn(Ej, Ek, 2)
        
        #Terms for thick correctors without quadrupole moment inside of them
        Ijc1_L = self.Ikc1_(Ej)
        Ijs1_L = self.Iks1_(Ej)
        
        dRij_terms =  (Cij1 * ( Cik2 + Cjk2 + 2*np.cos(np.pi * self.tune)**2) + 
                       Sij1 * ( Sik2 - Sjk2 + np.sin(2*np.pi*self.tune)*(2*np.heaviside(Ei.muB-Ek.muB, 0)
                           -2*np.heaviside(Ej.muB-Ek.muB, 0)-np.sign(Ei.muB-Ej.muB)))) 
        dTij_terms =  (Sij1 * (Cik2 - Cjk2 +  2* np.cos(np.pi * self.tune)**2) + 
                       Cij1 * (-Sjk2 - Sik2 + np.sin(2*np.pi * self.tune) * (-2*np.heaviside(Ei.muB-Ek.muB, 0) 
                           + 2*np.heaviside(Ej.muB-Ek.muB, 0) + np.sign(Ei.muB-Ej.muB))))
        
        ana_dORM_dq = self.sgn * (Ek.betaB * np.sqrt(Ei.betaB * Ej.betaB) 
         / (8 * np.sin(np.pi * self.tune)* np.sin(2 * np.pi * self.tune)) 
         * (Ijc1_L * dRij_terms + Ijs1_L * dTij_terms))
        
        
        return np.real(ana_dORM_dq) #Per assegurar que retorni un real bé
    
    def dRij_dqk_thick3(self, Ei : Elements, Ej : Elements, Ek : Elements):
        """
        Computes the dRij_dqk asssuming only thick quadrupoles
        """
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
    

    
    def dRij_dqk_thick23(self, Ei : Elements, Ej : Elements, Ek : Elements):
        """
        
        Computes the dRij_dqk asssuming thick correctors without quadrupolar 
        component and thick quadrupoles
        
        """
        #Remember it is only applied if it has not been previously applied.
        Ek.correct_entrance()
        
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

        #Choose integral along correctors depening on if they have quadrupole inside.
        if hasattr(Ej, "K") and Ej.K[0] !=0:
            Ijc1_L = self.Ikc1(Ej)
            Ijs1_L = self.Iks1(Ej)   
        else:
            Ijc1_L = self.Ikc1_(Ej)
            Ijs1_L = self.Iks1_(Ej)
        
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
    

    
    