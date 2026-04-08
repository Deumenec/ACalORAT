import numpy as np
from ..Elements import Elements

class ORM:
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
    
    def Rab_thick2_K(self, Ea : Elements, Eb: Elements):
        """ Returns the ORM with thick correctors WITH quadrupolar moment inside
        """
        Eb.correct_entrance()
        Cij1 = self.Cabn(Ea, Eb, 1)
        Sij1 = self.Sabn(Ea, Eb, 1)
        
        Ijc = self.Ikc1(Eb)
        Ijs = self.Iks1(Eb)
        
        return np.real(np.sqrt(Ea.betaB*Eb.betaB)/(2*np.sin(np.pi*self.tune))*(Ijc*Cij1+Ijs*Sij1))
    

    def Rab_thick2_disp(self, Ea : Elements, Eb: Elements):
        """ Returns the dispersion term of the ORM with thick correctors WITHOUT quadrupolar moment inside and no dipole component
        Basically it takes the average...
        """
        if not hasattr(Eb, 'avDispersion'):
            Eb.average()
        return np.real(-Ea.dispersionB*(Eb.avDispersionB))/(self.mcf*self.circumference)
    