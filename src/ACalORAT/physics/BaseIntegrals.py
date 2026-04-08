import numpy as np
from ..Elements import Elements


class BaseIntegrals:
    """
    Generic inegrals used for the calculations of analytical formulas
    """
    def Cabn(self, Ea: Elements, Eb: Elements, n: int):
        """ """
        return np.cos(n*np.abs(Ea.muB-Eb.muB)-n*np.pi*self.tune)
    
    def Sabn(self, Ea: Elements, Eb: Elements, n: int):
        """ """
        #The minus epsilon ensures that sign(0) = -1 as defined in the report.
        return np.sign(Ea.muB-Eb.muB-0.0000001)*np.sin(n*np.abs(Ea.muB-Eb.muB)-n*np.pi*self.tune)
    
    def Ik0_(self, Ek: Elements):
        """Integral for element WITHOUT quadrupolar moment"""
        return Ek.betaB * Ek.LengthB - Ek.alphaB * Ek.LengthB**2
    
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
        """Integral for element WITH quadrupolar moment (the definition with a hat)"""
        return ((np.sqrt(Ek.betaB/Ek.KB)*np.sin(Ek.LengthB*np.sqrt(Ek.KB))+(Ek.alphaB*(np.cos(Ek.LengthB*np.sqrt(Ek.KB))-1))/(Ek.KB*np.sqrt(Ek.betaB))))/(Ek.LengthB*np.sqrt(Ek.betaB))
        
    def Iks1(self, Ek: Elements):
        """Integral for element WITH quadrupolar moment (the definition with a hat)"""
        return -(np.cos(Ek.LengthB*np.sqrt(Ek.KB))-1)/(Ek.KB*np.sqrt(Ek.betaB)*Ek.LengthB*np.sqrt(Ek.betaB))  
    
    def Iks2_(self, Ek: Elements): 
        """Integral term WITHOUT quadrupole moment inside"""
        return Ek.LengthB**2 - (2 * Ek.alphaB / (3 * Ek.betaB)) * Ek.LengthB**3
        
    def Iks2(self, Ek: Elements): 
        """Integral term with quadrupole moment inside"""
        return 1/(2*Ek.KB)*(1-np.cos(2*np.sqrt(Ek.KB)*Ek.LengthB)+Ek.alphaB/(Ek.betaB)*(np.sin(2*np.sqrt(Ek.KB)*Ek.LengthB)/np.sqrt(Ek.KB)-2*Ek.LengthB))
      
    
    def Ikc2_(self, Ek: Elements):
        """Integral term without quadrupole moment inside"""
        return Ek.betaB * Ek.LengthB - Ek.alphaB * Ek.LengthB**2 + ((Ek.alphaB**2 - 1) / (3.0 * Ek.betaB)) * Ek.LengthB**3
        
    def Ikc2(self, Ek: Elements):
        """Integral term with quadrupole moment inside"""
        return self.Ik0(Ek) + (np.sin(2*np.sqrt(Ek.KB)*Ek.LengthB)/(2*np.sqrt(Ek.KB))-Ek.LengthB)/(Ek.KB*Ek.betaB)
    
    
    
    
    
    def Ijc1_L(self, Ej: Elements):
        """Integral term for elements WITH quadrupole moment inside divided by the length of the element"""
        return np.real(np.sin(np.sqrt(Ej.KB)*Ej.LengthB)/(Ej.LengthB*np.sqrt(Ej.KB)) + Ej.alphaB*(np.cos(np.sqrt(Ej.KB)*Ej.LengthB)-1)/(Ej.KB*Ej.betaB*Ej.LengthB))
        
    def Ijs1_L(self, Ej: Elements):
        """Integral term for elements WITH quadrupole moment inside divided by the length of the elmeent"""
        return -(np.cos(np.sqrt(Ej.KB)*Ej.LengthB)-1)/(Ej.KB*Ej.betaB*Ej.LengthB)
    
    
    ###########################################################################
    # Thik integrals for sextupole feedbacks dispersion:
    ###########################################################################

    def I_disp_l0_nf(self, El: Elements, i, j):
        """Integral for dispersion * beta inside sextupoles for the quadrupole 
        feedback to calculate the derivative of dispersion when changing only
        a bending mangent.
        
        i = dx_l/dqk

        j = dx_l'/dqk
        """
        a = El.alphaB
        b = El.betaB
        c = El.gammaB
        L = El.LengthB
        s = El.SB
        n = El.dispersionB
        np = El.dispersionpB
        
        
        return 2*s*( 1/12*(4*(L**3*c - 3*L**2*a + 3*L*b)*i + (3*L**4*c - 8*L**3*a 
                    + 6*L**2*b)*j)*n + 1/60*(5*(3*L**4*c - 8*L**3*a + 6*L**2*b)*i 
                    + 2*(6*L**5*c - 15*L**4*a + 10*L**3*b)*j)*np)

    def I_disp_lc2_nf(self, El: Elements, i, j):
        """Integral for dispersion * beta inside sextupoles for the quadrupole 
        feedback to calculate the derivative of dispersion when changing only
        a bending mangent.
        
        i = dx_l/dqk

        j = dx_l'/dqk
        """
        a = El.alphaB
        b = El.betaB
        c = El.gammaB
        L = El.LengthB
        s = El.SB
        n = El.dispersionB
        np = El.dispersionpB
        
        
        return 2*s*( 1/5*(a**2*j*np/b - j*np/b)*L**5 + 1/4*(a**2*j*n/b + a**2*i*np/b 
                   - 2*a*j*np - j*n/b - i*np/b)*L**4 + 1/3*(a**2*i*n/b - 2*a*j*n 
                   - 2*a*i*np + b*j*np - i*n/b)*L**3 + L*b*i*n - 1/2*(2*a*i*n 
                   - b*j*n - b*i*np)*L**2)
                                        
    def I_disp_ls2_nf(self, El: Elements, i, j):
        """Integral for dispersion * beta inside sextupoles for the quadrupole 
        feedback to calculate the derivative of dispersion when changing only
        a bending mangent.
        
        i = dx_l/dqk

        j = dx_l'/dqk
        """
        a = El.alphaB
        b = El.betaB
        c = El.gammaB
        L = El.LengthB
        s = El.SB
        n = El.dispersionB
        np = El.dispersionpB
        
        
        return 2*s*( -1/30*(5*(2*(2*L**3*a - 3*L**2*b)*i + (3*L**4*a - 4*L**3*b)*j)*n 
                     + (5*(3*L**4*a - 4*L**3*b)*i + 3*(4*L**5*a - 5*L**4*b)*j)*np)/b)
        
        
    ###########################################################################
    # Thik integrals for sextupole feedbacks response matrix:
    ###########################################################################

    def Ilk0(self, El: Elements, i, j, k):
        """Integral of the derivative of displacement in its broadcasting dimensions
        
        Derivative of parameters governing the closed orbit inside of the sextupole
        with already the correct broadcasting dimensions

        i = dx_l/dqk

        j = dx_l'/dqk

        k = dtheta_l/dqk

        """ 
        a = El.alphaB
        b = El.betaB
        c = El.gammaB
        L = El.LengthB
        s = El.SB
        return s*(1/10*(5*c*j + 2*c*k)*L**4 + 1/6*(4*c*i - 8*a*j - 3*a*k)*L**3 - 1/3*(6*a*i - 3*b*j - b*k)*L**2 + 2*L*b*i)
     
    def Ilkc2(self, El: Elements, i, j, k):
        """Integral of the derivative of displacement in its broadcasting dimensions
        
        Derivative of parameters governing the closed orbit inside of the sextupole
        with already the correct broadcasting dimensions

        i = dx_l/dqk

        j = dx_l'/dqk

        k = dtheta_l/dqk
        
        """
        a = El.alphaB
        b = El.betaB
        c = El.gammaB
        L = El.LengthB
        s = El.SB
        return s*(1/10*(5*c*j + 2*c*k - 10*j/b - 4*k/b)*L**4 + 1/6*(4*c*i - 8*a*j - 3*a*k - 8*i/b)*L**3 - 1/3*(6*a*i - 3*b*j - b*k)*L**2 + 2*L*b*i)
        
    def Ilks2(self, El: Elements, i, j, k):
        """Integral of the derivative of displacement in its broadcasting dimensions
        
        Derivative of paramaeters governing the closed orbit inside of the sextupole
        with already the correct broadcasting dimensions
        i = dx_l/dqk
        j = dx_l'/dqk
        k = dtheta_l/dqk
        """
        a = El.alphaB
        b = El.betaB
        c = El.gammaB
        L = El.LengthB
        s = El.SB
        return s*(-1/5*L**4*(5*a*j/b + 2*a*k/b) - 1/6*L**3*(8*a*i/b - 8*j - 3*k) + 2*L**2*i)
