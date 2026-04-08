# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:15:19 2026

@author: dhuerta

Script testing the analytical formula for the dispersion derivative with respect to
changing the dipole component of a CFD only.
"""

import os
from pathlib import Path
import numpy as np
import at
import at
import copy
import matplotlib.pyplot as plt

from ACalORAT import numerical
from ACalORAT import read
from ACalORAT import AnaORM
from ACalORAT import plot_utils
from ACalORAT import math_utils



ROOT = Path(__file__).resolve().parent.parent
SAVE = ROOT / "outputs" / "ALBAII_CORR_NOFEEDBACK_ONLY_BEND"

###############################################################################
# Parameters to pass for the calculations                                     #
###############################################################################

    
lattice_file   = 'ring_a2.mat' #Read ALBA II lattice ring_a2.mat or THERING.mat to read the ALBA one
lattice_folder = 'lattices' #Important quan treballis amb aquests!
results        =  SAVE  #A1 for the ALBA lattice and A2 for the ALBAII lattice and CFDA2
direction      = 'h' #v: vertical h: horizontal (SI NOMÉS ES FA EL CÀLCUL D'UNA)
step_exp       =  6
step           =  10**(-step_exp)
compute        =  False
dispersion     =  True
lin_all        =  False #To turn off higher order multipoles
max_ind        =  2     #cutoff index in polynomB

###############################################################################
# Reading the lattice parameters
###############################################################################

ring, ind = read.ALBAII(ROOT  / "data" / "ring_a2.mat")



###############################################################################
# Configuration of file name for the different options used
###############################################################################

if lin_all == True:
    linname = "l"+ str(max_ind)+"_"  
if lin_all == False:
    linname = "nl_"
    
if dispersion == True:
    dsname = "d_"    
if dispersion == False:
    dsname = "nd_" 

prefix = linname + dsname

if dispersion == False:
    ring.disable_6d()

if lin_all == True: #DESACTIVA TOTS ELS Sextupols i ordres superiors 
    for element in filter(at.checkattr("PolynomB"), ring):
        #print(element.FamName)
        i=max_ind
        while(i !=0):
            if len(element.PolynomB)>i:
                element.PolynomB[i] = 0
                i +=1
            else:
                i=0

#Experimental index:  32, an only vertical sextupole  corrector 
#To compare with the most simple term with only one sextupole!
#ring[32].PolynomB[2] = 100


for i in ind["cor"]["h"]: ring[i].KickAngle = np.array([0,0])
for i in ind["cor"]["v"]: ring[i].KickAngle = np.array([0,0])



"""
Optionally REMOVING ENTRANCE ANGLES IN DIPOLES TO SIMPLIFY CALCULATIONS
"""
    
"""
for i in ind["dip"]: 
    ring[i].EntranceAngle = 0
    ring[i].ExitAngle = 0
"""


###############################################################################
# Calculating the numerical dispersion derivative with respect to changing dip in CFDs
###############################################################################


def disp_i_dk(ring, cor_idx):
    """
    For a given horizontal corrector, calculate the derivative of horizontal 
    dispersion at the BPMs with respect to changing its kick angle.
    cor_idx: index of the changed corrector
    """
    ring = copy.deepcopy(ring)

    # Step horizontal KickAngle (+step)
    ring[cor_idx].KickAngle[0] += step
    optics_p = at.get_optics(ring, refpts=range(len(ring)))[2]
    disp_p = optics_p["dispersion"][ind["bpm"], 0]
    x_p = optics_p["closed_orbit"][ind["sex"], 0]
    xp_p = optics_p["closed_orbit"][ind["sex"], 1]
    e1 = optics_p["closed_orbit"][:, 4]

    # Step horizontal KickAngle (-step)
    ring[cor_idx].KickAngle[0] -= 2*step
    optics_n = at.get_optics(ring,refpts=range(len(ring)))[2]
    disp_n = optics_n["dispersion"][ind["bpm"], 0]
    x_n = optics_n["closed_orbit"][ind["sex"], 0]
    xp_n = optics_n["closed_orbit"][ind["sex"], 1]
    e2 = optics_n["closed_orbit"][:, 4]
    
    return (disp_p-disp_n)/(2*step), np.average(e1-e2)/(2*step), (x_p-x_n)/(2*step), (xp_p-xp_n)/(2*step)

def ddisp_de(ring):
    """
    Returns the derivative of dispersion with respect to changing the energy.
    """
    step = 0.0001
    ring = copy.deepcopy(ring)
    mcf_val = numerical.get_mcf(ring)
    ring_freq = ring.get_rf_frequency()
    delta_total = (2 * step) / (mcf_val * ring_freq)
    
    ring.set_cavity(Frequency=ring_freq + step)
    optics_p = at.get_optics(ring,refpts=range(len(ring)))[2]
    disp_p = optics_p["dispersion"][ind["bpm"], 0]

    ring.set_cavity(Frequency=ring_freq - step)
    optics_n = at.get_optics(ring,refpts=range(len(ring)))[2]
    disp_n = optics_n["dispersion"][ind["bpm"], 0]
    
    return (disp_p-disp_n)/(delta_total)



# Initialize arrays with the length of horizontal correctors!
n_cor = len(ind["cor"]["h"])

di_dk = np.zeros((n_cor, len(ind["bpm"]))) 
de_dk = np.zeros((n_cor))
dx_dk = np.zeros((n_cor, len(ind["sex"])))
dxp_dk = np.zeros((n_cor, len(ind["sex"])))

if compute:
    for i in range(n_cor):
        di_dk[i], de_dk[i], dx_dk[i], dxp_dk[i] = disp_i_dk(ring, ind["cor"]["h"][i])
        
    np.save(SAVE / (linname +  "di_dk"), di_dk )
    # ... (rest of your saves)
        
    np.save(SAVE / (linname +  "di_dk"),di_dk )
    np.save(SAVE / (linname +  "de_dk"),de_dk )    
    np.save(SAVE / (linname +  "dx_dk"),dx_dk )        
    np.save(SAVE / (linname +  "dxp_dk"),dxp_dk )    
    
di_dk = np.load(SAVE / (linname + "di_dk.npy"))
de_dk = np.load(SAVE / (linname + "de_dk.npy"))
dx_dk = np.load(SAVE / (linname + "dx_dk.npy"))
dxp_dk = np.load(SAVE / (linname + "dxp_dk.npy"))





dRij_de = np.transpose(numerical.quickdORMdEnergy(ring, ind, c_disp=0)["h"], axes = (1, 0))


cORM = AnaORM.AnaORM(ring,"h", ind)
cORM.assign_optics()
cORM.bpm.broadcasters(1, 2)
cORM.dip.broadcasters(0, 2)
cORM.quad.broadcasters(0, 2)
cORM.cor.broadcasters(0, 2)


Rij = (cORM.Rab_thick2_(cORM.bpm, cORM.cor)+ cORM.Rab_thick2_disp(cORM.bpm, cORM.cor) )

di_dk_no_sext = -dRij_de


di_dk_k_change = cORM.dni_dqk_integral(cORM.bpm, cORM.cor) 

cORM.bpm.broadcasters(1, 3)
cORM.cor.broadcasters(0, 3)
cORM.sex.broadcasters(2, 3)

#Closed Orbit calculations

#dx_dk = np.transpose( dx_dk)
#dxp_dk = np.transpose( dxp_dk)



di_denergy = ddisp_de(ring)
di_dk_energy_term = di_denergy[None, :]* de_dk[:, None]

a_error= di_dk- di_dk_no_sext
f1 = -di_dk_no_sext -di_dk_energy_term  #AQUESTA ÉS LA MILLOR FORMULA QUE HE TROBAT

error1 = math_utils.normalized_RMSE(di_dk, f1,1)
print(np.std(di_dk-f1))
plt.plot(error1)
adif = di_dk-f1

cORM = AnaORM.AnaORM(ring,"h", ind)
cORM.assign_optics()
cORM.bpm.broadcasters(1, 2)
cORM.cor.broadcasters(0, 2)
cORM.quad.broadcasters(0, 2)


plt.savefig("err_by_dip.pdf")

"""
#a_term = cORM.dni_dhk_sex_term(cORM.bpm, cORM.dip, cORM.sex, dx_dk[:, :, None], dxp_dk[:, :, None])

a_error = di_dk - di_dk_super

aa_sex_simple = cORM.dni_dhk_sex_term_simple(cORM.bpm, cORM.dip, cORM.sex, dx_dk[:, :, None], dxp_dk[:, :, None])

cORM = AnaORM.AnaORM(ring,"h", ind)
cORM.assign_optics()
cORM.bpm.broadcasters(0, 2)
cORM.sex.broadcasters(1, 2)

av_dx = dx_dk + dxp_dk*cORM.sex.Length/2
a_test_dq = cORM.dni_dqk_integral(cORM.bpm, cORM.sex)

aa_bo = av_dx[None, :, :]* a_test_dq[:, None, :]

cORM = AnaORM.AnaORM(ring,"h", ind)
cORM.assign_optics()
cORM.sex.broadcasters(0, 2)
cORM.dip.broadcasters(1, 2)
a = cORM.Rab_thick2_(cORM.sex, cORM.dip)[:, 0]
ax = dx_dk[0]

for i in ind["sex"]: ring[i].KickAngle = np.array([0,0])
"""

"""
Resp = at.latticetools.OrbitResponseMatrix(ring, "h", ind["bpm"], ind["sex"], steerdelta= 1e-6)
Resp.build_tracking(tol=1e-12, max_iterations=150) 
ORM_test = Resp.response
"""
#Extremadament interessant comentar tot això amb el ZEUS SIUSPLAU el DILLUNS amb el terme dels sextupols posat!!!

""" #Checking 
for i in ind["dip"]: ring[i].KickAngle = np.array([0,0])
Resp = at.latticetools.OrbitResponseMatrix(ring, "h", ind["bpm"], ind["dip"], steerdelta= 1e-6)
Resp.build_tracking(tol=1e-12, max_iterations=150) 
ORMH = np.transpose( Resp.response, axes=(1, 0))
"""

"""
aa = cORM.Rab_thick2_disp_K(cORM.bpm, cORM.dip)

cORM = AnaORM.AnaORM(ring,"h", ind)
cORM.assign_optics()
cORM.bpm.broadcasters(1, 3)
cORM.dip.broadcasters(0, 3)
cORM.quad.broadcasters(2, 3)
cORM.CFD.broadcasters(2, 3)

denergy   = cORM.ddip_denergy(cORM.dip)[:, None, None]*cORM.dip.KB/cORM.dip.BendB

quad1 = cORM.dni_dqk_integral(cORM.bpm, cORM.CFD)
quad2 = cORM.dni_dqk_integral(cORM.bpm, cORM.quad)

delta_K_CFD  = -cORM.CFD.KB * denergy
delta_K_quad = -cORM.quad.KB * denergy

# Multiply the derivative by Delta K and sum!
chromatic_disp =np.real( np.sum(quad1 * delta_K_CFD, axis=2) 
                 + np.sum(quad2 * delta_K_quad, axis=2) )

di_dk_test = di_dk_ana + chromatic_disp

"""
"""
error1 = math_utils.normalized_RMSE(di_dk, di_dk_super ,1)

plt.plot(error1)

cORM = AnaORM.AnaORM(ring,"h", ind)
cORM.assign_optics()
cORM.bpm.broadcasters(0, 3)
cORM.dip.broadcasters(1, 3)
cORM.sex.broadcasters(2, 3)


a = cORM.dRij_dqk_thick23(cORM.bpm, cORM.dip, cORM.sex)
ab = cORM.dni_dqk_integral(cORM.bpm,cORM.sex)

"""
    
    