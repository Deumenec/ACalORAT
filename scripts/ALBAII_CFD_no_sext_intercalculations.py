# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 12:37:59 2026

@author: dhuerta

Calculating the derivative of the Orbit Response Matrix numerically with dispersion
but without sextupoles in the ALBAII lattice and testing analytical formulas.
"""

import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import at
from ACalORAT import numerical
from ACalORAT import read
from ACalORAT import AnaORM
from ACalORAT import plot_utils

ROOT = Path(__file__).resolve().parent.parent

SAVE = ROOT / "outputs" / "ALBAII_CFD_no_sext"

os.chdir(ROOT)
if not os.path.exists(SAVE):
    os.mkdir(SAVE)

###############################################################################
# Parameters to pass for the calculations
###############################################################################

    
lattice_file   = 'ring_a2.mat' #Read ALBA II lattice ring_a2.mat or THERING.mat to read the ALBA one
results        = 'A2' #A1 for the ALBA lattice and A2 for the ALBAII lattice and CFDA2
step           =  1e-5

p              ={"lin_all"        :  True,  #To turn off higher order multipoles
                 "max_ind"        :  3,      #Cutoff index in polynomB, simplifies the ring for certain calculations
                 "calculate"      :  False}


###############################################################################
# Reading the lattice parameters
###############################################################################

ring, ind = read.ALBAII(ROOT  / "data" / "ring_a2.mat")

    
if p["lin_all"] == True: #DESACTIVA TOTS ELS Sextupols i ordres superiors 
    for element in filter(at.checkattr("PolynomB"), ring):
        #print(element.FamName)
        i=p["max_ind"]
        while(i !=0):
            if len(element.PolynomB)>i:
                element.PolynomB[i] = 0
                i +=1
            else:
                i=0

#Adding kickangles to the correctors in ALBAII
for i in ind["cor"]["h"]: ring[i].KickAngle = np.array([0,0])
for i in ind["cor"]["v"]: ring[i].KickAngle = np.array([0,0])

###############################################################################
# Calculating the numerical dORMdCFD if required and saving them
###############################################################################

pathCFD = "Cor_SVD"
ind["CFD"] = ind["CFD"][0:25]

#Prompt to calculate numerically the derivative of the response matrix with respect to CFDº

if  p["calculate"]:
    if not os.path.exists(SAVE / pathCFD):
        os.mkdir(SAVE / pathCFD)
    num_dORM_dqH, num_dORM_dqV, dFreq_dCFD, dKicksH_dCFD, dKicksV_dCFD, x_sex, dx_sex, energy, dni_dCFD = numerical.dORM_dCFD(ring, ind, step ,multithread=True, method="Cor_SVD") #In ALBAII all dipoles are CFD!
    np.save(SAVE /pathCFD /"num_dORM_dqH",num_dORM_dqH)
    np.save(SAVE /pathCFD /"num_dORM_dqV",num_dORM_dqV)
    np.save(SAVE /pathCFD /"dFreq_dCFD",dFreq_dCFD)
    np.save(SAVE /pathCFD /"dKicksH_dCFD",dKicksH_dCFD)
    np.save(SAVE /pathCFD /"dKicksV_dCFD",dKicksV_dCFD)
    np.save(SAVE /pathCFD /"x_sex",x_sex)
    np.save(SAVE /pathCFD /"dx_sex",dx_sex)
    np.save(SAVE /pathCFD /"energy",energy)
    np.save(SAVE /pathCFD /"dni_dCFD",dni_dCFD)


else:
    try:
        num_dORM_dqH = np.load(SAVE /pathCFD / "num_dORM_dqH.npy")
        num_dORM_dqV = np.load(SAVE /pathCFD / "num_dORM_dqV.npy")
        dFreq_dCFD   = np.load(SAVE /pathCFD / "dFreq_dCFD.npy")
        dKicksH_dCFD = np.load(SAVE /pathCFD / "dKicksH_dCFD.npy")
        dKicksV_dCFD = np.load(SAVE /pathCFD / "dKicksV_dCFD.npy")
        x_sex        = np.load(SAVE /pathCFD / "x_sex.npy")
        dx_sex        = np.load(SAVE /pathCFD / "dx_sex.npy")
        energy       = np.load(SAVE /pathCFD / "energy.npy")
        dni_dCFD     = np.load(SAVE /pathCFD / "dni_dCFD.npy")

    except:
        raise ImportError("No hi ha respostes calculades")


cORM = AnaORM.AnaORM(ring,"v", ind) #Automatically uses the vertical correctors!
cORM.assign_optics()
#Propagate optics along the dip to get the exit


cORM.bpm.broadcasters(1, 3)
cORM.cor.broadcasters(2, 3)
cORM.CFD.broadcasters(0, 3)
cORM.quad.broadcasters(0,3)

#Terms for the energy perturbation:    

#dRijdEnergy_num = numerical.dORMdEnergy(ring, ind)
#dRijdEnergy_ana= cORM.dRij_dEnergy(cORM.bpm, cORM.cor, cORM.quad, cORM.CFD)
dRijdEnergy_quick = numerical.quickdORMdEnergy(ring, ind)

Rij = np.sum( cORM.Rab_thick2_(cORM.bpm, cORM.cor), axis = 0)

#Term corresponding to the quadrupole in CFD with a factor due to the extra change due to change in effective quadrupole because of the change in dipole moment

cORM2 = AnaORM.AnaORM(ring, "h", ind)
cORM2.assign_optics()
cORM2.bpm.broadcasters(0, 3)
cORM2.cor.broadcasters(1, 3)
cORM2.CFD.broadcasters(2, 3)
Rij_bo = np.squeeze(cORM2.Rab_thick2_(cORM2.bpm, cORM2.cor))
delta_dk = cORM2.dCFD_denergy(cORM2.bpm, cORM2.cor, cORM2.CFD)

  
#delta_dk = cORM.dRij_dCFD_energy(cORM.bpmh, cORM.corh, cORM.diph)[0:26]
#Observem com en aquest cas, la diferència entre la matriu de resposta i la analítica amb només els quadrupols dona matrius proporcionals respecte cada quadrupol          

#Aquesta gràfica compara la fórmula de l'energia 1. Numèrica, 2. Analítica 3. que minimitza la diferència 4. estimació amb els canvis a l'anell

optics = at.get_optics(ring, refpts=range(len(ring)))

all_disp = optics[2]["dispersion"]
av_disp =np.array(numerical.compute_average_dispersion(ring, ind["cor"]["v"], all_disp))
av_disp_CFD =np.array(numerical.compute_average_dispersion(ring, ind["CFD"], all_disp))

#av_disp = all_disp[ ind["cor"]["v"], 0]
#av_disp = np.zeros(len(ind["cor"]["v"]))
mcf_val = numerical.get_mcf(ring)


ana_energy = np.real(delta_dk)

# Fixed signs, removed Length, and added mcf_val * ring.circumference to the denominator

num_energy = np.real(-dFreq_dCFD / (mcf_val * ring.get_rf_frequency()) + (cORM2.CFD.avDispersion * (cORM2.CFD.Bend / cORM2.CFD.K)) / (mcf_val * ring.circumference))


plt.plot(ana_energy, color = "green", label = "Energia analítica")
plt.plot(energy, color = "red", label = "Energia numèrica", linestyle = "--")


plt.plot(num_energy, color = "blue", label = "Estimació energia numèrica", linestyle = "--") 
plt.legend()
plt.show()

#Test to check if the numerical orbit feedback condition
#As this works, we can as well now calculate the orbit response matrix from respect to CFD in the ring with CFD turned off


#In the vertical direction it is just:
    
dRij_dCFD = cORM.dRij_dqk_thick23(cORM.bpm, cORM.cor, cORM.CFD) + dRijdEnergy_quick["h"][None, :, :]*ana_energy[:,None, None]

#The horizontal displacment at the entrance of sextupoles should just be

cORM3 = AnaORM.AnaORM(ring, "h", ind, old_optics=cORM2.all_optics)
cORM3.assign_optics()
cORM3.bpm.broadcasters(0, 4)
cORM3.cor.broadcasters(1, 4)
cORM3.CFD.broadcasters(2, 4)
cORM3.sex.broadcasters(3, 4)
cORM3.add_element("presex", ind["sex"]-1, "h")
cORM3.presex.broadcasters(3, 4)


x_sex_ana = np.real(cORM3.dxldCFDk(cORM3.bpm, cORM3.cor, cORM3.CFD, cORM3.sex))
x_sex_ana_0 = np.real(cORM3.dxldCFDk(cORM3.bpm, cORM3.cor, cORM3.CFD, cORM3.presex))
dx_sex_ana = (x_sex_ana-x_sex_ana_0)/cORM3.presex.Length
#dKicksH_dCFD_num = np.real(cORM3.dkickdCFD(cORM3.bpm, cORM3.cor, cORM3.CFD))
a = 7
plt.plot(x_sex[a][10:]-x_sex_ana[a][10:])

###############################################################################
# Test on the full dispersion derivative
###############################################################################





