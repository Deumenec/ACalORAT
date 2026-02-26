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
import copy 
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
step           =  1e-4

p              ={"lin_all"        :  True,  #To turn off higher order multipoles
                 "max_ind"        :  2,      #Cutoff index in polynomB, simplifies the ring for certain calculations
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
#Prompt to calculate numerically the derivative of the response matrix with respect to CFDº

if  p["calculate"]:
    if not os.path.exists(SAVE / pathCFD):
        os.mkdir(SAVE / pathCFD)
    num_dORM_dqH, num_dORM_dqV, dFreq_dCFD, dKicksH_dCFD, dKicksV_dCFD, x_sex, energy = numerical.dORM_dCFD(ring, ind, step ,multithread=True, method="Cor_SVD", num = 26) #In ALBAII all dipoles are CFD!
    np.save(SAVE /pathCFD /"num_dORM_dqH",num_dORM_dqH)
    np.save(SAVE /pathCFD /"num_dORM_dqV",num_dORM_dqV)
    np.save(SAVE /pathCFD /"dFreq_dCFD",dFreq_dCFD)
    np.save(SAVE /pathCFD /"dKicksH_dCFD",dKicksH_dCFD)
    np.save(SAVE /pathCFD /"dKicksV_dCFD",dKicksV_dCFD)
    np.save(SAVE /pathCFD /"x_sex",x_sex)
    np.save(SAVE /pathCFD /"energy",energy)
    

else:
    try:
        num_dORM_dqH = np.load(SAVE /pathCFD /"num_dORM_dqH.npy")
        num_dORM_dqV = np.load(SAVE /pathCFD /"num_dORM_dqV.npy")
        dFreq_dCFD   = np.load(SAVE /pathCFD /"dFreq_dCFD.npy")
        dKicksH_dCFD = np.load(SAVE /pathCFD /"dKicksH_dCFD.npy")
        dKicksV_dCFD = np.load(SAVE /pathCFD /"dKicksV_dCFD.npy")
        x_sex        = np.load(SAVE /pathCFD /"x_sex.npy")
        energy       = np.load(SAVE /pathCFD /"energy.npy")
    except:
        raise ImportError("No hi ha respostes calculades")

optics = at.get_optics(ring, refpts = range(len(ring)))[2]

#Kicktests:
#Tests if the kicks performed by the numerical calculation respect the dispersion sum condition

corx_disp = optics["dispersion"][ind["cor"]["h"]][:,0]
cory_disp = optics["dispersion"][ind["cor"]["v"]][:,0]



feedx = np.sum(corx_disp*dKicksH_dCFD, axis = 1)
feedy = np.sum(cory_disp*dKicksV_dCFD, axis = 1)
feedxm = np.sum(dKicksH_dCFD, axis = 1)
feedym = np.sum(dKicksV_dCFD, axis = 1)


cORM = AnaORM.AnaORM(ring,"v", ind) #Automatically uses the vertical correctors!
cORM.assign_optics()
#Propagate optics along the dip to get the exit

cORM.dip.correct_strength() #Correct entrance and effective force for the calculations in the CFD
cORM.bpm.broadcasters(1, 3)
cORM.cor.broadcasters(2, 3)
cORM.dip.broadcasters(0, 3)

cORM.add_element("allQuad", ind["dip"], "h")
cORM.allQuad.broadcasters(0,3)
#Terms for the energy perturbation:    

cORM.add_element("bpmh", ind["bpm"], "h")
cORM.add_element("diph", ind["dip"], "h") 
cORM.diph.average()
cORM.add_element("corh", ind["cor"]["h"], "h")

#dRijdEnergy_num = numerical.dORMdEnergy(ring, ind)
dRijdEnergy= cORM.dRij_dEnergy(cORM.bpm, cORM.cor, cORM.allQuad)
dRijdEnergy_quick = numerical.quickdORMdEnergy(ring, ind)

Rij = np.sum( cORM.Rab_thick2_(cORM.bpm, cORM.cor), axis = 0)

#Term corresponding to the quadrupole in CFD with a factor due to the extra change due to change in effective quadrupole because of the change in dipole moment
Aana_dORM_dCFDV00 = cORM.dRij_dqk_thick23(cORM.bpm, cORM.cor, cORM.dip)   
delta_dk = cORM.dRij_dCFD_energy(cORM.bpmh, cORM.corh, cORM.diph)[0:26]
#Observem com en aquest cas, la diferència entre la matriu de resposta i la analítica amb només els quadrupols dona matrius proporcionals respecte cada quadrupol          

aatest= num_dORM_dqV-Aana_dORM_dCFDV00[0:26]

#I per exemple:

aatest[0]/aatest[1]

#Es constant 12 exepte els punts on la resposta és molt petita (error numèric) 
#Això suggereix que el terme "important" que falta és la derivada respecte l'energia.
#De fet, podem calcular la constant de proporcionalitat per cada CFD, obtenint una estimació de l'energia

constants0 = num_dORM_dqV/dRijdEnergy
c0_av = np.average(constants0, axis = (1,2))
c0_dv = np.std(constants0, axis = (1,2))

constants1 = (num_dORM_dqV-Aana_dORM_dCFDV00[0:26])/dRijdEnergy
c1_av = np.average(constants1, axis = (1,2))# /( cORM.diph.Bend[0:26]/(cORM.diph.K[0:26]*cORM.diph.Length[0:26]))/2
c1_dv = np.std(constants1, axis = (1,2))

#Observem com utilitzant aquests canvis d'energia numèrics, podem aconseguir estimacions molt bones!

Aana_dORM_dCFDV1 = Aana_dORM_dCFDV00[0:26] + c1_av[:,None, None]*dRijdEnergy[None, :, :]
    
#Aquesta gràfica compara la fórmula de l'energia 1. Numèrica, 2. Analítica 3. que minimitza la diferència 4. estimació amb els canvis a l'anell

optics = at.get_optics(ring, refpts=range(len(ring)))

all_disp = optics[2]["dispersion"]
av_disp =np.array(numerical.compute_average_dispersion(ring, ind["cor"]["v"], all_disp))
av_disp_dip =np.array(numerical.compute_average_dispersion(ring, ind["dip"], all_disp))

#av_disp = all_disp[ ind["cor"]["v"], 0]
#av_disp = np.zeros(len(ind["cor"]["v"]))
mcf_val = numerical.get_mcf(ring)
num_energy =  dFreq_dCFD/ (mcf_val * ring.get_rf_frequency()) - av_disp_dip[0:26]*cORM.diph.Bend[0:26]/(cORM.diph.K[0:26]*cORM.diph.Length[0:26])

ana_energy = delta_dk

plt.plot(energy, color = "red", label = "Energia numèrica")
plt.plot(num_energy, color = "blue", label = "Estimació energia numèrica") 
#plt.plot(c1_av, color = "orange", label = "Optimal Energy")  
plt.plot(ana_energy/2, color = "green", label = "Energia analítica")
plt.legend()
plt.show()

#Test to check if the numerical orbit feedback condition is satisfied:
    
energy_test = np.sum(cORM.corh.avDispersion[None, :]*dKicksH_dCFD, axis =1)







