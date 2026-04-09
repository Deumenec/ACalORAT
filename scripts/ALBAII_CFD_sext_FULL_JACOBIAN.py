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

SAVE = ROOT / "outputs" / "ALBAII_CFD_sext"

os.chdir(ROOT)
if not os.path.exists(SAVE):
    os.mkdir(SAVE)

###############################################################################
# Parameters to pass for the calculations
###############################################################################



lattice_file   = 'ring_a2.mat' #Read ALBA II lattice ring_a2.mat or THERING.mat to read the ALBA one
results        = 'A2' #A1 for the ALBA lattice and A2 for the ALBAII lattice and CFDA2
step           =  1e-6

p              ={"lin_all"        :  False,  #To turn off higher order multipoles
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
    num_dORM_dqH, num_dORM_dqV, dFreq_dCFD, dKicksH_dCFD, dKicksV_dCFD, x_sex,dx_sex , energy = numerical.dORM_dCFD(ring, ind, step ,multithread=True, method="Cor_SVD") #In ALBAII all dipoles are CFD!
    np.save(SAVE /pathCFD /"num_dORM_dqH",num_dORM_dqH)
    np.save(SAVE /pathCFD /"num_dORM_dqV",num_dORM_dqV)
    np.save(SAVE /pathCFD /"dFreq_dCFD",dFreq_dCFD)
    np.save(SAVE /pathCFD /"dKicksH_dCFD",dKicksH_dCFD)
    np.save(SAVE /pathCFD /"dKicksV_dCFD",dKicksV_dCFD)
    np.save(SAVE /pathCFD /"x_sex",x_sex)
    np.save(SAVE /pathCFD /"dx_sex",dx_sex)
    np.save(SAVE /pathCFD /"energy",energy)
    

else:
    try:
        num_dORM_dqH = np.load(SAVE /pathCFD /"num_dORM_dqH.npy")
        num_dORM_dqV = np.load(SAVE /pathCFD /"num_dORM_dqV.npy")
        dFreq_dCFD   = np.load(SAVE /pathCFD /"dFreq_dCFD.npy")
        dKicksH_dCFD = np.load(SAVE /pathCFD /"dKicksH_dCFD.npy")
        dKicksV_dCFD = np.load(SAVE /pathCFD /"dKicksV_dCFD.npy")
        x_sex        = np.load(SAVE /pathCFD /"x_sex.npy")
        dx_sex        = np.load(SAVE /pathCFD /"dx_sex.npy")
        energy       = np.load(SAVE /pathCFD /"energy.npy")
    except:
        raise ImportError("No hi ha respostes calculades")


#Now here we calculate numerically the Jacobian with quadrupoles tu
dRij_dEnergy = numerical.quickdORMdEnergy(ring, ind)


#Horizontal derivative calculation:

cORM = AnaORM.AnaORM(ring,"h", ind)
cORM.assign_optics()
cORM.bpm.broadcasters(0, 3)
cORM.cor.broadcasters(1, 3)
cORM.dip.broadcasters(2, 3)

#We also use the horizontal response calculations to get the necessary information
#For the vertical calculations: the energy change, sextupole orbit change
#sextupole orbit derivative change and corrector activation
denergy = cORM.dCFD_denergy(cORM.bpm, cORM.cor, cORM.dip)
#TODO: Write and validate dkicksH

cORM.bpm.broadcasters(0, 4)
cORM.cor.broadcasters(1, 4)
cORM.dip.broadcasters(2, 4)
cORM.sex.broadcasters(3, 4)

x_sex_ana  = x_sex #cORM.dxldCFDk(cORM.bpm, cORM.cor, cORM.dip, cORM.sex)
dx_sex_ana = dx_sex #cORM.dpxldCFDk(cORM.bpm, cORM.cor, cORM.dip, cORM.sex)
dKicksH_dCFD_ana = 0
#TODO: write the theta the theta analytically and validate!
#dtheta_sex = AnaORM.extract_kicks(dKicksH_dCFD,ind["cor"]["h"], ind["sex"])

#Put the relevant constants in the right broadcasting dimensions
x_sex_ana  = x_sex_ana[None, None, :, :]
dx_sex_ana = dx_sex_ana[None, None, :, :]
dtheta_sex = dKicksH_dCFD[None, None, :, :]

#Revert broadcasters
cORM.bpm.broadcasters(0, 3)
cORM.cor.broadcasters(1, 3)
cORM.dip.broadcasters(2, 3)
#Sembla malament l'horitzontal total fet així!
thickh = ( cORM.dRij_dqk_thick23(cORM.bpm, cORM.cor, cORM.dip) 
          + 0*cORM.dRij_dqk_thick23_disp(cORM.bpm, cORM.cor, cORM.dip) #Aquí aquest terme ajuda però falta bastanta cosa!!!
          + cORM.dRij_dk_energy_term(cORM.bpm, cORM.cor, cORM.dip, dRij_dEnergy["h"], denergy))

a = cORM.dRij_dk_energy_term(cORM.bpm, cORM.cor, cORM.dip, dRij_dEnergy["h"], denergy)
a = np.transpose(a, (2, 0, 1))
cORM.bpm.broadcasters(0, 4)
cORM.cor.broadcasters(1, 4)
cORM.dip.broadcasters(2, 4)
cORM.sex.broadcasters(3, 4)


AAsexth = cORM.dRi_dk_sex_term(cORM.bpm, cORM.cor, cORM.dip, cORM.sex, x_sex_ana, dx_sex_ana, dtheta_sex)


#TODO: Write the sextupole term in here!
#Vertical derivative calculation:

cORM = AnaORM.AnaORM(ring,"v", ind)
cORM.assign_optics()
cORM.dip.correct_entrance()
cORM.bpm.broadcasters(0, 3)
cORM.cor.broadcasters(1, 3)
cORM.dip.broadcasters(2, 3)

#La formula validada per els CFD que prediu perfecte el canvi amb la component quadrupolar
#Més el terme corresponent a la variació de l'energia, recordem que aquí les dimensions de
#Broadcasting estàn hard-coded així que millor no tocar res sense validar


thickv = (cORM.dRij_dqk_thick23(cORM.bpm, cORM.cor, cORM.dip) 
          + 0*cORM.dRij_dk_energy_term(cORM.bpm, cORM.cor, cORM.dip, dRij_dEnergy["v"], denergy))

#Finally, we add the sextupole term by rebuilding the broadcasters

cORM.bpm.broadcasters(0, 4)
cORM.cor.broadcasters(1, 4)
cORM.dip.broadcasters(2, 4)
cORM.sex.broadcasters(3, 4)


AAsextv = cORM.dRi_dk_sex_term(cORM.bpm, cORM.cor, cORM.dip, cORM.sex, x_sex_ana, dx_sex_ana, dtheta_sex)



##########################################################
# Validation plot
##########################################################

#Importat to keep the correct broadcasting dimensions!

thickv = np.transpose(thickv, (2,0,1))
thickh = np.transpose(thickh, (2,0,1))
AAsextv = np.transpose(AAsextv, (2,0,1))
AAsexth = np.transpose(AAsexth, (2,0,1))
#La vertical sembla que està bastant aprop!

av = -num_dORM_dqV +thickv
ah = num_dORM_dqH- thickh

plot_utils.plot_both_Zeus(num_dORM_dqV , num_dORM_dqH, thickv, thickh)





