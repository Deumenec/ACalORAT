# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 09:03:37 2025

@author: dhuerta

This file contains functions used to draw plots
"""

import matplotlib.pyplot as plt
from matplotlib import rc
import math_utils

def plot_both(ORMv, ORMh, nORMv, nORMh, latex = False):
    
    vquadERROR = math_utils.normalized_RMSE(ORMv, nORMv, (1,2))
    vERROR = math_utils.normalized_RMSE(ORMv, nORMv,(0,1,2))

    hquadERROR = math_utils.normalized_RMSE(ORMh, nORMh, (1,2))
    hERROR = math_utils.normalized_RMSE(ORMh, nORMh, (0,1,2))
    #Creating the plot Zeus asked me to:
       
    if latex == True:
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
    fig, axis = plt.subplots(1,2,figsize=(10,5))
    #fig.suptitle("Errors along quadrupoles for the thin formula ", fontsize = 20)
    fig.subplots_adjust(top=0.85)
    plt.ylabel('dORM/dq normalized_RMSE \%')

    axis[0].set_xlabel('Quadrupole')
    axis[1].set_xlabel('Quadrupole')
    axis[0].title.set_text("Vertical direction, Total = "+f"{vERROR:.4f}\%")
    axis[0].plot(vquadERROR)

    axis[1].title.set_text("Horizontal direction, Total = "+f"{hERROR:.4f}\%")
    axis[1].plot(hquadERROR)
    plt.show()
    return
    
def plot_both_Zeus(ORMv, ORMh, nORMv, nORMh, latex = False):
    
    vquadERROR = math_utils.errorZeus(ORMv, nORMv)
    vERROR = math_utils.normalized_RMSE(ORMv, nORMv,(0,1,2))

    hquadERROR = math_utils.errorZeus(ORMh, nORMh)
    hERROR = math_utils.normalized_RMSE(ORMh, nORMh, (0,1,2))
    #Creating the plot Zeus asked me to:
       
    if latex == True:
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
    fig, axis = plt.subplots(1,2,figsize=(13,3))
    #fig.suptitle("Errors along quadrupoles for the thin formula ", fontsize = 20)
    fig.subplots_adjust(top=0.85)
    plt.ylabel('dORM/dq normalized_RMSE \%')

    axis[0].set_xlabel('Quadrupole')
    axis[1].set_xlabel('Quadrupole')
    axis[0].title.set_text("Vertical direction, Total RMSD = "+f"{vERROR:.4f}\%")
    axis[0].plot(vquadERROR)

    axis[1].title.set_text("Horizontal direction, Total RMSD = "+f"{hERROR:.4f}\%")
    axis[1].plot(hquadERROR)
    plt.savefig("plot.pdf")
    plt.show()
    return