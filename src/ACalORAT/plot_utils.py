# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 09:03:37 2025

@author: dhuerta

This file contains functions used to draw plots
"""

import matplotlib.pyplot as plt
from matplotlib import rc
import os
from . import math_utils

def plot_both(ORMv, ORMh, nORMv, nORMh, latex = False, title = None):
    
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
    plt.ylabel(r'dORM/dq normalized_RMSE \%')
    fig.title("jholaaa")
    axis[0].set_xlabel('Quadrupole')
    axis[1].set_xlabel('Quadrupole')
    axis[0].title.set_text("Vertical direction, Total = "+f"{vERROR:.4f}"+r"\%")
    axis[0].plot(vquadERROR)

    axis[1].title.set_text("Horizontal direction, Total = "+f"{hERROR:.4f}"+r"\%")
    axis[1].plot(hquadERROR)
    plt.show()
    return

def plot_double(vec1, vec2, title1, title2, title = None, latex = False, yaxis= None):
    """Ploting two generic vectors"""
    
    if latex == True:
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
    fig, axis = plt.subplots(1,2,figsize=(10,5))
    #fig.suptitle("Errors along quadrupoles for the thin formula ", fontsize = 20)
    fig.subplots_adjust(top=0.85)
    if yaxis is not None: plt.ylabel(yaxis)
    axis[0].set_xlabel('BPM')
    axis[1].set_xlabel('BPM')
    axis[0].title.set_text(title1)
    axis[0].plot(vec1)

    axis[1].title.set_text(title2)
    axis[1].plot(vec2)
    plt.show()
    
    
def compare(mA, mB):
    difference = math_utils.normalized_RMSE(mA, mB, 1)
    plt.plot(difference)
    plt.show()
    
def plot_both_Zeus(ORMv, ORMh, nORMv, nORMh, latex = False, SAVE = None, title = None):
    
    vquadERROR = math_utils.errorZeus(ORMv, nORMv)
    vERROR = math_utils.normalized_RMSE(ORMv, nORMv,(0,1,2))

    hquadERROR = math_utils.errorZeus(ORMh, nORMh)
    hERROR = math_utils.normalized_RMSE(ORMh, nORMh, (0,1,2))
    #Creating the plot Zeus asked me to:
       
    if latex == True:
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
    fig, axis = plt.subplots(1,2,figsize=(13,3))
    
    if title is not None: fig.suptitle("Errors along quadrupoles for the thin formula ", fontsize = 20)
    fig.subplots_adjust(top=0.85)
    plt.ylabel(r'dORM/dq normalized_RMSE \%')

    axis[0].set_xlabel('Quadrupole')
    axis[1].set_xlabel('Quadrupole')
    axis[0].title.set_text("Vertical direction, Total RMSD = "+f"{vERROR:.4f}"+r"\%")
    axis[0].plot(vquadERROR)

    axis[1].title.set_text("Horizontal direction, Total RMSD = "+f"{hERROR:.4f}"+r"\%")
    axis[1].plot(hquadERROR)
    if SAVE is not None:
        plt.savefig("plot.pdf")
    plt.show()
    return

matrix = [0]
SAVE = "a" 
def rainbow_plot(matrix, SAVE=None, NAME = None):
    """
    Plots the rows of a matrix each one with a different color from the rainbow.
    """
    total = len(matrix)
    
    cm = plt.get_cmap('rainbow')
    for i, m in enumerate(matrix):
        plt.plot(m, color = cm(i/total))
    
    
    if SAVE is not None:
        "Guarda la gràfica"
        if not os.path.isdir(SAVE):
            os.mkdir(SAVE)   
        try: 
            plt.savefig(SAVE / NAME)
        except:
            raise MemoryError("No s'ha pogut guardar correctament")
    plt.show()
    return

    
    
    
    
    