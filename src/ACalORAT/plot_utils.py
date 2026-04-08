# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 09:03:37 2025

@author: dhuerta

All useful functions to plot the data analized during the project
"""

import matplotlib.pyplot as plt
from matplotlib import rc
import os
from . import math_utils

def plot_both(ORMv, ORMh, nORMv, nORMh, latex = False, title = None):
    """
    Takes the numerical and analytical Jacobians of the ORM both in the 
    horizontal and the vertical transverse dimensions and compares them,
    the expected broadcast dimensions for the arrays are 
    [quadrupole, bpm, corrector] 
    
    It compares them using the standard RMSD among the full ORM.

    Parameters
    ----------
    ORMv : np.array
        numerical reference
    ORMh : np.array
        numerical reference
    nORMv : np.array
        version to compare
    nORMh : np.array
        version to compare
        
    latex : bool, optional
         If true, sets the font of the plot in LaTeX, Computer Modern must be installed. The default is False.
    
    title : TYPE, optional
        If added, the plot has the given title. The default is None.

    Returns
    -------
    None.

    """
    
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
    axis[0].set_xlabel('Quadrupole')
    axis[1].set_xlabel('Quadrupole')
    axis[0].title.set_text("Vertical direction, Total = "+f"{vERROR:.4f}"+r"\%")
    axis[0].plot(vquadERROR)

    axis[1].title.set_text("Horizontal direction, Total = "+f"{hERROR:.4f}"+r"\%")
    axis[1].plot(hquadERROR)
    plt.show()
    return

def plot_both_Zeus(ORMv, ORMh, nORMv, nORMh,  latex = False, SAVE = None, title = None, xlabel = "Quadrupole",):
    """
    Takes the numerical and analytical Jacobians of the ORM both in the 
    horizontal and the vertical transverse dimensions and compares them,
    the expected broadcast dimensions for the arrays are 
    [quadrupole, bpm, corrector].
    
    It compares them with the same metrics Zeus uses.

    Parameters
    ----------
    ORMv : np.array
        numerical reference
    ORMh : np.array
        numerical reference
    nORMv : np.array
        version to compare
    nORMh : np.array
        version to compare
        
    latex : bool, optional
         If True, sets the font of the plot in LaTeX, Computer Modern must be installed. The default is False.
    
    SAVE : pathlib._local.PosixPath
        If not None saves the plot in the indicated SAVE directory as plot.pdf
    
    title : TYPE, optional
        If added, the plot has the given title. The default is None.

    Returns
    -------
    None.

    """
    
    vquadERROR = math_utils.errorZeus(ORMv, nORMv)
    vERROR = math_utils.normalized_RMSE(ORMv, nORMv,(0,1,2))

    hquadERROR = math_utils.errorZeus(ORMh, nORMh)
    hERROR = math_utils.normalized_RMSE(ORMh, nORMh, (0,1,2))
    #Creating the plot Zeus asked me to:
       
    if latex == True:
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
    fig, axis = plt.subplots(1,2,figsize=(10,3))
    
    if title is not None: fig.suptitle(title, fontsize = 20)
    fig.subplots_adjust(top=0.85)
    plt.ylabel(r'dORM/dq normalized_RMSE \%')

    axis[0].set_xlabel(xlabel)
    axis[1].set_xlabel(xlabel)
    axis[0].title.set_text("Vertical direction, Total RMSD = "+f"{vERROR:.4f}"+r"\%")
    axis[0].plot(vquadERROR)

    axis[1].title.set_text("Horizontal direction, Total RMSD = "+f"{hERROR:.4f}"+r"\%")
    axis[1].plot(hquadERROR)
    if SAVE is not None:
        try: 
            plt.savefig( SAVE / "plot.pdf")
        except:
            raise MemoryError("Could not save successfully (check if the folder exists)")
    plt.show()
    return

def plot_double(vec1, vec2, title1, title2, title = None, latex = False, yaxis= None):
    """Ploting two generic vectors to compare them"""
    
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
    """Plots the difference between two vectors"""
    difference = math_utils.normalized_RMSE(mA, mB, 1)
    plt.plot(difference)
    plt.show()
    

def rainbow_plot(matrix, SAVE=None, NAME = None):
    """
    Plots the rows of a matrix each one with a different color from the rainbow.
    """
    total = len(matrix)
    
    cm = plt.get_cmap('rainbow')
    for i, m in enumerate(matrix):
        plt.plot(m, color = cm(i/total))
    
    
    if SAVE is not None:
        if not os.path.isdir(SAVE):
            os.mkdir(SAVE)   
        try: 
            plt.savefig(SAVE / NAME)
        except:
            raise MemoryError("Could not save successfully")
    plt.show()
    return

    
    
    
    
    