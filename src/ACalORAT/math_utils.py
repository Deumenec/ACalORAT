# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 11:21:04 2025

@author: dhuerta
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def symetric_error(arrA, arrB, dims):
    """
    Calculates the symetric error for all elements along a dimension.

    Parameters
    ----------
    arrA : np.array
        reference array
    arrB : np.array
        array that is compared to the refecence
    dim : np.array
        contracted dimensions

    Returns
    -------
    comp : np.array
        array containing the comparison
    """
    if (type(dims) != int): 
        norm = 1
        for dim in dims:
            norm *= arrA.shape[dim]
    else: norm = arrA.shape[dims]
    print(norm)
    comp = 2*np.abs(arrA-arrB)/(np.abs(arrA)+np.abs(arrB))*100
    comp = np.average(comp, axis = dims)/norm
    return comp

def errorZeus(dORMA, dORMB):
    """Dos np.arrays 1ra de referència i 2na de comparació"""
    return np.average(np.std(dORMA-dORMB, axis = 1)/np.std(dORMA, axis = 1), axis = 1)*100
def normalized_RMSE(arrA, arrB, dims):
    """Calculates the RMSD as a %
    """
    rmse = np.sqrt( np.sum((arrA-arrB)**2, axis = dims))
    nrmse = rmse/ np.sqrt(np.sum(arrA**2, axis=dims))
    return nrmse*100
    
    
def tensorComparison(tensor1, tensor2 , dimensions):
    """

    Parameters
    ----------
    tensor1 : np.array
    tensor2 : np.array
    dimension : list 
        Dimensions along which we contract 0 for bpms, 1 for correctors and 2 for quadrupoles
    Returns
    -------
    1D array with the square differences among the arrays
    """
    diference = tensor1-tensor2
    norm = 1 
    if (type(dimensions) != int): 
        for dim in dimensions:
            norm *= tensor1.shape[dim]
    else: norm = tensor1.shape[dimensions]
    vec = np.sum(np.abs(diference), axis = dimensions)
    return (vec/norm)


def listPlot(vectors, vecNames,title ,savename):
    """ Creates a plot commparing the components of vectors of equal length"""
    colors = plt.cm.rainbow(np.linspace(0, 1, len(vectors)))
    for i, vector in enumerate(vectors):
        plt.plot(vector, color = colors[i], label= vecNames[i])
    plt.legend()
    plt.title(title)
    #plt.savefig(os.path.join("plots", savename), format = "pdf")
    plt.show() #Això ha de ser l'últim perqué es peta la figura
    return

