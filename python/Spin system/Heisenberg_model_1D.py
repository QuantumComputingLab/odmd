# -*- coding: utf-8 -*-
"""
@author: Aaron Szasz 
"""

import numpy as np
from functools import reduce
import time
import pickle
import os

Id = np.eye(2,dtype=int)
Sz = np.array([[1,0],[0,-1]])
Sx = np.array([[0,1],[1,0]])
iSy = np.array([[0,1],[-1,0]])

ops = {'X':Sx, 'Z':Sz, 'Id':Id}
two_site_ops = {'XX':Sx, 'ZZ':Sz, '-YY':iSy, 'IdId':Id}

def generate_H(N, J=1, Delta=1, h=0, periodic = False):
    """
    Returns the full 2^N x 2^N matrix giving the Heisenberg model
    
    H = J*\sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Delta*Z_i Z_{i+1}) 
        - h*\sum_i Z_i

    Parameters
    ----------
    N : int
        Number of qubits/spins
    J : float, optional
        Spin-spin interaction strength. 
        The default is 1.
    Delta : float, optional
        Spin interaction anisotropy. 
        The default is 1.
    h : float, optional
        Magnetic field strength. 
        The default is 0.
    periodic : bool, optional
        Put the model on a ring if True, else on an open chain
        The default is False.

    Returns
    -------
    H : np.ndarray(float)
        Full Hamiltonian matrix
    """
    XX = generate_two_site('XX', N, periodic = periodic)
    mYY = generate_two_site('-YY', N, periodic = periodic)
    ZZ = generate_two_site('ZZ', N, periodic = periodic)
    Z = generate_onsite('Z', N)
    H = J*(XX - mYY + Delta*ZZ) - h*Z
    return H


def construct_two_site(op,N, bonds = None, periodic = False):
    """ 
    Construct full matrix for the operator corresponding to 
        op \otimes op
    On each bond between adjacent sites (taking into account periodicity)
    
    Parameters:
        `N` is the number of sites
    
        `op` should be a 2x2 Hermitian matrix
    
        if `bonds` is specified, only puts (op,op) on the specified bonds
        -> should be given as a list of the starting site, with the bond assumed
             to be with the next site in the chain
    """
    total_op = np.zeros((2**N,2**N))
    if bonds is None:
        if periodic: bonds = range(N)
        else: bonds = range(N-1)
    for bond in bonds:
        term = [Id]*N
        term[bond] = op
        term[(bond + 1)%N] = op
        term = reduce(np.kron,term)
        total_op += term
    return total_op

def construct_onsite(op, N):
    """
    Construct full matrix for the operator corresponding to the tensor product
    of `op` on each site
    
    Parameters:
        `N` is the number of sites
    
        `op` is a 2x2 Hermitian matrix
    """
    # Single-site operator, summed over all sites
    total_op = np.zeros((2**N,2**N))
    for site in range(N):
        term = [Id]*N
        term[site] = op
        term = reduce(np.kron,term)
        total_op += term
    return total_op

def generate_onsite(op_name, N, folder = 'ED_mats_H1D/'):
    # Wrapper on `construct_onsite` that saves the result to disk
    #  and only regenerates if needed
    if not os.path.isdir(folder): os.mkdir(folder)
    mat_filename = folder+op_name+'_N_'+str(N)
    mat_filename += '_ndarray.pkl'
    if os.path.isfile(mat_filename):
        with open(mat_filename, 'rb') as file:
            data = pickle.load(file)
            total_op = data[op_name]
    else:
        op = ops[op_name]
        t0 = time.time()
        total_op = construct_onsite(op, N)
        print("Finished "+op_name+":", time.time() - t0)
        with open(mat_filename, 'wb') as file:
            pickle.dump({op_name:total_op}, file)
    return total_op    
            
def generate_two_site(op_name, N, bonds=None, folder = 'ED_mats_H1D/', periodic=False):
    # Wrapper on `construct_two_site` that saves the result to disk
    #  and only regenerates if needed
    if not os.path.isdir(folder): os.mkdir(folder)
    mat_filename = folder+op_name+'_N_'+str(N)+'_triangles_'+str(bonds)+'_periodic_'+str(periodic)
    mat_filename += '_ndarray.pkl'
    if os.path.isfile(mat_filename):
        with open(mat_filename, 'rb') as file:
            data = pickle.load(file)
            total_op = data[op_name]
    else:
        op = two_site_ops[op_name]
        t0 = time.time()
        total_op = construct_two_site(op, N, bonds=bonds, periodic=periodic)
        print("Finished "+op_name+":", time.time() - t0)
        with open(mat_filename, 'wb') as file:
            pickle.dump({op_name:total_op}, file)
    return total_op

