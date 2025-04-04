# -*- coding: utf-8 -*-
"""
@author: Aaron Szasz
"""

import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt

# =============================================================================
# ODMD processing
# =============================================================================

def ODMD_estimate(overlaps, d, tol = 0.1, plot_eigs = False, return_evec = False):
    """
    Takes in a list of overlaps at evenly spaced times n*dt
    Generates the Hankel matrices containing expectation values of successive
      powers of e^{-iH dt} along each anti-diagonal.
    The LHS matrix starts with <e^{-iHt}>, the RHS with <1>
    
    Solves the linear least squares problem to estimate the linear 
        transformation A that transforms the RHS to LHS, and thus predicts
        the dependence of <e^{-iH n dt}> on the {<e^{-iHmdt}>|m<n}
        e^{-iHdt}.  
        
    We return minus the largest phase among eigenvalues of A, which is an
        approximation to the E*dt where E is the ground state energy of H.
        Note that the returned value is not divided by dt!

    Parameters
    ----------
    overlaps : np array or list
        overlap values with constant spacing, eg at t = [0, dt, 2dt, ...]
    tol : float, optional
        Relative threshold for truncating singular values in the linear least
          squares problem
        The default is 0.1.
    plot_eigs : bool, optional
        Examine all eigenvalues returned by generalized eigensolver. 
        The default is False.
    return_evec: bool, optional
        Whether to return a vector that gives an estimate for the ground 
          state eigenvector as a linear combination of the Krylov states
        The default is False.

    Returns
    -------
    float
        Estimated ground state energy of H*dt
        
    if return_evec:
        Also returns a 1D numpy array giving the approximate ground state 
          eigenvector as a linear combination of the Krylov states 

    """
    if np.abs(1-overlaps[0]) > 1e-6: overlaps = np.concatenate( ([1], overlaps) )
    num_cols = len(overlaps) - d
    LHS = np.zeros( (d, num_cols), dtype=complex)
    RHS = np.zeros( (d, num_cols), dtype=complex)
    for col in range(num_cols):
        LHS[:,col] = overlaps[col+1:col+1+d]
        RHS[:,col] = overlaps[col:col+d]
    u,d,v = np.linalg.svd(RHS)
    first_cut = np.where(d > tol*max(d))[0][-1]+1
    u = u[:,:first_cut]
    d = d[:first_cut]
    v = v[:first_cut]
    A = LHS @ v.conj().T @ np.diag(1/d) @ u.conj().T
    e,v = np.linalg.eig(A.T)
    if plot_eigs:
        plt.figure()
        plt.scatter(range(len(e)), -np.angle(e))
        plt.figure()
        plt.scatter(range(len(e)), np.abs(e))
    if return_evec:
        filtered = filter(lambda x : np.abs(x[0]) > 10**-12, zip(e,v.T))
        angle_vector_list = sorted([(np.angle(e),v) for (e,v) in filtered],key=lambda x: x[0])
        return -angle_vector_list[-1][0], angle_vector_list[-1][1]
    else:
        e = np.array(sorted(filter(lambda x : np.abs(x) > 10**-12, e)))
        return -max(np.angle(e))

# =============================================================================
# UVQPE
# =============================================================================

def UVQPE_estimate(overlaps, tol = 0.1, plot_eigs = False, return_evec = False):
    """
    Takes in a list of overlaps at evenly spaced times n*dt
    Generates the Toeplitz matrices U, O.
    O is the overlap matrix, U is the matrix for the operator e^{-iH dt}
    
    Solves the generalized eigenvalue problem to estimate eigenvalues of 
        e^{-iHdt}.  If we take the angles of these eigenvalues, we get 
        eigenvalues of -H*dt, and we return -max(e[-H dt]).  We do not divide
        by dt here!

    Parameters
    ----------
    overlaps : np array or list
        overlap values with constant spacing, eg at t = [0, dt, 2dt, ...]
    tol : float, optional
        Relative threshold for truncating singular values in the generalized
         eigenvalue problem
        The default is 0.1.
    plot_eigs : bool, optional
        Examine all eigenvalues returned by generalized eigensolver. 
        The default is False.
    return_evec: bool, optional
        Whether to return a vector that gives an estimate for the ground 
          state eigenvector as a linear combination of the Krylov states
        The default is False.

    Returns
    -------
    float
        Estimated ground state energy of H*dt
        
    if return_evec:
        Also returns a 1D numpy array giving the approximate ground state 
          eigenvector as a linear combination of the Krylov states

    """
    ns = len(overlaps)
    overlaps = np.array(overlaps)
    ov_mat = np.zeros((ns,ns), dtype=complex)
    for diagonal in range(ns):
        ov_mat += np.diag(overlaps[diagonal]*np.ones(ns-diagonal, dtype=complex),k=diagonal)
    for diagonal in range(1,ns):
        ov_mat += np.diag(overlaps[diagonal].conj()*np.ones(ns-diagonal, dtype=complex),k=-diagonal)
    O = ov_mat[:-1,:-1]
    U = ov_mat[:-1,1:]
    
    if return_evec:
        e,v = get_evals(U, O, ov_thresh = tol, Hermitian=False, return_evecs=True)
    else:
        e = get_evals(U, O, ov_thresh = tol, Hermitian=False)
    
    if plot_eigs:
        plt.figure()
        plt.scatter(range(len(e)), -np.angle(e))
        plt.figure()
        plt.scatter(range(len(e)), np.abs(e))
        
    if return_evec:
        filtered = filter(lambda x : np.abs(x[0]) > 10**-12, zip(e,v.T))
        angle_vector_list = sorted([(np.angle(e),v) for (e,v) in filtered],key=lambda x: x[0])
        return -angle_vector_list[-1][0], angle_vector_list[-1][1]
    else:
        e = np.array(sorted(filter(lambda x : np.abs(x) > 10**-12, e)))
        return -max(np.angle(e))

def get_evals(H_eff, overlaps, ov_thresh=10**-6, check=False, verbose=0, Hermitian=True, return_evecs = False, relative_thresh=True):
    eo,vo = np.linalg.eigh(overlaps)
    if relative_thresh:
        ov_thresh = ov_thresh * np.max(np.abs(eo))
    if ov_thresh < 0:
        ov_thresh = max(-min(np.min(eo),0), np.abs(ov_thresh))
    if True: #min(np.abs(eo)) < ov_thresh: # Singular or poorly conditioned problem
        to_keep = np.where((eo > 0) & (np.abs(eo) >= ov_thresh))[0]
        to_cut = np.where((eo <= 0) | (np.abs(eo) < ov_thresh))[0]
        if verbose > 0:
            print("Cutting",len(to_cut),"components")
        vo = vo[:,to_keep]
        eo = eo[to_keep]
        H_eff_cut = vo.conj().transpose() @ H_eff @ vo
        overlaps_cut = np.diag(eo)
    if Hermitian:
        evals,evecs = sp.linalg.eigh(H_eff_cut, overlaps_cut)#, eigvals_only = True)#, subset_by_index = [0,1])
    else:
        evals,evecs = sp.linalg.eig(H_eff_cut, overlaps_cut)
    # Check solution
    if check:
        errors = np.zeros(len(evals))
        for i in range(len(evals)):
            v = vo @ evecs[:,i]
            errors[i] = np.linalg.norm(H_eff @ v - evals[i]*overlaps @ v) 
        print(errors)
    if return_evecs: return evals, vo @ evecs
    return evals