# -*- coding: utf-8 -*-
"""
@author: Aaron Szasz
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

import Heisenberg_model_1D as H1d
import classical_post_processing as cpp

default_database_folder = 'overlap_data/'

def d2b(d, N):
    # Convert the (decimal) state index d to an N-qubit bitstring
    return [ (d//2**j)%2 for j in range(N)]

def b2d(b, N):
    # Convert an N-qubit bitstring b into a decimal index
    return sum([b[j]*(2**j) for j in range(N)])

def run():
    """
    Run this to generate data and figures corresponding to the right panel of
      Fig 5
    """
    generate_overlaps_for_database_exact(n_steps=250, use_cat = False)
    generate_overlaps_for_database_exact(n_steps=250, use_cat = True)
    make_eigenvector_convergence_fig(max_steps=250)
    convert_residuals_to_dat()
    # Done!

def generate_overlaps_for_database_exact(n_steps = 100, database_folder = default_database_folder, use_cat = True):
    """
    Generates overlap matrix elements
    
        <psi0 | e^{-iH n dt} | psi0>
        
    where:
        - psi0 is the initial state.  If not `use_cat`, psi0 is the state that 
           alternates |up,down,up,down, ...>, or in qubit language 
           |0,1,0,1,...>.  (Note that we require the number of qubits to be 
           even.)  If `use_cat`, psi0 is instead the superpostion
           (|0101...> + |1010...>)/sqrt(2)
        - H is the 1D Heisenberg model J*sum XX+YY+ZZ, with J=1 and periodic
           boundary conditions.
        - dt is some small time step
        - n specifies the total time by giving the number of time steps
        
    We specifically compute using:
        - N=8 qubits, and a time step size dt = 0.15
        - N=12 qubits, and a time step size dt = 0.1
        
    The time evolution is computed exactly, with no noise and no Trotter 
      approximation
      
     Results are stored as a dictionary with the following structure:
         [Number of qubits] : {'H': (e,v),
                               'E0': E0,
                               'times': times,
                               'states': states,
                               'S_eff': S_eff,
                               'H_eff': H_eff}
         
         `H`: (e,v), the eigenvalues and vectors of H as returned by np.eigh
         `E0`: the ground state energy of H
         `times`: the list of times {n dt} for matrix elements
         `states` stores the time-evolved states e^{-iH n dt}|psi0>
         `S_eff` stores the overlap matrix for UVQPE.  This is a Hermitian
           Toeplitz matrix whose top row contains the entries
           1, <e^{-iHdt}>, <e^{-2iHdt}>, ..., <e^{-niHdt}>
         `H_eff` stores the effective matrix for H in the subspace defined by
           the time-evolved states.  This is also a Hermitian Toeplitz matrix,
           with top row entries
           <H>, <H e^{-iHdt}>, <H e^{-2iHdt}>, ..., <H e^{-niHdt}>
         
         This dictionary is both returned and saved to disk
    
    Parameters
    ----------
    n_steps : int
        Number of time steps at which to compute the matrix elements (max value
          of n, plus 1)
    
    database_folder : str
        Where to store the computed results

    Returns
    -------
    database : dict
        Dictionary as described above
    """
    
    if use_cat:
        filename = database_folder + 'overlaps_data.pkl'
    else:
        filename = database_folder + 'overlaps_data_psi0_product.pkl'
    
    data = {}
    
    for N,dt in [(8,0.15),(12,0.1)]:
        print("On N:", N)
        data[N] = {}
        H = H1d.generate_H(N, periodic=True)
        e,v = np.linalg.eigh(H)
        data[N]['H'] = (e,v)
        data[N]['E0'] = e[0]

        times = dt*np.arange(n_steps+1, dtype=float)
        times = np.round(times, 4)
        data[N]['times'] = times
        
        S_eff = np.zeros(times.shape, dtype=complex)
        H_eff = np.zeros(times.shape, dtype=complex)
        states = [None for _ in range(len(times))]
        
        state = generate_initial_state(N, use_cat=use_cat)
        states[0] = state
        S_eff[0] = 1.
        decomposition = v.conj().T @ state
        H_eff[0] = decomposition.conj() @ (e*decomposition)
        
        for t_idx, t in enumerate(times[1:],1):
            timeEvolved = np.exp(-1j*t*e) * decomposition
            final_state = v @ timeEvolved
            states[t_idx] = final_state
            S_eff[t_idx] = state.conj() @ final_state
            
            H_eff[t_idx] = decomposition.conj() @ (e * timeEvolved)
            
        data[N]['states'] = states
        data[N]['S_eff'] = S_eff
        data[N]['H_eff'] = H_eff

    os.makedirs(database_folder, exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
            
    return data
     

def generate_initial_state(N, use_cat = True):
    # Generates initial state psi0 as described in docstring for
    #  `generate_overlaps_for_database_exact`
    assert N%2 == 0
    v = np.zeros(2**N)
    v[b2d([0,1]*(N//2), N)] = 1
    if use_cat:
        if N%4 == 0:
            v[b2d([1,0]*(N//2), N)] = 1
        else:
            v[b2d([1,0]*(N//2), N)] = -1
        return v/np.sqrt(2)
    else:
        return v

def make_eigenvector_convergence_fig(max_steps = 20):
    """
    NOTE: before running this function, you must run
    - `generate_overlaps_for_database_exact(<n_steps>)`
    - `generate_overlaps_for_database_exact(<n_steps>, use_cat = False)`
    where `<n_steps>` is some number >= `max_steps`
    
    Uses data generated in `generate_overlaps_for_database_exact` to run ODMD, 
    getting estimates of the ground state energy for different numbers of time 
    steps/Krylov vectors used, up to the maximum, `max_steps`
    
    The procedure is as follows:
        1) Read in data stored by the previous function 
        2) Take just the real part of the matrix elements stored in the field 
             'S_eff' (see above).  
        3) Add noise to the overlap matrix elements.  The noise has magnitude  
             drawn from a normal distribution of width 0.01 (independent of the  
             size of the overlap or which time step we are on).  
        4) For each number of time steps from 0 through max_steps-1:
             - Run ODMD function in `classical_post_processing` to find the 
               approximate ground state eigenvalue and eigenvector.  We use
               a noise filtering threshold of 0.1
             - Compute the overlap with the true ground state
             - Compute the residual norm
             
        Finally, we plot the results and save the gs overlaps and residual
          norms to disk
    """
    
    with open('overlap_data/overlaps_data_psi0_product.pkl','rb') as file:
        product_data = pickle.load(file)
    with open('overlap_data/overlaps_data.pkl','rb') as file:
        cat_data = pickle.load(file)
        
    print('Exact energies')
    print('8:', product_data[8]['H'][0][0])
    print('12:', product_data[12]['H'][0][0])
        
    def compute_residual_norm(H, state, E_estimate):
        (e,v) = H
        return np.linalg.norm(v @ (e * ( (v.conj().T) @ state) ) - E_estimate * state )
    
    def compute_groundstate_overlap(H, state):
        (e,v) = H
        return 1 - np.abs(v[:,0].conj() @ state)**2
    
    def add_noise(overlaps, width):
        noise = np.random.normal(0, width, len(overlaps))
        if overlaps.dtype == complex:
            noise = noise * np.exp(1j*2*np.pi*np.random.rand(len(overlaps)))
            overlaps = overlaps + noise
            angles = np.angle(overlaps)
            norms = np.abs(overlaps)
            norms[norms > 1] = 1
            overlaps = norms * np.exp(1j*angles)
            overlaps[0] = 1
            return overlaps
        else:
            overlaps = np.real(overlaps) + noise
            overlaps[0] = 1.
            overlaps[overlaps < -1] = -1
            overlaps[overlaps > 1] = 1
            return overlaps
    
    residuals = np.zeros((4, max_steps))
    residuals[:,:1] = np.nan
    
    gs_overlaps = np.zeros((4, max_steps))
    gs_overlaps[:,:1] = np.nan
    
    def add_data(idx, data, tol = 0.1, noise = 0.01):
        print("Adding data")
        H = data['H']
        states = data['states']
        S_eff = data['S_eff']
        print(S_eff)
        S_eff = np.real(S_eff)
        S_eff = add_noise(S_eff, noise)
        print(S_eff)
        dt = data['times'][1] - data['times'][0]
        for step in range(1,max_steps):
            overlaps = S_eff[:step+1]
            e,v = cpp.ODMD_estimate(overlaps, (step+1)//2, tol = tol, return_evec = True)
            print(e/dt)
            state = sum([coeff*phi_i for (coeff,phi_i) in zip(v, states[:step+1]) ])
            state = state/np.linalg.norm(state)
            residuals[idx, step] = compute_residual_norm(H, state, e/dt)
            gs_overlaps[idx, step] = compute_groundstate_overlap(H, state)
        # Divide residual by 2-norm
        residuals[idx] /= max(abs(H[0][0]),abs(H[0][-1]))
            
    add_data(0, product_data[8])
    add_data(1, product_data[12])
    add_data(2, cat_data[8])
    add_data(3, cat_data[12])
    
    f,a = plt.subplots()
    a.scatter(range(max_steps), residuals[0], label = '8, product')
    a.scatter(range(max_steps), residuals[1], label = '12, product')
    a.scatter(range(max_steps), residuals[2], label = '8, cat')
    a.scatter(range(max_steps), residuals[3], label = '12, cat')
    a.set_yscale('log')
    a.legend()
    
    g,a = plt.subplots()
    a.scatter(range(max_steps), gs_overlaps[0], label = '8, product')
    a.scatter(range(max_steps), gs_overlaps[1], label = '12, product')
    a.scatter(range(max_steps), gs_overlaps[2], label = '8, cat')
    a.scatter(range(max_steps), gs_overlaps[3], label = '12, cat')
    a.set_yscale('log')
    a.legend()
    
    to_save_residuals = {'8 product' : residuals[0],
                         '12 product' : residuals[1],
                         '8 cat' : residuals[2],
                         '12 cat' : residuals[3]
                         }
    to_save_overlaps = {'8 product' : gs_overlaps[0],
                        '12 product' : gs_overlaps[1],
                        '8 cat' : gs_overlaps[2],
                        '12 cat' : gs_overlaps[3]
                        }
    to_save = {'residual_norm': to_save_residuals, 'gs_overlap': to_save_overlaps}
    
    with open(default_database_folder + 'state_convergence.pkl', 'wb') as file:
        pickle.dump(to_save, file)
    
    return residuals

def convert_residuals_to_dat():
    # Converts saved GS overlap and residual norm data to a different file
    #   format
    with open(default_database_folder + 'state_convergence.pkl', 'rb') as file:
        data = pickle.load(file)
        
    for data_type in ('residual_norm','gs_overlap'):
        to_write_str = ''
        for idx, (c8,p8,c12,p12) in enumerate(zip(data[data_type]['8 cat'],data[data_type]['8 product'],data[data_type]['12 cat'],data[data_type]['12 product'])):
            if np.isnan(c8):
                continue
            to_write_str += str(idx)+' '+str(c8)+' '+str(p8)+' '+str(c12)+' '+str(p12)
            to_write_str += '\n'
        
        with open(default_database_folder + 'Heisenberg_'+data_type+'_norm.dat','w') as file:
            file.write(to_write_str)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
        