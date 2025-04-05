This directory contains the MATLAB scripts used to generate/load data for molecular Hamiltonian and post-process results for ODMD, VQPE, UVQPE, and matrix-pencil (MP) convergence studies.

### Included MATLAB Scripts

- `run_molecule.m` – main script for generating/loading overlap data and running eigensolver benchmarks
- `run_compare.m` - command in `run_molecule.m` for running an algorithm of choice (ODMD/VQPE/UVQPE/MP) on a molecule (H6/LiH/Cr2) using a specified time grid
- `plot_compare.m` - command in `run_molecule.m` for plotting the resulting energy convergence over time across different thresholding parameters
- `generate_phi.m` - command in `run_molecule.m` for generating a reference state with prescribed ground state overlap
- `generate_samples.m` - command in `run_molecule.m` for generating Hamiltonian and overlap matrix elements over time


### Included MATLAB Utilities
- `odmd.m` - contains routines for running ODMD
- `vec2hankel.m.m` - Hankelizes overlap matrix elements to build the data matrices required for ODMD
- `vqpe.m` - contains routines for running VQPE
- `uvqpe.m` - contains routines for running UVQPE
- `mp.m` - contains routines for running MP
- `lam2lamt.m` - scales the Hamiltonian spectrum (for conditioning).
- `lamt2lam.m` - rescales eigenvalue estimates to the original energy scale (inverse of the above)
