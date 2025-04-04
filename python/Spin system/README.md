This directory contains the Python scripts used to generate data for Heisenberg spin Hamiltonian and post-process results for ODMD, VQPE, UVQPE, and QCELS convergence studies.

### Included Python Scripts

- `get_overlaps_and_convergence_metrics_Fig5.py` – main script for generating overlap data and running eigensolver benchmarks  
- `Heisenberg_1D.py` – defines the Heisenberg spin chain model  
- `classical_post_processing.py` – contains routines for analyzing and formatting simulation output  


### Reproducing the Python Script Data

To regenerate the input data, run the following from `get_overlaps_and_convergence_metrics_Fig5.py`:
```
python
generate_overlaps_for_database_exact(n_steps=250, use_cat=True)   # generates 'superposition' data  
generate_overlaps_for_database_exact(n_steps=250, use_cat=False)  # generates 'product' data

----------------------------------------------------------------------------------------------------
To run ODMD and UVQPE convergence analysis (with noise added), execute:

python
make_eigenvector_convergence_fig(max_steps=250)

[Note: This step introduces random noise, so the resulting numerical values will vary between runs.]

----------------------------------------------------------------------------------------------------
To save the output in the format used for plotting, execute:

python
convert_residuals_to_dat()

----------------------------------------------------------------------------------------------------
To perform the full data generation and analysis as described, execute:

python
run()
```

### Included Interactive Python Notebooks

- `ODMD_utilities.ipynb` – contains routines for running ODMD 
- `VQPE_utilities.ipynb` – contains routines for running VQPE and UVQPE
- `QCELS_utilities.ipynb` – contains routines for running QCELS
- `compare_methods_spin_Fig5.ipynb` – compares VQPE, UVQPE, and QCELS for the 1d Heisenberg model with data generated above




