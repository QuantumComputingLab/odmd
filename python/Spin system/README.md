This directory contains the Python scripts used to generate data for Heisenberg spin model and post-process results for ODMD and UVQPE convergence studies.

### Included Scripts

- `get_overlaps_and_convergence_metrics.py` – main script for generating overlap data and running eigensolver benchmarks  
- `Heisenberg_1D.py` – defines the Heisenberg spin chain model  
- `classical_post_processing.py` – contains routines for analyzing and formatting simulation output  



### Reproducing the Data

To regenerate the input data, run the following from `generate_overlaps_classical_H1D_v2.py`:

```python
generate_overlaps_for_database_exact(n_steps=250, use_cat=True)   # generates 'superposition' data  
generate_overlaps_for_database_exact(n_steps=250, use_cat=False)  # generates 'product' data

----------------------------------------------------------------------------------------------------
To run ODMD and UVQPE convergence analysis (with noise added), execute:

```python
make_eigenvector_convergence_fig(max_steps=250)

[Note: This step introduces random noise, so the resulting numerical values will vary between runs.]

----------------------------------------------------------------------------------------------------
To save the output in the format used for plotting, run:

```python
convert_residuals_to_dat()





