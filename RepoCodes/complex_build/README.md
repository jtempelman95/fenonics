To execute the files in this folder, you will need a working installation of `dolfinx`
and `dolfinx_mpc` based on a complex build of PETSc. You can obtain a working
environment via conda using the `environment.yaml` file in this directory. To install
the necessary packages, first make sure you have conda installed on your system, then
navigate to the location where you have downloaded `environment.yaml` and run

```bash
conda env create -f environment.yml
```

This will create a new environment named `dolfinx_complex` on your system.
