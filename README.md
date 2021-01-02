# Steps to run

1. Obtain the [Ohio T1DM](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html).
2. In the root of the solution, create a directory `config` and inside of it, directory `ohio`.
3. Put the data it `config/ohio` directory.
4. Run the "Hedia.ipynb" Jupyter notebook.

The current code version utilizes the GPU usage. If you do not have the GPU, go to `evaluateAll.py` and in line 72 change the value assigned to `gpus_per_trial` from 1 to 0.
