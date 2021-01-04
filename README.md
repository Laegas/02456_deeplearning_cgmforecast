# Blood Glucose Forecasting Based on CGM Data

Authors:

- Gustaw Żyngiel
- Krystof Spiller
- Veniamin Tzingizis 

## Steps to run

1. Obtain the [Ohio T1DM](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html).
2. In the root of the solution, create a directory `config` and inside of it, directory `ohio`.
3. Put the data it `config/ohio` directory.
4. Run the "Hedia.ipynb" Jupyter notebook.

**IMPORTANT**: The current code version utilizes the GPU usage which is required to run the code.

## Recommended environment

The code has been developed using the Google Colab and we recommend that environment for launching the code.

1. Upload the `02456_deeplearning_cgmforecast` (with the added data as stated in the *Steps to run* section of README) directory to Google drive.
2. Open the `Hedia.ipynb` in Google Colab.
3. Enable GPU by selecting from the menu `Runtime`->`Change runtime type`->select `GPU`->`Save`.
4. Mount your drive as described in the first cell of the `Hedia.ipynb`.
5. Run the `Hedia.ipynb` Jupyter notebook.

If you have troubles using Google Colab, we recommend to instead run the python scripts locally: in the root of the solution, after adding the datasets, run `python evaluateAll.py`.

## Notes

The model can be found in `src/models` directory. The file `best_model_2314.py` is the one containing the model talked about in the paper.

The results can be found in the `results/` directory.
