The code to run in the example is in simple_demo.ipynb.

To run the code, please make sure to set the following paths:
- Inside simple_demo.ipynb: `DATA_DIR` (the filepath to the `ExampleData` folder; the folder containing all the relevant GEX and morphology data), `CONFIG` (the filepath to the `attempt_1.yaml` file, currently in `Model/configs/attempt_1.yaml`; the file dictating the hyperparameters to GeoAdvAE), and `OUTPUT` (a folder of your choosing where new folders called `logs` and `outputs` will be made)
- Inside the `attempt_1.yaml` file: `prior_matrix_path` (the filepath to the `Corr_matrix.csv` file, currently in `ExampleData/Corr_matrix.csv`; the filepath to the correspondence matrix used by GeoAdvAE to compute the prior cluster alignment penalty)