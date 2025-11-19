The code to run in the example is in simple_demo.ipynb.

To run the code, please make sure to set the following paths:
- Inside simple_demo.ipynb: `DATA_DIR` (the filepath to the `ExampleData` folder; the folder containing all the relevant GEX and morphology data), `CONFIG` (the filepath to the `attempt_1.yaml` file, currently in `Model/configs/attempt_1.yaml`; the file dictating the hyperparameters to GeoAdvAE), and `OUTPUT` (a folder of your choosing where new folders called `logs` and `outputs` will be made)
- Inside the `attempt_1.yaml` file: `prior_matrix_path` (the filepath to the `Corr_matrix.csv` file, currently in `ExampleData/Corr_matrix.csv`; the filepath to the correspondence matrix used by GeoAdvAE to compute the prior cluster alignment penalty)

Since this is minimalistic simulation, some of the clustering `.csv` files are not immediately discernible. Here is a brief description of the three clustering `.csv` files and their role in a typical usage of GeoAdvAE run:
- `GEX_CLUSTER_PATH`: This is a broad clustering of cells based on GEX.
- `MORPHO_CLUSTER_PATH`: This is a broad clustering of the cells based on morphology.
- `RNA_FAMILY_PATH`: This is the "true" cell-type label, used only for the 1-NN accuracy measure. This is not used in the training of GeoAdvAE itself, but only used for model diagnostics.

Both ``GEX_CLUSTER_PATH` and `MORPHO_CLUSTER_PATH` are the clustering assignments used in the correspondence matrix in `Corr_matrix.csv`.