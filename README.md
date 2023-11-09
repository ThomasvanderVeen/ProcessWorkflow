# Process Workflow

## Table of Contents
1. [Introduction](#introduction)
2. [Start guide](#start-guide)
3. [Config file formatting](#config-file-formatting)


## Introduction
Finite element analysis (FEA) data is combined with artificial neural networks (ANNs) and physics informed neural networks (PINNs) in the process workflow.
The process workflow (written in Python) is a six-step workflow developed to optimize root cause detection in defective end products and works as follows:

1. Train and validate the parameter predictor.
  The parameter predictor is a feedforward neural network responsible for estimating the process parameters from the contour geometry of a faulty product.
2. Train and validate the product predictor.
  The product predictor estimates the geometry and stresses of the product after each forming step for a given set of process parameters.
3. Identify the relevant features of the defective products from the assembly line.
  The quality of the product is tested on the production line. If a product is faulty, its relevant features need to be measured to be used in the next step. In
  the example we study in this paper, the contour geometry of the final product is the relevant feature.
4. Estimate the process parameters for the faulty sample.
  This step yields a prediction for the process parameters that have resulted in the faulty product.
5. Predict the product’s features.
  This step confirms that the process parameters estimated in step (iv) result in the same geometry as the faulty product. In addition, the stress state throughout the product can be estimated for further analysis.
6. Identify the root cause(s) of the defective product.
  The final step of the workflow is to take the estimated process parameters and manipulate them to identify the one responsible for the defective product.


## Start-guide
The Process Workflow is written in python (3.10) on a windows 10 PC not utilizing the graphics card.

To run the code, the following files are necessary:
  1. .issn (calculated parameters in each node) & .mshn (node information) files of each simulation in the 'Simulations' folder provided by MSC marc (or other FEA software that provides these files)
  2. .ida (material model) file in the 'Variation_and_material/material' folder.
  3. .xslx (variaton scheme) file in the 'Variation_and_material/variation_scheme' folder.

A few examples are already provided to allow the user to run the code as is.

In the file "ProcessPredictor", running ProcessPredictor.py will fit a model that predicts process parameters from product geometries & running Predict.py will evaluate the model for a single product geometry.

In the file "ProductPredictor", running ProductPredictorGeometry.py will fit a model that predicts product geometries from process parameters & will fit a model that predicts product stresses from process parameters

All models are saved in .h5 format

## Config-file-formatting

There is a config file in Both "ProcessPredictor" & "ProductPredictor" folders, which allows the user to change parameters.

[General]
Save_data => True -> will save data (slower), False -> will use saved data (quicker)
Last_step_name => Name of last step in FEA, simulations with this name will be imported (often the last step)
Boundary_index => The amount of total simulations (n_test + n_train)
use_variable_index => specify which columns of the variation scheme should be used

[Network]
Architecture => number of layers and neurons in each layer
N_epochs => the number of maximum epochs during the learning stage
Learning_rate => the learning rate for the ANN
Testing_ratio => the ratio between the number of testing and training simulations

[PINN]
N_collocation_points => the amount of collocation points for the physics informed neural network (PINN)

