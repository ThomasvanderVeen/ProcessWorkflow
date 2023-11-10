########################################################
#This file contains the networks that learn coordinates#
########################################################
from Datahandler import *
import numpy as np
from NetworkClass import Network
from Lossfunctions import *
from Config import Config
from Config import Data
import os
import logging

#set seeds
set_seed(1)
tf.get_logger().setLevel(logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#initialize
variables = ['Flowstress', 'Von_Mises', 'Stress_xx', 'Stress_yy', 'Stress_zz', 'Stress_xy']
coords = ['x', 'y']
Data.N_variables = len(variables)
Data.testlabel = np.absolute(np.random.rand(Config.N_collocation_points, 8))
N_points_in_element = 4

#################################################################################################################################################################
#Save#
#################################################################################################################################################################
#define variables for neural network
if Config.save_data:        #if save data is set to true all the training data will be saved in numpy arrays
    clear_and_print('[Saving data to folder]')
    save_training_data(coords, Config.filename, 'coordinates_forward')
    save_training_data(variables, Config.filename, 'stress_forward')
    
#################################################################################################################################################################
#coordinates network training and saving#
#################################################################################################################################################################

#load the data
x, y, Importer = load_training_data('coordinates_forward')

#remove NaN simulations
x, y = delete_nan(x, y, False)

#remove other faulty simulations
del_index = np.where(y[:, 0]==0)
x = x[:, 2:10]
x, y = np.delete(x, del_index, axis=0), np.delete(y, del_index, axis=0)

#get x and y coordinates
[xcoords, ycoords] = y.T[::2, :], y.T[1::2, :]

#set coordinates with last node as 0,0
for i in range(xcoords.shape[1]):
    ycoords[:, i] = ycoords[:, i] - ycoords[-1, i]
    xcoords[:, i] = xcoords[:, i] - xcoords[-1, i]

#stack the coordinates into a single array
y = np.vstack([xcoords, ycoords]).T
save_variables(x, y, Importer)

#normalize x and y
x, x_interval = fit_transform(x)
y, y_interval = fit_transform(y)

#split training and test
x_train_norm, x_test_norm, y_train_norm, y_test_norm = split_training_test_single(x, y, 0)

#reshape such that network can manage the arrays
x_test_norm = x_test_norm.reshape(1, -1)
y_test_norm = y_test_norm.reshape(1, -1)

#initialize network with all parameters
NN = Network(N_input = Data.N_features, N_output = y_train_norm.shape[1], N_steps = Config.N_steps_coords,   \
label=x_train_norm, data=y_train_norm, PINN=False, variables=coords, print_data=False)
NN.init_model(Config.layers_coords)
NN.set_optimizer(Config.initial_learning_rate_coords, 1, 1)

#train network
NN.train(get_loss, y_test_norm, x_test_norm, y_interval)

#save coordinate model and errors / intervals
NN.model.save("Models/model_xy.h5")
