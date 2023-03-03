#####################################################################
#This file contains the networks that learn coordinates and stresses#
#####################################################################
from Datahandler import *
import numpy as np
from NetworkClass import Network
from Lossfunctions import *
from Config import Config
from Config import Data
import os
import logging


#set all the seeds
set_seed(1)
tf.get_logger().setLevel(logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#initialize
variables = ['Von_Mises', 'Stress_xx', 'Stress_yy', 'Stress_zz', 'Stress_xy']
coords = ['x', 'y']
Data.N_variables = len(variables)
N_points_in_element = 4
Data.test_index = np.arange(int(3))
np.random.shuffle(Data.test_index)

#save training data if necessary
if Config.save_data:        #if save data is set to true all the training data will be saved in numpy arrays
    clear_and_print('[Saving data to folder]')
    save_training_data(coords, Config.filename, 'coordinates_forward')
    save_training_data(variables, Config.filename, 'stress_forward')
    save_training_data(['Flowstress'], Config.filename, 'flowstress')

#################################################################################################################################################################
#stresses network training and saving#
#################################################################################################################################################################

#load data from folders
clear_and_print('[Loading data from folder]')
x, y, Importer = load_training_data('stress_forward')
flowstress = np.genfromtxt('Save_data/flowstress/y.csv', delimiter=',').T

#delete simulations with NaN values, and get the right sequence
x, y, flowstress = delete_nan(x, y, flowstress)

#remove other faulty simulations (some have all 0 as input)
del_index = np.where(y[:, 0]==0)
x, y, flowstress = np.delete(x, del_index, axis=0), np.delete(y, del_index, axis=0), np.delete(flowstress, del_index, axis=0)

#only the noise/process parameters
x = x[:, [2, 3, 4, 5, 6, 7, 8, 9]]

#save important variables
save_variables(x, y, Importer)

#initialize some more
N_elements = 1
elements = np.linspace(0, N_elements-1, N_elements, dtype = int)
errors_PINN, errors_DNN  = np.zeros([25, N_elements]), np.zeros([25, N_elements])
intervals = np.zeros([2*Data.N_variables*N_points_in_element, N_elements])

#loop over all elements, train a neural network for all elements, compare PINN with DNN.
for i in range(N_elements):

    #Split training/test
    Data.i = i
    x_train, x_test, y_train, y_test, flowstress_train = split_training_test(x, y, flowstress, Config.validation_ratio)

    #normalize
    x_train_norm, x_interval = fit_transform(x_train)
    x_test_norm = transform(x_test, x_interval)
    Data.x_interval = x_interval

    #Split data for each element and normalize y
    flowstress_train_elements = np.split(flowstress_train, Data.N_elements, axis=1)
    y_train_elements = np.split(y_train, Data.N_elements, axis=1)
    y_test_elements = np.split(y_test, Data.N_elements, axis=1)
    Data.flowstress = np.array(flowstress_train_elements[elements[i]])
    y_train_norm, Data.interval = fit_transform(y_train_elements[elements[i]])
    y_test_norm = transform(y_test_elements[elements[i]], Data.interval)

    #initialize DNN
    NN = Network(N_input=Data.N_features, N_output=len(variables)*N_points_in_element, N_steps=Config.N_steps,   \
    label=x_train_norm, data=y_train_norm, PINN=False, variables=variables, print_data=True)
    NN.init_model(Config.layers)
    NN.set_optimizer(Config.initial_learning_rate, Config.decay_steps, Config.decay_rate)

    #train DNN
    NN.train(get_loss, y_test_norm, x_test_norm, Data.interval)

    #initialize PINN
    PINN = Network(N_input=Data.N_features, N_output=len(variables)*N_points_in_element, N_steps=Config.N_steps,   \
    label=x_train_norm, data=y_train_norm, PINN=True, variables=variables, print_data=True)
    PINN.init_model(Config.layers)
    PINN.set_optimizer(Config.initial_learning_rate, Config.decay_steps, Config.decay_rate)

    #train PINN
    PINN.train(get_loss, y_test_norm, x_test_norm, Data.interval)

    #save PINN model
    if not os.path.exists("Models"):
        os.mkdir("Models")

    PINN.model.save("Models/model_" + str(elements[i]) + ".h5")







