################################################################################
#this file holds all the functions that handles the training and validation set#
################################################################################
import numpy as np
import os 
from Dataholder import DataHolder
from Functions import *
import pandas as pd
from Config import Config
from Config import Data

#Combines all simulations and save the training data onto the hard drive
def save_training_data(variables, foldername, savename):
    #Initialize the class that imports the data from all the files
    Importer = DataHolder()

    #get the variation scheme values and process names from the files
    variation_string = os.listdir('Variation_and_material/Variation_scheme')[0]  
    variation_scheme = np.array(pd.read_excel('Variation_and_material/Variation_scheme/' + str(variation_string), engine='openpyxl'))[:, :].T

    #get all useful variables
    fnames = os.listdir('Simulations')     
    Data.N_samples, Data.N_variables = len(fnames), len(variables)     
    Importer.initialize(fnames[0], foldername)
    Data.N_nodes = int(np.size(Importer.prepare_data(variables))/Data.N_variables)
    y = np.empty([Data.N_nodes*Data.N_variables, Data.N_samples])
    delete_list = []

    #get all coordinates, only if the amount of nodes is = to first simulations (basically remove all the remeshed simulations)
    for i in range(Data.N_samples):
        Importer.initialize(fnames[i], foldername)
        try:
            y[:, i] = Importer.prepare_data(variables)
        except:
            delete_list.append(i)

    #check if the folder exists, otherwise create it
    if not os.path.exists("Save_data"):
        os.mkdir("Save_data")
    dir = os.path.join("Save_data", savename)
    if not os.path.exists(dir):
        os.mkdir(dir)
    #save the data

    save_object(Importer, 'Importer', "Save_data/" + savename + "/")
    np.savetxt("Save_data/" + savename + "/y.csv", y, delimiter=",")
    np.savetxt("Save_data/" + savename + "/x.csv", variation_scheme, delimiter=",")

#load all the data that was saved in the last function
def load_training_data(savename):
    y = np.loadtxt("Save_data/" + savename + "/y.csv", delimiter=",")
    x = np.loadtxt("Save_data/" + savename + "/x.csv", delimiter=",")
    Importer = load_object("Importer", "Save_data/" + savename + "/")
    return x.T, y.T, Importer

#save all the variables, basically dimensions of the dataset
def save_variables(x, y, Importer):
    Data.N_elements = Importer.N_elements
    Data.N_integration_points = Importer.N_integration_points
    Data.N_samples = y.shape[0]
    Data.N_features = x.shape[1]

#split the training data and test data
def split_training_test(x, y, flowstress, training_ratio):
    Data.N_test = int((1-training_ratio)*Data.N_samples)
    Data.test_index = Data.test_index[:Data.N_test]

    x_test = np.copy(x[Data.test_index, :])
    x_train = np.delete(x, Data.test_index, 0)
    
    y_test = np.copy(y[Data.test_index, :])
    y_train = np.delete(y, Data.test_index, 0)

    flowstress_train = np.delete(flowstress, Data.test_index, 0)
    return x_train, x_test, y_train, y_test, flowstress_train

#split training and test for a single testing simulation
def split_training_test_single(x, y, i):

    x_train = np.delete(x, i, 0)
    x_test = np.copy(x[i, :])

    y_train = np.delete(y, i, 0)
    y_test = np.copy(y[i, :])
    return x_train, x_test, y_train, y_test

#normalizes a numpy array per column and gives the intervals
def fit_transform(array):
    new_array = np.zeros(array.shape)
    N_parameters = array.shape[-1]
    interval = np.zeros([2, N_parameters])
    for i in range(N_parameters):
        interval[0, i], interval[1, i] = np.max(array[:, i])+0.00001, np.min(array[:, i])-0.00001
        new_array[:, i] = normalize(array[:, i], interval[0, i], interval[1, i])
    return new_array, interval

#normalizes a numpy array per column, but uses the intervals
def transform(array, interval):
    new_array = np.zeros(array.shape)
    N_parameters = array.shape[-1]
    for i in range(N_parameters):
        new_array[:, i] = normalize(array[:, i], interval[0, i], interval[1, i])
    return new_array

#denormalizes the numpy arrays with a given interval
def detransform(array, interval):
    new_array = np.zeros(array.shape)
    N_parameters = array.shape[-1]
    for i in range(N_parameters):
        new_array[:, i] = denormalize(array[:, i], interval[0, i], interval[1, i])
    return new_array

#get the difference between predicted and real values (error)
def get_error(network, y_test_norm, x_test_norm, y_interval):
    y_model_norm = network.model(x_test_norm)
    error_list = [np.mean(abs(y_model_norm-y_test_norm))]
    y_model = detransform(y_model_norm, y_interval)
    y_test = detransform(y_test_norm, y_interval)
    output_model_split = np.split(y_model, network.N_output, axis=1)
    output_test_split = np.split(y_test, network.N_output, axis=1)
    for i in range(network.N_output):
        a = output_model_split[i]
        b = output_test_split[i]
        error_list.append(np.mean(abs(a-b)))
    return error_list

#save the input variables to the data class
def set_input_variables():
    variablename = Config.variable_to_vary
    input_variable_names = pd.read_excel('Save_data/VariationScheme.xlsx', engine='openpyxl', index_col=0).columns.ravel()[1:]
    variable_index = np.where(input_variable_names == variablename)[0][0]
    Data.variable_index = variable_index

#load all the variables
def load_data():
    variables = np.array(['Equivalent_Cauchy_stress', 'Stress_xx', 'Stress_yy', 'Stress_zz', 'Stress_xy'])
    intervals = np.loadtxt("Save_data/stress_forward/intervals.csv", delimiter=",")
    Importer = load_object("Importer", "Save_data/stress_forward/")
    x = np.loadtxt("Save_data/stress_forward/x.csv", delimiter=",")
    interval = np.loadtxt("Save_data/coordinates_forward/intervals.csv", delimiter=",")
    variation_string = os.listdir('Variation_and_material/Variation_scheme')[0]
    input_variable_names = pd.read_excel('Variation_and_material/Variation_scheme/' + str(variation_string), engine='openpyxl', index_col=0).columns.ravel()[1:]
    return variables, intervals, Importer, x, input_variable_names, interval
    
#initialize data that was used for visualization
def initialize_data(x, N_integration_points, N_elements):
    input_variables = np.full((1, 8), 0.5)
    networks = []
    variable = np.zeros([N_integration_points])
    return networks, N_integration_points, N_elements, input_variables, variable

#delete faulty simulations (with NaN values)
def delete_nan(x, y, flowstress):
    x = x[x[:, 0].argsort(), :]
    y = y[~np.isnan(x).any(axis=1), :]
    if flowstress is not False:
        flowstress = flowstress[~np.isnan(x).any(axis=1), :]
    x = x[~np.isnan(x).any(axis=1), :]
    if flowstress is not False:
        return x, y, flowstress
    return x, y