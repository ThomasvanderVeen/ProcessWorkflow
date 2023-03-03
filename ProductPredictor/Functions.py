##############################################
#this file holds all kind of useful functions#
##############################################
import numpy as np
import os
import pickle
import tensorflow as tf
import time
#this folder holds all the useful functions

#rewrite the filename such that there is a whitespace between all the columns
def rewrite(filename):
    with open(filename, 'r') as file :      #define the file
        filedata = file.read()      #read into the file

    filedata = filedata.replace(' -', '-')      #create a whitespace before the minus
    filedata = filedata.replace('-', ' -')      #create a whitespace before the minus
    filedata = filedata.replace('E -', 'E-')        #delete the whitespace after E

    with open(filename, 'w') as file:       #write the filename
        file.write(filedata)        
    return

#average the nodal values of a variable for all elements
def average_variables(variable, N_elements):
    variable_split = np.split(variable, N_elements)
    for i in range(N_elements):
        variable_split[i] = np.mean(variable_split[i])
    return variable_split     

#clear the terminal
def clear_terminal():
    clear = lambda: os.system('cls')
    clear()
    return

#normalize an array
def normalize(normalize, max, min):
    norm = (normalize-min)/(max-min)
    return norm

#do the inverse of normalization
def denormalize(norm, max, min):
    normalize = norm*(max-min)+min
    return normalize

#save an object with name from folder path
def save_object(object, name, path):
    with open(path + name + '.pickle', 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

#load an object with name from folder path
def load_object(name, path):
    with open(path + name + '.pickle', 'rb') as handle:
        object = pickle.load(handle)
    return object

#remove a line
def clear_line(n=1):
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for i in range(n):
        print(LINE_UP, end=LINE_CLEAR)

#function that removes 0's from lists
def remove_0_list(data):
    new_data = []
    for i in range(len(data)):
        if not data[i] <= 0:
            new_data.append(data[i])
    return new_data

#converts string 2 booleans
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

#set all the seeds
def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

#clear terminal and print a line
def clear_and_print(string):
    clear_terminal()
    print(string)

#calculate the mean, max, min and standard deviation of an array
def mean_max_min_std(array):
    N = array.shape[0]
    array_mean, array_max, array_min, array_std = np.zeros([N]), np.zeros([N]), np.zeros([N]), np.zeros([N])

    for i in range(N):
        array_mean[i] = np.mean(array[i, :])
        array_max[i] = np.max(array[i, :])
        array_min[i] = np.min(array[i, :])
        array_std[i] = np.std(array[i, :])
    return array_mean, array_max, array_min, array_std
