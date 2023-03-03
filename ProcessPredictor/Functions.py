##############################################
#this file holds all kind of useful functions#
##############################################
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tabulate import tabulate
import pandas as pd
import time
import sys
from Config import *

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

#remove a line
def clear_line(n=1):
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR)

#clear a line and print a line in the terminal
def clear_and_print(string):
    clear_terminal()
    print(string)

#Delete all samples that contain a NaN (incomplete samples)
def delete_nan(x, y):
    y = y[:, y[0, :].argsort()]
    y = y[Config.var_index, :]
    x = x[:, ~np.isnan(y).any(axis=0)]
    y = y[:, ~np.isnan(y).any(axis=0)]
    return x, y
