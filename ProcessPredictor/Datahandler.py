################################################################################
#this file holds all the functions that handles the training and validation set#
################################################################################
from Dataholder import DataHolder
from Config import Data
from Config import Config
from Functions import *

#Combines all simulations and save the training data onto the hard drive
def save_training_data(variables, foldername, savename):
    #Initialize the class that imports the data from all the files
    Importer = DataHolder()

    #get the variation scheme values and process names from the files
    variation_string = os.listdir('Variation_and_material/Variation_scheme')[0]  
    variation_scheme = np.array(pd.read_excel('Variation_and_material/Variation_scheme/' + str(variation_string), engine='openpyxl'))[:, :].T

    #get the foldernames of all the simulations
    fnames = os.listdir('Simulations')     

    #get all useful variables
    N_dataset, N_variables = len(fnames), len(variables)
    Importer.initialize(fnames[0], foldername)     
    coordinates = np.empty([Importer.N_nodes*N_variables, N_dataset])

    #get all coordinates, only if the amount of nodes is = to first simulations (basically remove all the remeshed simulations)
    for i in range(N_dataset):
        Importer.initialize(fnames[i], foldername)
        try:
            coordinates[:, i] = Importer.prepare_data(variables)
        except:
            print('Warning: Failed to import simulation')
        
    #check if the folder exists, otherwise create it
    if not os.path.exists("Save_data"):
        os.mkdir("Save_data")
    dir = os.path.join("Save_data", savename)
    if not os.path.exists(dir):
        os.mkdir(dir)

    #save the data
    np.savetxt("Save_data/" + savename + "/coordinates.csv", coordinates, delimiter=",")
    np.savetxt("Save_data/" + savename + "/variation_scheme.csv", variation_scheme, delimiter=",")
    return

#load all the data that was saved in the last function
def load_training_data(savename):
    variation_string = os.listdir('Variation_and_material/Variation_scheme')[0]  
    x = np.loadtxt("Save_data/" + savename + "/coordinates.csv", delimiter=",")
    y = np.loadtxt("Save_data/" + savename + "/variation_scheme.csv", delimiter=",")
    input_variable_names = pd.read_excel('Variation_and_material/Variation_scheme/' + str(variation_string), engine='openpyxl', index_col=0).columns.ravel()[np.array(Config.var_index)-1]
    return x, y, input_variable_names

#save all the variables, basically dimensions of the dataset
def save_variables(y, x):
    Data.N_samples = y.shape[1]
    Data.N_variables = y.shape[0]
    Data.N_coordinates = x.shape[0]

#normalizes a numpy array per column and gives the intervals
def fit_transform(array):
    #Initialize the arrays and variables
    new_array = np.zeros(array.shape)
    N_parameters = array.shape[0]
    interval = np.zeros([N_parameters, 2])

    #normalize, and create the interval array
    for i in range(N_parameters):
        interval[i, 0], interval[i, 1] = np.max(array[i])+0.00000000000000000001, np.min(array[i])-0.00000000000000000001
        new_array[i] = normalize(array[i], interval[i, 0], interval[i, 1])
    return new_array, interval

#normalizes a numpy array per column, but uses the intervals
def transform(array, interval):
    new_array = np.zeros(array.shape)
    N_parameters = array.shape[0]
    for i in range(N_parameters):
        new_array[i] = normalize(array[i], interval[i, 0], interval[i, 1])
    return new_array

#denormalizes the numpy arrays with a given interval
def detransform(array, interval):
    new_array = np.zeros(array.shape)
    N_parameters = array.shape[0]
    for i in range(N_parameters):
        new_array[i] = denormalize(array[i], interval[i, 0], interval[i, 1])
    return new_array

#split the training data and test data
def split_training_test(y, x, testing_ratio):

    #creates an element array with randomized integers to split the training/test set according to a testing ratio
    N_test = int(testing_ratio*Data.N_samples)
    test_index = np.arange(Config.boundary_index)
    np.random.shuffle(test_index)
    test_index = test_index[:N_test]

    #actually splits the data
    y_test = y[:, test_index]
    x_test = x[:, test_index]
    y_train = np.delete(y, test_index, axis=1)
    x_train = np.delete(x, test_index, axis=1)

    return y_test, x_test, y_train, x_train


def split_training_test_single(y, x, i):

    #actually splits the data
    y_test = y[:, i]
    x_test = x[:, i]
    y_train = np.delete(y, i, axis=1)
    x_train = np.delete(x, i, axis=1)

    x_test = x_test.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return y_test, x_test, y_train, x_train