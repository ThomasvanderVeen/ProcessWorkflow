############################################################################
#this file holds a class that generates data for a specific simulation file#
############################################################################
from Functions import *
import warnings

#ignore some warnings
warnings.filterwarnings("ignore")

#Class that imports and manipulates the data files from a folder
class DataHolder:       #define all the variables in the class
    def __init__(self, mesh_data = 0, element_data = 0, parameter_values = 0, parameter_names = 0, \
            N_nodes = 0, N_elements = 0, elements = 0, N_integration_points = 0, numbering = 0, components_coords = 0, contour = 0):
        self.mesh_data = mesh_data
        self.element_data = element_data
        self.parameter_values = parameter_values
        self.parameter_names = parameter_names
        self.N_nodes = N_nodes
        self.N_elements = N_elements
        self.N_integration_points = N_integration_points
        self.elements = elements
        self.numbering = numbering
        self.components_coords = components_coords
        self.contour = contour

    #read the relevant folders
    def readfolder(self, folderpath, stepname):
        material_string = os.listdir('Variation_and_material/Material')[0] 
        rewrite('Simulations/' + str(folderpath) + '/' + str(stepname) + '.issn')       #issn file has to be rewritten before it can be imported
        self.element_data = np.genfromtxt('Simulations/' + str(folderpath) + '/' + str(stepname) + '.mshn', invalid_raise=False, dtype= np.int32)   
        self.mesh_data = np.genfromtxt('Simulations/' + str(folderpath) + '/' + str(stepname) + '.mshn', invalid_raise=False, skip_header = np.shape(self.element_data)[0]+1)       #some files need header skips etc.    
        self.parameter_values = np.genfromtxt('Simulations/' + str(folderpath) + '/' + str(stepname) + '.issn', invalid_raise=False, skip_header=5)
        self.parameter_names = np.genfromtxt('Variation_and_material/Material/' + str(material_string), skip_header=2, dtype=str)

    #initialize the importer by defining all the relevant variables
    def initialize(self, folderpath, stepname):
        self.readfolder(folderpath, stepname)     #get the data from the function readfolder
        self.N_elements = int(np.max(self.element_data[:, 0]))
        self.parameter_values = np.ndarray.flatten(self.parameter_values)
        self.elements = self.element_data[:, 0]
        numbering = np.ndarray.flatten(self.element_data[:, 2:])
        self.numbering = numbering.astype(int)-np.min(numbering)
        self.N_nodes = np.max(self.numbering)+1
        self.N_integration_points = int(np.shape(self.parameter_values)[0]/30)
        self.components_coords = np.split(self.numbering, self.N_elements)
        self.parameter_names = self.parameter_names[:, 0]

    #get the x y and z coordinates of the file
    def get_coordinates(self):
        data = [self.mesh_data[:, 1], self.mesh_data[:, 2], self.mesh_data[:, 3]]
        return data

    #prepare different variables into a single array, which is needed for the neural network. Without normalization
    def prepare_data(self, parameter_values):

        #Initialize variables
        N_parameters = len(parameter_values)
        dataset = np.array([])

        #get the coordinates
        coordinates = self.get_coordinates()

        #append coordinates into 1 dataset
        for i in range(N_parameters):
            if parameter_values[i] == 'x':
                dataset = np.append(dataset, coordinates[0])
            elif parameter_values[i] == 'y':
                dataset = np.append(dataset, coordinates[1])
            elif parameter_values[i] == 'z':
                dataset = np.append(dataset, coordinates[2])
        return dataset

    #gets the value of all the contour nodes in order
    def get_contour(self):
        #all the edge nodes will be added to this array
        edge = np.array([], dtype=np.int32)
        #bincount counts how many times an integer occurs in an array
        count = np.bincount(self.numbering) 
        for i in range(self.N_nodes):
            k = count[i]
            if k == 1:      #if a node occurs 1 time -> corner, 2 times -> edge, 2+ times -> volume
                edge = np.append(edge, i)
            elif k == 2:
                edge = np.append(edge, i)
        
        N_edge = edge.size
        x, y = self.mesh_data[edge, 1], self.mesh_data[edge, 2]
        self.contour = np.array([], dtype=np.int32)
        j = 0

        #for every edge node determine which one is the closest, this orders the contour nodes
        for i in range(N_edge+1):
            x_i, y_i = x[j], y[j]
            x_diff = x-x_i
            y_diff = y-y_i
            xy_diff = np.sqrt(np.square(x_diff) + np.square(y_diff))
            xy_diff[j], x[j], y[j] = np.finfo(xy_diff.dtype).max, np.finfo(x.dtype).max, np.finfo(y.dtype).min
            j = xy_diff.argmin()
            self.contour = np.append(self.contour, edge[j])

        #append the first contour as the last to complete the circle   
        self.contour = np.append(self.contour, self.contour[0])