############################################################################
#this file holds a class that generates data for a specific simulation file#
############################################################################
from Functions import *
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
from Config import Config

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
        rewrite( 'Simulations/' + str(folderpath) + '/' + str(stepname) + '.issn')       #issn file has to be rewritten before it can be imported
        self.element_data = np.genfromtxt('Simulations/' + str(folderpath) + '/' + str(stepname) + '.mshn', invalid_raise=False, dtype= np.int32)   
        self.mesh_data = np.genfromtxt('Simulations/' + str(folderpath) + '/' + str(stepname) + '.mshn', invalid_raise=False, skip_header = np.shape(self.element_data)[0]+1)       #some files need header skips etc.    
        self.parameter_values = np.genfromtxt('Simulations/' + str(folderpath) + '/' + str(stepname) + '.issn', invalid_raise=False, skip_header=5)
        self.parameter_names = np.genfromtxt('Variation_and_material/Material/' + str(material_string), skip_header=2, dtype=str)

    #initialize the importer by defining all the relevent variables
    def initialize(self, folderpath, stepname):
        self.readfolder(folderpath, stepname)     #get the data from the function readfolder
        self.N_elements = int(np.max(self.element_data[:, 0]))
        self.parameter_values = np.ndarray.flatten(self.parameter_values)
        self.elements = self.element_data[:, 0]
        numbering = np.ndarray.flatten(self.element_data[:, 2:])
        self.numbering = numbering.astype(int)-np.min(numbering)
        self.N_nodes = np.max(self.numbering)+1
        self.N_integration_points = numbering.size
        self.components_coords = np.split(self.numbering, self.N_elements)
        self.parameter_names = self.parameter_names[:, 0]
        self.get_contour()      #use these variables to get contour numbering of the nodes


    #choose a variable as a string and get the variables 
    def get_variable(self, variablestring):

        #determine and initialize variables
        integer = np.where(self.parameter_names == variablestring)[0][0]       #determine which integer the variable correlates to 
        arg = np.zeros([self.N_integration_points])

        #loop over the !total! number of integration points
        for i in range(self.N_integration_points):        
            arg[i] = self.parameter_values[i*int(self.parameter_values.size/self.N_integration_points)+integer]

        #take average of the nodes to get the value of the variable for all elements
        average_arg = average_variables(arg, self.N_elements)       
        return arg, average_arg

    #get the x y and z coordinates of all nodes in the product
    def get_coordinates(self):
        data = [self.mesh_data[:, 1], self.mesh_data[:, 2], self.mesh_data[:, 3]]
        return data

    #prepare different variables into a single array, which is needed for the neural network. Without normalization
    def prepare_data(self, parameter_values):

        #initialize
        N_parameters = len(parameter_values)
        dataset = np.array([])
        coordinates = self.get_coordinates()

        #if coordinates are used (has to be different because N_nodes is not equal to N_integration_points)
        if 'x' in parameter_values:
            arr2 = np.tile(np.arange(0, (N_parameters)*self.N_nodes, self.N_nodes), self.N_nodes)
            arr = np.repeat(np.arange(self.N_nodes), N_parameters)

        #if coordinates are not used
        else:
            arr2 = np.tile(np.arange(0, (N_parameters)*self.N_integration_points, self.N_integration_points), self.N_integration_points)
            arr = np.repeat(np.arange(self.N_integration_points), N_parameters)

        #get the total index for reshuffling
        index = arr+arr2

        #append variables to dataset
        for i in range(len(parameter_values)):
            if parameter_values[i] == 'x':
                dataset = np.append(dataset, coordinates[0])
            elif parameter_values[i] == 'y':
                dataset = np.append(dataset, coordinates[1])
            elif parameter_values[i] == 'z':
                dataset = np.append(dataset, coordinates[2])
            else:
                arg, average_arg = self.get_variable(parameter_values[i])
                dataset = np.append(dataset, arg)

        #reshuffle from 111222333 to 123123123 for example, to get integration points of an element together.
        return np.ndarray.flatten(dataset[index])

    #get the numbering of the contour nodes (edge and corner nodes)
    def get_contour(self):

        #initialize
        contour = np.array([], dtype=np.int32)
        self.contour = np.array([], dtype=np.int32)
        j= 0        #j is where the contour loop starts, 0 is standard

        #determine which node is corner, edge or volume
        count = np.bincount(self.numbering) 

        #loop over all the nodes
        for i in range(self.N_nodes):
            k = count[i]
            if k == 1:      #if a node has 1 neighbour -> corner, 2 neighbours -> edge, 2+ neighbours -> volume, corner and edge notes go on the contour
                contour = np.append(contour, i)
            elif k == 2:
                contour = np.append(contour, i)
        N_contour = contour.size

        #get x and y values (actually r and z) of all the contour nodes
        x, y = self.mesh_data[contour, 1], self.mesh_data[contour, 2]

        #loop over all contour nodes + 1
        for _ in range(N_contour+1):
            x_i, y_i = x[j], y[j]       #get current x, y values of current node
            x_diff = x-x_i              #get distance of all other nodes to current node
            y_diff = y-y_i
            xy_diff = np.sqrt(np.square(x_diff) + np.square(y_diff))
            xy_diff[j], x[j], y[j] = np.finfo(xy_diff.dtype).max, np.finfo(x.dtype).max, np.finfo(y.dtype).min
            j = xy_diff.argmin()        #closest physical node will be next in line and changed to current node
            self.contour = np.append(self.contour, contour[j])

        #close the contour loop by adding the first as last
        self.contour = np.append(self.contour, self.contour[0])

    #plotting a specific variable and the x/y coordinates
    def plot_variable(self, variablename):
        arg, average_arg = self.get_variable(variablename)      #get the non-normalized variable you want to plot
        norm_arg = normalize(average_arg, np.max(average_arg), np.min(average_arg))
        for i in range(self.N_elements):        #plot lines and fill for every element
#            plt.plot(self.mesh_data[self.components_coords[i], 1], self.mesh_data[self.components_coords[i], 2] \
#               , c = 'black', linewidth=0.1)
            plt.fill(self.mesh_data[self.components_coords[i], 1], self.mesh_data[self.components_coords[i], 2] \
                , c = mpl.cm.plasma(norm_arg[i]))
        plt.plot(self.mesh_data[self.contour, 1], self.mesh_data[self.contour, 2], c='black', linewidth=1.5)        #plot the contour in a thicker black line
        #plt.scatter(self.mesh_data[:, 1], self.mesh_data[:, 2], c='black', s = 0.01)       #plot the points in black
        plt.xlim(np.min(self.mesh_data[self.contour, 1]), np.max(self.mesh_data[self.contour, 1]))
        plt.ylim(np.min(self.mesh_data[self.contour, 2]), np.max(self.mesh_data[self.contour, 2]))
        cmap = mpl.cm.plasma        #the colormap plasma (arbitrarily chosen)
        norm = mpl.colors.Normalize(vmin=np.min(average_arg), vmax=np.max(average_arg))     #normalize the colorbar to the max and min of the variable
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label=variablename)
        plt.xlabel('x axis')
        plt.xlabel('y axis')
        plt.title('Geometry and ' +str(variablename))       #title
        plt.show()
