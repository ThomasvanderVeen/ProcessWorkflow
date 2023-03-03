#############################################
#this file contains the neural network class#
#############################################
import tensorflow as tf
from Lossfunctions import *
from Functions import*
import matplotlib.pyplot as plt
import time
from Datahandler import *

#set all the neural network variables in the network class. Allows us to generate multiple networks easily
class Network:      
    def __init__(self, N_input, N_output, N_steps, label, data, PINN,\
                error_list = 0, act_function='tanh', model=0, learning_time = 0, variables = 0, print_data = False):
        self.N_input = N_input
        self.N_output = N_output
        self.N_steps = N_steps
        self.label = label
        self.data = data
        self.PINN = PINN
        self.error_list = error_list
        self.act_function = act_function
        self.model = model
        self.learning_time = learning_time
        self.variables = variables
        self.print_data = print_data
        
    #initialize the model
    def init_model(self, layers):
        #set seed for weight initializer 
        initializer = tf.keras.initializers.GlorotNormal(seed = 0)    

        #use sequential model for network
        self.model = tf.keras.Sequential()      

        #add input layer with N_input neurons
        self.model.add(tf.keras.layers.Input(shape=(self.N_input,))) 

        #add the hidden layers
        for i in range(len(layers)):
            self.model.add(tf.keras.layers.Dense(layers[i],
                                            activation=tf.keras.activations.get(self.act_function),     #activation function can be defined
                                            kernel_initializer=initializer))        #Xavier normal initializer (for tanh activation function)

        #output layer
        self.model.add(tf.keras.layers.Dense(self.N_output,
                                            activation=tf.keras.activations.get(self.act_function),     #activation function can be defined
                                            kernel_initializer=initializer))       #add an output layer with dimensions N   

    #get the gradient of the loss relative to the trainable variables
    def get_gradient(self, get_loss):
        with tf.GradientTape(persistent=True) as tape:

            #gradient with relation to trainable variables
            tape.watch(self.model.trainable_variables)   

            #calculate loss with loss function defined in Lossfunctions.py
            loss, losses = get_loss(self, self.label, self.data, self.PINN)

        #calculate gradient
        gradient = tape.gradient(loss, self.model.trainable_variables)    
        del tape
        return gradient, loss, losses   

    #tensorflow function decreases learning time significantly, this function performs a single training step
    @tf.function
    def train_step(self, get_loss):  

        #compute current loss and gradient with relation to trainable parameters
        gradient, loss, losses = self.get_gradient(get_loss)

        #perform gradient descent step     
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        return loss, losses

    #the train function 
    def train(self, get_loss, output_test, input_test, interval):

        #intitialize
        self.error_list = np.zeros([6 + len(self.variables), 25])
        j=0

        #train for N_steps
        for i in range(self.N_steps): 

            #get the losses      
            loss, losses = self.train_step(get_loss)        

            #if the data should be saved and printed, this part of the code runs
            if self.print_data:

                #save and or print loss, real error, PINN error and learning percentage 25 times in total
                if i%int((self.N_steps/25)) == 0:      
                    error = get_error(self, output_test, input_test, interval)
                    clear_line()
                    print('[Network ' + str(Data.i) + ', Learning: ' + str(int((i/self.N_steps)*100)) + '% finished, current loss: ' + str(loss.numpy()) + ', current real error: ' + str(error[0]) + ']') 
                    errors = np.append(np.array([loss.numpy(), error[0], losses[0], losses[1], losses[2], losses[3]]), error[1:len(self.variables)+1])
                    self.error_list[:, j] = errors
                    j += 1

        #get testing error of final network
        error = get_error(self, output_test, input_test, interval)
        return error[0]

    #set the optimizer
    def set_optimizer(self, initial, decay_steps, decay_rate):

        #exponential decay learning schedule = initial*decay_rate^(step/decay_steps), goes to constant if decay rate = 1
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(       
        initial_learning_rate=initial,
        decay_steps=decay_steps,
        decay_rate=decay_rate)

        #add leraning schedule to adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)                