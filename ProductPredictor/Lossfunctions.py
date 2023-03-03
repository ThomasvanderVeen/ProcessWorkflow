###############################################################
#This file contains loss functions for all specific situations#
###############################################################
import tensorflow as tf
from Functions import *
from Datahandler import *
from Config import Config
import time

#Set all the seeds
tf.keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(0)

#loss function when a PINN is present
def get_loss(network, x, y_data, PINN):
    #get the data loss: mean(y_pred - y_data)^2
    y_pred = network.model(x)        
  
    loss_d = tf.reduce_mean(tf.square(y_pred - y_data))        #get the data loss: mean(model - data)^2

    #test if there is a PINN. if there is no PINN, the loss is simply the data loss
    if PINN == True:        
        #create collocation points and insert them into the network to yield a prediction
        x_PINN = np.absolute(np.random.rand(Config.N_collocation_points, 8))
        y_PINN = network.model(x_PINN)

        #get the features and samples from the simulation and make a prediction
        x_PINN_2 = tf.cast(x, tf.float32)
        y_PINN_2 = network.model(x_PINN_2)

        #Initialize the losses
        loss_1, loss_2, loss_3 = 0, 0, 0
        N_points_in_element = 4

        #loop over the integration points in 1 element (4 in this case)
        for i in range(N_points_in_element):

            #get the flowstress from the equations of a specific integration point
            flowstress = Data.flowstress[:, i]

            #split the network predictions such that you get the values for 1 integration point
            y_PINN_split = y_PINN[:, i*len(network.variables): (i+1)*len(network.variables)]
            y_PINN_split_2 = y_PINN_2[:, i*len(network.variables): (i+1)*len(network.variables)]

            #get the intervals for 1 integration point (for denormalizing: needed for calculation)
            interval_split = Data.interval[:, i*len(network.variables): (i+1)*len(network.variables)]

            #denormalize and split the variables: yields predicted stress parameters for 1 integration point and all the samples (for collocation points & simulation points)
            [stress_equiv, stress_xx, stress_yy, stress_zz, stress_xy] = \
                denormalize_output_tf(y_PINN_split, interval_split, network.variables)       
            [stress_equiv_2, stress_xx_2, stress_yy_2, stress_zz_2, stress_xy_2] = \
            denormalize_output_tf(y_PINN_split_2, interval_split, network.variables)

            #get anisotropy from collocation points
            R = detransform(x_PINN, Data.x_interval)[:, -1]

            #calculate alpha values
            alpha = get_alpha(R)        #from these R values generate the alpha values

            #calculate equivalent stress from stress components
            stress_equiv_model = get_equivalent_stress([stress_xx, stress_yy, stress_zz, stress_xy], alpha)   #from the stress components and alpha generate the equivalent stress  

            #denormalize the PINN loss functions, and calculate them
            norm = Data.interval[0, 0] - Data.interval[0, 1]
            loss_3 += tf.reduce_mean(tf.square(((stress_equiv_model-stress_equiv)/norm)))                   #Equivalent stress from the model should be equal to calculated equivalent stress
            loss_1 += tf.reduce_mean(tf.square(tf.nn.relu(((stress_equiv_2-flowstress)/norm))))
            loss_2 += tf.reduce_mean(tf.square(tf.nn.relu(-stress_equiv)/norm))                                #Equivalent cauchy stress > 0 
        
        #weigh the PINN loss functions with the data loss
        loss = (loss_1**2 + loss_2**2 + loss_3**2 + loss_d**2)/(loss_1 + loss_2 + loss_3 + loss_d)
        
        return loss, [loss_d, loss_1, loss_2, loss_3]
    return loss_d, [0, 0, 0, 0]
    
#a function that denormalizes and split the output from the neural network
def denormalize_output_tf(y_norm, interval, variables):

    #initialize
    N_variables = len(variables)
    y_split = []

    #split the model output into N equal sized tensors
    y_split_norm = tf.split(y_norm, num_or_size_splits=N_variables, axis=1)

    #denormalize each variable
    for i in range(N_variables):
        denormalized_paramater = denormalize(tf.reshape(y_split_norm[i], [tf.size(y_split_norm[i]),]), interval[0, i], interval[1, i])
        y_split.append(denormalized_paramater)

    return y_split

#a function that calculates the equivalent stress from the stress components and alpha
def get_equivalent_stress(stress_components, alpha):

    #get the alpha components
    [a_1, a_2, a_3, a_6] = alpha 

    #calculate equivalent stress
    stress_eq = tf.sqrt(tf.multiply(a_1, tf.square((stress_components[1]-stress_components[2])))+tf.multiply(a_2, tf.square((stress_components[2]-stress_components[0]))) \
        + tf.multiply(a_3, tf.square((stress_components[0]-stress_components[1]))) \
        +3*tf.multiply(a_6, tf.square(stress_components[3])))/np.sqrt(2)        
        
    return stress_eq

#gets the alpha vectors from the R values
def get_alpha(R):
    a_1 = 2 - 2/(1 + R)
    a_2 = 2/(1 + R)
    a_3 = 2/(1 + R)
    a_6 = 2.
    return [tf.cast(a_1, tf.float32), tf.cast(a_2, tf.float32), tf.cast(a_3, tf.float32), tf.cast(a_6, tf.float32)]