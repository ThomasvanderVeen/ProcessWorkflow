###################################################################################
#this file imports from the config file and saves temporary data in the data class#
###################################################################################
import configparser
import json

#start configparser and reat the config file
config = configparser.ConfigParser()
config.read("ConfigFile.ini")

#get all the relevant variables from the config file and save them in the class
class Config: 

    save_data = config.getboolean("General", "Save_data")
    filename = config.get("General", "Last_step_name")

    layers = json.loads(config.get("Network Stress", "Architecture"))
    N_steps = config.getint("Network Stress", "N_steps")
    initial_learning_rate = config.getfloat("Network Stress", "Initial_learning_rate")    
    decay_steps = config.getint("Network Stress", "Decay_steps")
    decay_rate = config.getfloat("Network Stress", "Decay_rate")
    validation_ratio = config.getfloat("Network Stress", "Validation_ratio")

    layers_coords = json.loads(config.get("Network Coordinates", "Architecture"))
    N_steps_coords = config.getint("Network Coordinates", "N_steps")
    initial_learning_rate_coords = config.getfloat("Network Coordinates", "Initial_learning_rate")    
    decay_steps_coords = config.getint("Network Coordinates", "Decay_steps")
    decay_rate_coords = config.getfloat("Network Coordinates", "Decay_rate")
    validation_ratio_coords = config.getfloat("Network Coordinates", "Validation_ratio")

    N_collocation_points = config.getint("PCNN", "N_collocation_points")

#store all temporary data here
class Data:
    datastorage = 1
