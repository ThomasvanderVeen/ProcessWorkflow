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
    foldername = config.get("General", "Last_step_name")
    boundary_index = config.getint("General", "Boundary_index")
    var_index = json.loads(config.get("General", "use_variable_index"))

    layers = json.loads(config.get("Network","Architecture"))
    N_epochs = config.getint("Network", "N_epochs")
    learning_rate = config.getfloat("Network", "Learning_rate")    
    testing_ratio = config.getfloat("Network", "Testing_ratio")

#store all temporary data here
class Data:
    Datastorage = 1