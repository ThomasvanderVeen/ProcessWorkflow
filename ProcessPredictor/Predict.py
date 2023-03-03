############################################################################
#This file predicts the process parameters from a single (testing) geometry#
############################################################################
from Functions import *
from Datahandler import *
import time

#Initialize and load everything
savename = 'Process_predictor_saves'
interval_y = np.loadtxt("Save_data/" + str(savename) + "/interval_y.csv", delimiter=",")
x_test = np.loadtxt("Save_data/" + str(savename) + "/x_test.csv", delimiter=",")
y_test = np.loadtxt("Save_data/" + str(savename) + "/y_test.csv", delimiter=",")
model = tf.keras.models.load_model("Models/ProcessPredictor.h5")
input_variable_names = ['Blankholder Force', 'Punch diameter', 'Redraw depth', 'Fillet radius']        #only the 8 process paremeters are useful, not CTQ's
predictions = np.zeros([len(input_variable_names), 3])
index = ['Predicted', 'Real', 'difference']

#predict using the neural network
x_predict = np.reshape(x_test, (np.shape(x_test)[0], 1)).T
y_predict_norm = model(x_predict).numpy()[0]
y_predict = detransform(y_predict_norm, interval_y)
y_real = detransform(y_test, interval_y)

#round and produce some statistics
for i in range(len(input_variable_names)):
    predictions[i, 0] = np.around(y_predict[i], 5)
    predictions[i, 1] = np.around(y_real[i], 5)
    predictions[i, 2] = np.around(abs(y_predict[i]-y_real[i]), 5)

#visualize the data neatly in a table
df = pd.DataFrame(predictions, index=input_variable_names, columns=index)
clear_terminal()
print(tabulate(df, headers='keys', tablefmt='psql'))
df.to_csv('Output/table_predictions.csv', index=True, header=True, sep=',')