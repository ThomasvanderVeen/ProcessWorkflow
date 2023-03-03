############################################################################
#this file combines all the data scripts, keeps and runs the neural network#
############################################################################
from Functions import *
clear_and_print('[Initializing]')
from Datahandler import *
from Config import Config
#Initialize

variables = ['x', 'y']
folderpath = 'Simulations'
savename = 'Process_predictor_saves'
input_variable_names = ['Blankholder Force', 'Punch diameter', 'Redraw depth', 'Fillet radius']

Importer = DataHolder()
Importer.initialize(str(os.listdir(folderpath)[0]), Config.foldername)
Importer.get_contour()

#if saving data is True in config, then save the data to csv
if Config.save_data:
    clear_and_print('[Saving training data]')
    save_training_data(variables, Config.foldername, savename)

#load the training data
clear_and_print('[Loading training data]')
x, y, _ = load_training_data(savename)

#delete faulty simulations, importing was skipped in save_training_data
x, y = delete_nan(x, y)
del_index = np.where(x[0, :]==0)
x, y = np.delete(x, del_index, axis=1), np.delete(y, del_index, axis=1)

#transform to nodes that are on the contour
clear_and_print('[Transforming to only contour data]')
[xcoords, ycoords] = np.split(x, len(variables))
y_contour = ycoords[Importer.contour, :]
x_contour = xcoords[Importer.contour, :]

#overlap all the contours by having the last node be at 0, 0
for i in range(xcoords.shape[1]):
    y_contour[:, i] = y_contour[:, i] - ycoords[-1, i]
    x_contour[:, i] = x_contour[:, i] - xcoords[-1, i]

#stack the x and y coordinates
x = np.vstack([x_contour, y_contour])

#add noise parameters to input
x = np.vstack([x, y[[2, 3, 6, 7], :]])

#use process parameters as labels
y = y[[0, 1, 4, 5], :]

#save some variables used for fitting
save_variables(y, x)

#split taining and test with a 'testing ratio' defined in config
x, interval_x = fit_transform(x)
y, interval_y = fit_transform(y)
y_test, x_test, y_train, x_train = split_training_test(y, x, Config.testing_ratio)

#neural network using keras
clear_and_print('[Initialize and compile neural network model]')
model = tf.keras.Sequential()       #define the model

model.add(tf.keras.layers.Input(shape=(Data.N_coordinates,)))   #add an input layer with # neurons equal to coordinates (nodes * degrees of freedom (2))
for i in Config.layers:     #add a layer with activation tanh and dimension i defined in config
    model.add(tf.keras.layers.Dense(i, activation='tanh'))
    model.add(tf.keras.layers.Dropout(0.1))         #add dropout layer (helps against overfitting)
model.add(tf.keras.layers.Dense(Data.N_variables))      #add output layer

#compile neural network with mean absolute error loss function and adam optimizer
clear_and_print('[Compiling]')
model.compile(loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(Config.learning_rate))

#fit the model using the testing and training data
clear_and_print('[Fitting]')
history = model.fit(x_train.T, y_train.T, epochs=Config.N_epochs)

#get testing error
errors_MAE = abs(y_test.T - model(x_test.T))[0]
errors_nominal = abs(detransform(y_test, interval_y) - detransform(model(x_test.T).numpy().T, interval_y))


##########################################################################################################################
#This is for plotting purposes#
##########################################################################################################################

unit = ['[N]', '[mm]', '[mm]', '[mm]']
min_ticks = [100, 1.50, 25, 0.5]
max_ticks = [500, 2.00, 33, 1.0]
N_ticks = [5, 6, 5, 6]
letter = ['(a)', '(b)', '(c)', '(d)', ]

fig, axis = plt.subplots(2, 2)
fig.set_size_inches(5, 5)
fig.tight_layout()
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
axis = np.ndarray.flatten(axis)

for i in range(4):    
    axis[i].scatter(detransform(y_train, interval_y)[i, :], detransform(model(x_train.T).numpy().T, interval_y)[i, :], color = "#c1272d", s = 10, marker = "o", label='Training', facecolors='none')
    axis[i].scatter(detransform(y_test, interval_y)[i, :], detransform(model(x_test.T).numpy().T, interval_y)[i, :], color = "#0000a7", s = 20, marker = "s", label='Validation', facecolors='none')
    axis[i].autoscale(False)
    axis[i].plot([-10e5, 10e5], [-10e5, 10e5], color = "black", linestyle = 'dashed', lw=1)
    axis[i].set_title(input_variable_names[i], y = 1)
    axis[i].set_ylabel('network')
    axis[i].set_xlabel('real')
    axis[i].tick_params(axis='both', which='major', labelsize=8)
    axis[i].set_ylabel('network ' + str(unit[i]))
    axis[i].set_xlabel('real ' + str(unit[i]))
    axis[i].text(-0.3, 0.95, letter[i], transform=axis[i].transAxes)
    axis[i].xaxis.set_ticks(np.linspace(min_ticks[i], max_ticks[i], N_ticks[i]))
    axis[i].yaxis.set_ticks(np.linspace(min_ticks[i], max_ticks[i], N_ticks[i]))
    range = (max_ticks[i]-min_ticks[i])*0.07
    axis[i].set_xlim(min_ticks[i]-range, max_ticks[i]+range)
    axis[i].set_ylim(min_ticks[i]-range, max_ticks[i]+range)
    axis[i].text(0.02, 0.92,  'error: ' + str(np.round(errors_MAE[i].numpy(), 3)), transform=axis[i].transAxes)


if not os.path.exists("Output"):
    os.mkdir("Output")
if not os.path.exists("Models"):
    os.mkdir("Models")

fig.tight_layout()
handles, labels = axis[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0), frameon=True, edgecolor="black", facecolor='white', ncol=3)
fig.savefig('Output/Accuracy_plot.png', dpi=300, bbox_inches='tight', pad_inches = 0)
fig.savefig('Output/Accuracy_plot.pdf', dpi=300, bbox_inches='tight', pad_inches = 0)


model.save("Models/ProcessPredictor.h5")
np.savetxt("Save_data/" + str(savename) + "/interval_x.csv", interval_x, delimiter=",")
np.savetxt("Save_data/" + str(savename) + "/interval_y.csv", interval_y, delimiter=",")
np.savetxt("Save_data/" + str(savename) + "/x_test.csv", x_test, delimiter=",")
np.savetxt("Save_data/" + str(savename) + "/y_test.csv", y_test, delimiter=",")
np.savetxt("Save_data/" + str(savename) + "/errors_MAE.csv", errors_MAE, delimiter=",")
np.savetxt("Save_data/" + str(savename) + "/errors_nominal.csv", errors_nominal, delimiter=",")
