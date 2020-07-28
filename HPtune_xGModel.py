# -*- coding: utf-8 -*-
# Investigating ConvNets to create high quality xG models - Hyperparameter Tuning - https://www.opengoalapp.com/xg-with-cnns-full-study
# by @openGoalCharles

import pandas as pd
import numpy as np
import tensorflow as tf
from ShotsToArrays import ShotsToArrays
from LoadEvents import LoadEvents
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import kerastuner as kt
import IPython # if using google colab you can get some progress plots whilst tuning


# STANDALONE VERSION OF CNN MODEL WITH HYPERPARAMTER TUNING

#-----------LOAD THE DATA---------------------

# choose one of the following options:

#shots = LoadEvents('shots') # call StatsBomb API for the latest open shot data - chuck everything into the model! WARNING - WILL TAKE AROUND 25 MINS AS THERE ARE OVER 850 GAMES
#DataIn, DataOut = ShotsToArrays(shots) # process the shot dataframe into set of 3 channel arrays and accompanying output data

# OR....

# pre-load a snapshot of the data (June 2020) that has already been processed
DataIn = np.load('input_arrays.npy') # loaded as float16 to save space - model will cast to float32 automatically
DataOut = pd.read_pickle("./output_arrays.pkl")

#---------------------------------------------

#split data into train and test
x_train, x_test, out_train, out_test= train_test_split(DataIn, DataOut, test_size=0.15, random_state = 42)
x_test = np.float32(x_test)
x_train = np.float32(x_train)

y_train= out_train.loc[:,'shot_outcome'].values.astype('float32')
y_test= out_test.loc[:,'shot_outcome'].values.astype('float32')


#-------CNN MODEL WITH KERAS TUNER WRAP AROUND FUNCTION--------------------------
def model_builder(hp):
    
    input_img = Input(shape=(40, 80, 3), dtype ='float16' )  
    
    hp_filters1 = hp.Int('filters1', min_value = 32, max_value = 128, step = 32)
    hp_kernelsize1 = hp.Choice('kernel_size1',values = [3, 5, 9])
    
    block1 = Conv2D(filters = hp_filters1, kernel_size = hp_kernelsize1, activation='relu', padding='same')(input_img)
    
    x = MaxPooling2D((2, 2), padding='same')(block1)
    
    hp_filters2 = hp.Int('filters2', min_value = 32, max_value = 128, step = 32)
    hp_kernelsize2 = hp.Choice('kernel_size2',values = [3, 5, 9])
    block2 = Conv2D(filters = hp_filters2, kernel_size = hp_kernelsize2, activation='relu', padding='same')(x)
    
    x = MaxPooling2D((2, 2), padding='same')(block2)
    
    hp_filters3 = hp.Int('filters3', min_value = 32, max_value = 128, step = 32)
    hp_kernelsize3 = hp.Choice('kernel_size3',values = [3, 5, 9])
    block3 = Conv2D(filters = hp_filters3, kernel_size = hp_kernelsize3, activation='relu', padding='same')(x)
    
    x = MaxPooling2D((2, 2), padding='same')(block3)
    
    x = Flatten()(x)
    
    hp_units = hp.Int('units', min_value = 16, max_value = 64, step = 16)
    x = Dense(hp_units, activation = 'relu')(x)
    
    
    output = Dense(1, activation = 'sigmoid')(x)
    
    model = Model(input_img, output)
    
    model.compile('sgd', loss='binary_crossentropy')
    
    return model

#-----------------------------------------------------------------------------------

# compile model
tuner = kt.Hyperband(model_builder, # can experiment with different searching algos - have chosen hyperband here
                     objective = 'val_loss',
                     max_epochs = 50,
                     factor = 3,
                     directory = 'C:\\YOUR\\DIRECTORY\\HERE',
                     project_name = 'hyperparams_output')

class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True) # if using google colab with interactive display - doesn't work in e.g. Spyder

# run tuning process
tuner.search(x_train, y_train, epochs = 30, validation_data=(x_test, y_test), callbacks = [ClearTrainingOutput()])

# get hyperparams of best model
best_hyperparameters = tuner.get_best_hyperparameters(1)[0].values

#{'filters1': 32,
# 'kernel_size1': 9,
# 'filters2': 64,
# 'kernel_size2': 5,
# 'filters3': 64,
# 'kernel_size3': 9,
# 'units': 48,
# 'learning_rate': 0.001,
# 'tuner/epochs': 15,
# 'tuner/initial_epoch': 0,
# 'tuner/bracket': 0,
# 'tuner/round': 0}

# PLUG OUTPUT VALUES BACK INTO ORIGINAL MODEL
