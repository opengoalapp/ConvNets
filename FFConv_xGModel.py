# -*- coding: utf-8 -*-

# Investigating ConvNets to create high quality xG models - https://www.opengoalapp.com/xg-with-cnns-full-study
# by @openGoalCharles

# Tested with tensorflow 2.2.0 - some of the visualisations will definitely need >= v2.0.0, not sure about the core code
# The model is lightweight - will train in a few minutes with a low end GPU, training on CPU is definitely acceptable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ShotsToArrays import ShotsToArrays
from LoadEvents import LoadEvents

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

#-----------LOAD THE DATA---------------------

# choose one of the following options:

#shots = LoadEvents('shots') # call StatsBomb API for the latest open shot data - chuck everything into the model!
#DataIn, DataOut = ShotsToArrays(shots) # process the shot dataframe into set of 3 channel arrays and accompanying output data

# OR....

# pre-load a snapshot of the data (June 2020) that has already been processed
loaded = np.load('data/input_compressed.npz')
DataIn = loaded['a'] 
DataOut = pd.read_pickle("data/output_arrays.pkl")

#---------------------------------------------


# split data into train and test
x_train, x_test, out_train, out_test = train_test_split(DataIn, DataOut, test_size=0.3, random_state = 42) 

# split into a further set for training post-calibration e.g. isotonic regression
#x_train, x_val, out_train, out_val = train_test_split(x_train, out_train, test_size = 0.3, random_state = 42)

x_test = np.float32(x_test)
x_train = np.float32(x_train)

y_train= out_train.loc[:,'shot_outcome'].values.astype('float32')
y_test= out_test.loc[:,'shot_outcome'].values.astype('float32')



#-------DEFINE CNN MODEL--------------------------
activation = tf.keras.layers.LeakyReLU(alpha=0.1)


input_img = Input(shape=(40, 80, 3), dtype ='float16' )  

block1 = Conv2D(32, (9, 9), activation=activation, padding='same')(input_img)

x = MaxPooling2D((2, 2), padding='same')(block1)

block2 = Conv2D(64, (5, 5), activation=activation, padding='same')(x)

x = MaxPooling2D((2, 2), padding='same')(block2)

block3 = Conv2D(64, (9, 9), activation=activation, padding='same')(x)

x = MaxPooling2D((2, 2), padding='same')(block3)

x = Flatten()(x)

x = Dense(48, activation = activation)(x)

output = Dense(1, activation = 'sigmoid')(x)

model = Model(input_img, output)

#--------------------------------------------------

#------COMPILE MODEL AND SET CALLBACKS FOR EARLY STOPPING AND REDUCTION OF LEARNING RATE -----------------
optimizer = 'sgd'

model.compile(optimizer=optimizer, loss='binary_crossentropy')

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, 
                           patience=10, verbose=0, mode='auto',
                           baseline=None, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00001)

#----------------------------------------------------------------------------------------------------------

#-------TRAIN MODEL--------------------------------
model.fit(x_train, y_train,
                epochs=200,
                batch_size=32, 
                shuffle=True,
                validation_data=(x_test, y_test),
                callbacks = [early_stop, reduce_lr])

#-------------------------------------------------- 

# generate model xG predictions of of training set
y_pred = model.predict(x_test)

# plot calibration curve for model
ccurve = calibration_curve(y_test, y_pred, n_bins = 15) # returns true proportion [0] and average predicted prob [1]
plt.scatter(ccurve[1],ccurve[0])
plt.title('CNN model Calibration Curve - Final with Optimization')
plt.xlabel('Average of model predicted xG')
plt.ylabel('Average of actual goal outcome')
x = [0,1]
y = [0,1]
plt.plot(x,y, '--')
plt.show()

# plot comparison of CNN model predictions vs StatsBomb's predictions on the same data
sb_xg = out_test.loc[:,'statsbomb_xg']
plt.scatter(y_pred,sb_xg, alpha = 0.1)
plt.show()

# plot calibration curve for StatsBomb model on the test data 
sb_ccurve = calibration_curve(y_test, sb_xg, n_bins = 15)
plt.scatter(sb_ccurve[1],sb_ccurve[0])
plt.title('StatsBomb model Calibration Curve')
plt.xlabel('Average of model predicted xG')
plt.ylabel('Average of actual goal outcome')
x = [0,1]
y = [0,1]
plt.plot(x,y, '--')
plt.show()

# calculate benchmark log loss values

ll_model = log_loss(y_test, y_pred) # CNN model
ll_sb = log_loss(y_test, sb_xg) # StatsBomb model
ll_fixed = log_loss(y_test, np.repeat(np.mean(y_train),len(y_test))) # fixed mean of training data prediction
ll_rand = log_loss(y_test, np.random.rand(len(y_test))) # random number

# do same for perfect model
goal_list = []
for shot in sb_xg: # simulate goal/no goal outcome assuming SB model is ground truth
    if np.random.rand() <=shot:
        goal_list.append(1)
    else:
        goal_list.append(0)
ll_perfect = log_loss(goal_list, sb_xg) # log loss for perfect model with representative data     

## train a logistic regression model using x and y locs as inputs
#X = np.array(out_train.loc[:,['loc_x', 'loc_y']]) # requires data processed from ShotsToArrays 
#lr_model = LogisticRegression()
#model_out = lr_model.fit(X,y_train)

#lr_test = model_out.predict_proba(np.array(out_test.loc[:,['loc_x', 'loc_y']]))[:,1] # generate the probabilities
#ll_lr = log_loss(y_test, lr_test) # calculate log loss of logistic regression model

# plot a calibration curve for the logistic regression model
#lr_ccurve = calibration_curve(y_test, lr_test, n_bins = 15)
#plt.scatter(lr_ccurve[0],lr_ccurve[1])
#x = [0,1]
#y = [0,1]
#plt.plot(x,y, '--')
#plt.show()



#------------ISOTONIC REGRESSION EXPERIMENT---------------------
# # uncomment this block and uncommment line 35 (additional split of data)
#ir = IsotonicRegression()
#ir.fit(model.predict(x_val).squeeze(),y_val)
#calibrated = ir.predict(xG.squeeze())
#
#calibrated = np.clip(calibrated,0.001,0.999) # stop divide by zero error
#
#calib_ccurve = calibration_curve(y_test, calibrated, n_bins = 15)
#plt.scatter(calib_ccurve[1],calib_ccurve[0])
#plt.title('CNN model with isotonic regression correction Calibration Curve')
#plt.xlabel('Average of model predicted xG')
#plt.ylabel('Average of actual goal outcome')
#x = [0,1]
#y = [0,1]
#plt.plot(x,y, '--')
#plt.show()
#
#ll_cal = log_loss(y_test, calibrated)

#----------------------------------------------------------------


#-------------------CNN VISUALISATION PLOTTING---------------------------------------
for i in range(5): # plot a selection of input images from the test set
    
    # display original images   
    plt.imshow(x_test[i,:,:,2]) # 4th argument is the channel number so select 0, 1 or 2
    plt.show()
    


layer_dict = dict([(layer.name, layer) for layer in model.layers]) # dictionary of layers in model with names

# The dimensions of our input image
img_width = 40
img_height = 80

 #Our target layer: we will visualize the filters from this layer.
# See `model.summary()` for list of layer names, if you want to change this.
layer_name = "conv2d_1"


# Set up a model that returns the activation values for our target layer
layer = model.get_layer(name=layer_name)
feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=layer.output)

# the following is adapted from https://keras.io/examples/vision/visualizing_what_convnets_learn/

# function to calculate loss 
def compute_loss(input_image, filter_index):
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


# define gradient ascent function - NOTE THIS DEFINITELY REQUIRES TF v2
@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img

# initialise image with some noise
def initialize_image():
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, 3))

    return (img)

# create the visualisation - can play around with the number of iterations and learning rate
def visualize_filter(filter_index):
    # We run gradient ascent for 20 steps
    iterations = 30
    learning_rate = 10.0
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img

# convert image into viewable form
def deprocess_image(img):
    # Normalize array: center on 0.,
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15 # set variance - can play around with this value to crisp up images

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img


 # plot some maximally activated inputs!
for filter_no in range(64): # make sure this value matches number of filters in the layer selected
    loss, img = visualize_filter(filter_no)
    plt.show()
    plt.imshow(img[:,:,2])
    plt.show()
    print(filter_no)


