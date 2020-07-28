# ConvNets
Exploring the use of Convolutional Neural Networks in football data analysis

This repo will host some exploratory work looking at the feasibility of creating models with ConvNets for use with football data that are comparable to current published state-of-the-art.

The first test case is xG, with the write up of findings found [here](https://www.opengoalapp.com/xg-with-cnns-full-study)

This repo currently contains the following Python files:

* FFConv_xGModel.py - the main script used to perform the evaluation containing the final CNN model. There are also various plotting functions for the graphs seen in the write-up.

* HPtune_xGModel.py - script untilising keras tuner to search for optimal hyperparameters for the model

* LoadEvents.py - a utility function to load in StatsBomb open data from the API for a number of event categories. THIS FUNCTION WILL BY DEFAULT LOAD THE WHOLE SET SO CAUTION/PATIENCE IS ADVISED. To get started playing around with the CNN model it is recommended to load in the pre-proecessed data taken from a June snapshot of the StatsBomb open data set.

* ShotsToArrays.py - a utility function to take shots data returned from the above script and process into freeze frames split into 3 channels and returned as arrays plus the outcome of the shot and other data. This script is only called when loading in data using LoadEvents.py.

This code has been tested on Tensorflow 2.2.0. The network takes around 5 minutes to train with a low end GPU.

Please feel free to log any issues and I will do my best to help.
 
