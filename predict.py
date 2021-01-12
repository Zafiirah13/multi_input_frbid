#!/usr/bin/env python
# coding: utf-8

"""
Authors : Zafiirah Hosenie
Email : zafiirah.hosenie@gmail.com or zafiirah.hosenie@postgrad.manchester.ac.uk
Affiliation : The University of Manchester, UK.
License : MIT
Status : Under Development
Description :
Python implementation for FRBID: Fast Radio Burst Intelligent Distinguisher.
This code is tested in Python 3 version 3.5.3  
"""
#------------------------------------------------------------------------------------------------#
# # FRBID prediction phase on new candidate files
#------------------------------------------------------------------------------------------------#

import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd

import matplotlib.pylab as plt
from keras.utils import np_utils
from time import gmtime, strftime
from frbid_code.util import makedirs, ensure_dir
from frbid_code.prediction_phase import load_candidate, FRB_prediction
from frbid_code.model import compile_model,model_save 
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-dd','--data_dir', help='The directory where the hdf5 candidates are located',type=str,default='./data/testing_set/')
parser.add_argument('-rd','--result_dir',help='The directory where the csv file after prediction will be saved',type=str,default='./data/results_csv/')
parser.add_argument('-m','--model_cnn_name',help='The network name choose from: MULTIINPUT', type=str,default='MULTIINPUT')
parser.add_argument('-p', '--probability', help='Detection threshold', default=0.5, type=float)
args = parser.parse_args()
data_dir, result_dir, model_cnn_name, probability = args.data_dir, args.result_dir, args.model_cnn_name, args.probability


#------------------------------------------------------------------------------------------------#
# # Load the new candidates
# - data_dir: The directory that contains the hdf5 files
# - n_images: can either take str 'dm_fq_time', 'dm_time', 'fq_time'
#------------------------------------------------------------------------------------------------#

test_dm, test_freq, ID_test    = load_candidate(data_dir = data_dir, n_images = 'dm_time_fq_time')

print("Total number of candidate instances: {}".format(str(len(ID_test))))
print("The Shape of the DM test set is {}".format(test_dm.shape))
print("The Shape of the Freq test set is {}".format(test_freq.shape))

#------------------------------------------------------------------------------------------------#
# # Prediction on new candidate files
# Here we will load the pre-existing train model using the parameter 
# INPUT:
# - model_name: 'MULTIINPUT'
# - X_test_dm, X_test_freq : Image data should have shape (Nimages,256,256,1). This will vary depending on the criteria one use for n_images.
# - ID: The candidate filename
# - result_dir: The directory to save the csv prediction file
# 
# OUTPUT:
# - overall_real_prob: An array of probability that each source is FRB. Value will range between [0 to 1.0]
# - overall_dataframe: A table with column candidate name of all sources and its associated probability that it is a FRB source and its labels
#------------------------------------------------------------------------------------------------#

overall_real_prob, overall_dataframe = FRB_prediction(model_name=model_cnn_name, X_test_dm=test_dm, X_test_freq=test_freq, ID=ID_test,result_dir=result_dir,probability=probability)
print('Prediction completed and is found at {}'.format(str(result_dir)))





