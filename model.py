import csv, random, numpy as np, pandas as pd
import keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img, flip_axis, random_shift

def model_99linessteering(input_shape, use_saved_state=None):
    if use_saved_state: return keras.models.load_model(use_saved_state)
    
    dropout_rate = 0.5

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='elu', input_shape=input_shape))
    model.add(MaxPooling2D())

    model.add(Convolution2D(32, 3, 3, activation='elu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(32, 3, 3, activation='elu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(128, 3, 3, activation='elu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(1024, activation='elu'))
    model.add(Dropout(dropout_rate))

    model.add(Dense(512, activation='elu'))
    model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='linear'))

    return model

def model_nvidia(input_shape, use_saved_state=None):
    if use_saved_state: return keras.models.load_model(use_saved_state)

    dropout_rate = 0.4
    initial_distribution = 'normal'

    model = Sequential()

    model.add(BatchNormalization(mode=1, 
                                 axis=-1, 
                                 weights=None, 
                                 beta_init='zero', 
                                 gamma_init='one', 
                                 gamma_regularizer=None, 
                                 beta_regularizer=None, 
                                 input_shape = input_shape))
    # 24@
    model.add(Convolution2D(24,5,5, subsample = (2,2), border_mode ='valid', init=initial_distribution))
    model.add(ELU())

    # 36@
    model.add(Convolution2D(36,5,5, subsample = (2,2), border_mode ='valid', init=initial_distribution))
    model.add(ELU())
    model.add(Dropout(dropout_rate))

    # 48@
    model.add(Convolution2D(48,5,5, subsample = (2,2), border_mode ='valid', init=initial_distribution))
    model.add(ELU())
    model.add(Dropout(dropout_rate))

    # 64@
    model.add(Convolution2D(64,3,3, subsample = (2,2), border_mode ='valid', init=initial_distribution))
    model.add(ELU())
    model.add(Dropout(dropout_rate))

    #64@
    model.add(Convolution2D(64,3,3, subsample = (2,2), border_mode ='valid', init=initial_distribution))
    model.add(ELU())
    model.add(Dropout(dropout_rate))

    model.add(Flatten())

    # Fully-connected layers
    model.add(Dense(1164, init=initial_distribution))
    model.add(Dropout(dropout_rate))
    model.add(ELU())

    model.add(Dense(100, init=initial_distribution))
    model.add(Dropout(dropout_rate))
    model.add(ELU())

    model.add(Dense(50, init=initial_distribution))
    model.add(Dropout(dropout_rate))
    model.add(ELU())

    model.add(Dense(10, init=initial_distribution))
    model.add(ELU())

    model.add(Dense(1, init=initial_distribution))

    return model

def preprocess_image(image):
    return image

from sklearn.model_selection import train_test_split
import csv

def import_and_split_csv_data(data_path):
    data = []
    with open(data_path) as file:
        reader = csv.reader(file)
        for line in reader:
            data.append(line)

    split = 0.8
    training_data, validation_data = train_test_split(data
                    ,train_size = split
                    ,random_state = 1)
    return training_data, validation_data

import cv2
import numpy as np
import sklearn
import os

def data_generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road

            X_data = np.array(images)
            y_data = np.array(angles)
            yield sklearn.utils.shuffle(X_data, y_data)
            
#main

#row, col, depth = 320, 160, 3

#model = model_99linessteering((row, col, depth))
#model.compile(loss='mse', optimizer="adam")
#model.summary()

train_data, val_data = import_and_split_csv_data('./data/driving_log.csv')

train_data_gen = data_generator(train_data)
for i in range(10):
    X, y = next(train_data_gen)
    print(np.shape(X), np.shape(y))