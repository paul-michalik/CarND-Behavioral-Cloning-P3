import csv, random, numpy as np
import keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img, flip_axis, random_shift

def model_99linessteering(input_shape, use_saved_state=None):
    if use_saved_state: return keras.models.load_model(use_saved_state)
    
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
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='linear'))
    #model.compile(loss='mse', optimizer="adam")

    return model

def model_nvidia(input_shape, use_saved_state=None):
    if use_saved_state: return keras.models.load_model(use_saved_state)

    # Keras Implementation of NVIDIA Model
    # drop out for regulazation
    drop_out_rate = 0.4

    # initial distribution of weights
    initial_distribution = 'normal'

    # building a sequential Keras model

    model = Sequential()

    # debugging
    # print ("DOR: " + str(drop_out_rate))
    # print ("Distribution: " + str(initial_distribution))
    # print ("Image shape: " + str(image_shape))

    # Normalizing
    model.add(BatchNormalization(epsilon=0.001, 
                                 mode=1, 
                                 axis=-1, 
                                 weights=None, 
                                 beta_init='zero', 
                                 gamma_init='one', 
                                 gamma_regularizer=None, 
                                 beta_regularizer=None, 
                                 input_shape = image_shape))

    # Convolutional feature maps:
    # 24@
    model.add(Convolution2D(24,5,5, subsample = (2,2), border_mode ='valid', init=initial_distribution))
    model.add(ELU())

    # 36@
    model.add(Convolution2D(36,5,5, subsample = (2,2), border_mode ='valid', init=initial_distribution))
    model.add(ELU())
    model.add(Dropout(drop_out_rate))

    # 48@
    model.add(Convolution2D(48,5,5, subsample = (2,2), border_mode ='valid', init=initial_distribution))
    model.add(ELU())
    model.add(Dropout(drop_out_rate))

    # 64@
    model.add(Convolution2D(64,3,3, subsample = (2,2), border_mode ='valid', init=initial_distribution))
    model.add(ELU())
    model.add(Dropout(drop_out_rate))

    #64@
    model.add(Convolution2D(64,3,3, subsample = (2,2), border_mode ='valid', init=initial_distribution))
    model.add(ELU())
    model.add(Dropout(drop_out_rate))

    model.add(Flatten())

    # Fully-connected layers
    model.add(Dense(1164, init=initial_distribution))
    model.add(Dropout(drop_out_rate))
    model.add(ELU())

    model.add(Dense(100, init=initial_distribution))
    model.add(Dropout(drop_out_rate))
    model.add(ELU())

    model.add(Dense(50, init=initial_distribution))
    model.add(Dropout(drop_out_rate))
    model.add(ELU())

    model.add(Dense(10, init=initial_distribution))
    model.add(ELU())

    model.add(Dense(1, init=initial_distribution))

    # debugging
    # model.summary()
    # plot(model, to_file='model.png', show_shapes=True)

    return model

row, col, depth = 320, 160, 3
model = model_99linessteering((row, col, depth))
model.compile(loss='mse', optimizer="adam")