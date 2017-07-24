import csv, random, os, cv2 as cv2, numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Cropping2D, ELU, BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img, flip_axis, random_shift
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import load_model

def model_input_layer(model, input_shape, cropping, preprocess_func):
    model.add(Cropping2D(cropping=cropping, input_shape=input_shape))
    model.add(Lambda(preprocess_func, input_shape=input_shape))
    return model

def model_99linessteering(input_shape, cropping, preprocess_func): 
    dropout_rate = 0.5

    model = Sequential()
    model = model_input_layer(model, input_shape, cropping, preprocess_func)

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

def model_nvidia(input_shape, cropping, preprocess_func):
    dropout_rate = 0.4
    initial_distribution = 'normal'

    model = Sequential()
    model = model_input_layer(model, input_shape, cropping, preprocess_func)
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
    return (image / 255.0) - 0.5

def import_and_split_csv_data(data_path, steering_corr, train_size):
    X, y = [], []
    with open(data_path) as file:
        reader = csv.reader(file)
        next(reader)
        for center, left, right, steering, throttle, brake, speed in reader:
            #X += [center, left, right]
            #y += [float(steering),
            #      float(steering) + steering_corr, 
            #      float(steering) - steering_corr]
            #X += [center]
            #y += [float(steering)]
            #X += [left]
            #y += [float(steering) + steering_corr]
            X += [right]
            y += [float(steering) - steering_corr]
            
        return train_test_split(X, y, train_size = train_size, random_state = 1)

def extract_sample(X, y, idx):
    name = './data/IMG/' + X[idx].split('/')[-1]
    image = img_to_array(load_img(name))
    angle = y[idx]
    return image, angle

def batched_sample_generator(X, y, batch_size=32):
    num_samples = len(X)
    assert(len(X) == len(y))

    #print("number of samples = {}, batch size = {}".format(num_samples, batch_size))
     
    while 1: # Loop forever so the generator never terminates
        shuffle(X, y)
        for offset in range(0, num_samples, batch_size):
            images = []
            angles = []
            for idx in range(offset, min(offset+batch_size, num_samples)):
                image, angle = extract_sample(X, y, idx)
                images.append(image)
                angles.append(angle) 

            X_data = np.array(images)
            y_data = np.array(angles)

            yield shuffle(X_data, y_data)


def train_model(model, steering_corr, train_size, batch_size=32, epochs=5, verbose=1):
    Xt,Xv, yt,yv = import_and_split_csv_data('./data/driving_log.csv',
                                             steering_corr=steering_corr,
                                             train_size=train_size)
    len_train_data = len(Xt)
    len_val_data = len(Xv)

    print("training = {},  validation = {}".format(len_train_data, len_val_data))

    train_data_gen = batched_sample_generator(Xt, yt, batch_size)
    val_data_gen = batched_sample_generator(Xv, yv, batch_size)

    #for i in range(0, len_train_data, batch_size):
    #    X, y = next(train_data_gen)
    #    print("{}/{}: {}, {}".format(i, len_train_data, len(X), len(y)))
    #
    #return model, 1
    
    history = model.fit_generator(train_data_gen, 
                                  samples_per_epoch = len_train_data,
                                  validation_data = val_data_gen,
                                  nb_val_samples = len_val_data,
                                  nb_epoch=epochs, 
                                  verbose=verbose)
    return model, history

import matplotlib.pyplot as plt

def draw_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

#main

row, col, depth = 160, 320, 3
cropping = ((40, 25), (0, 0))
batch_size = 32
epochs = 5

model_99 = model_99linessteering((row, col, depth), cropping, preprocess_image)
model_99.compile(loss='mse', optimizer="adam")
model_99.summary()
model_99, history = train_model(model_99,
                                steering_corr=0.,#0.05,
                                train_size=0.8,
                                batch_size=batch_size, 
                                epochs=epochs,
                                verbose=2)
model_99.save('model_99.h5')

