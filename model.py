import csv, random, os, numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import load_model

def resize(image, target_shape):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(image, target_shape)

def normalize(image):
    return (image / 255.0) - 0.5

def model_input_layer(model, input_shape, cropping, preprocess_func):
    model.add(Cropping2D(cropping=cropping, input_shape=input_shape))
    model.add(Lambda(preprocess_func, input_shape=input_shape))
    return model

# Model is inspired by https://blog.coast.ai/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a
def model_99linessteering(input_shape, cropping, preprocess_func): 
    dropout_rate = 0.5

    model = Sequential()
    model = model_input_layer(model, input_shape, cropping, preprocess_func)

    model.add(Convolution2D(32, 2, 2, activation='elu', input_shape=input_shape))
    model.add(MaxPooling2D())

    model.add(Convolution2D(32, 2, 2, activation='elu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(32, 2, 2, activation='elu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(64, 2, 2, activation='elu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(128, 2, 2, activation='elu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(1024, activation='elu'))
    model.add(Dropout(dropout_rate))

    model.add(Dense(512, activation='elu'))
    model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='linear'))

    return model

def import_and_split_csv_data(data_path, steering_corr, train_size):
    X, y = [], []
    with open(data_path) as file:
        reader = csv.reader(file)
        next(reader)
        for center, left, right, steering, throttle, brake, speed in reader:
            steering_angle = float(steering)
            # Removes half of angles less than 0.1 radians in order to reduce bias towards driving straight
            if abs(steering_angle) <= 0.1 and random.randint(0, 9) <= 4:
                continue

            X += [center]
            y += [steering_angle]

            # Only add left and right camera images if angles > 0.1.
            if abs(steering_angle) > 0.1:
                X += [left, right]
                y += [float(steering) + steering_corr, 
                      float(steering) - steering_corr]
            
        return train_test_split(X, y, train_size = train_size, random_state = 1)

def extract_sample(X, y, idx):
    name = './data/IMG/' + X[idx].split('/')[-1]
    image = img_to_array(load_img(name))
    angle = y[idx]
    return image, angle

def batched_sample_generator(X, y, batch_size=32):
    num_samples = len(X)

    assert(len(X) == len(y))
     
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

def preprocess(img):
    return normalize(resize(img, (trow, tcol)))

def build_and_train_model():
    row, col, depth = 160, 320, 3
    cropping = ((60, 25), (0, 0))
    trow, tcol = 64, 64
    batch_size = 32
    epochs = 3

    # For some weird reason, Keras is unable to load a model with lambda
    # layer which uses the "preprocess" function above...  :-/
    model_99 = model_99linessteering(input_shape=(row, col, depth), 
                                     cropping=cropping,
                                     preprocess_func=normalize)

    model_99.compile(loss='mse', optimizer="adam")
    
    return train_model(model_99,
                       steering_corr=0.2,
                       train_size=0.8,
                       batch_size=batch_size, 
                       epochs=epochs,
                       verbose=2)
    

if __name__ == '__main__':
    model99, history = build_and_train_model()
    model99.summary()
    model99.save('model_99.h5')


    

