# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model-visualization]: ./writeup-images/model-visualization.png "Model Visualization"
[loss-curves]: ./writeup-images/loss-curves.png "Loss curves"
[steering-data-distrib]: ./writeup-images/steering-data-distrib.png "Steering angle distribution"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 2x2 filter sizes and depths between 32 and 128 (model.py lines 23-54) 

The model includes ELU activation layers to introduce nonlinearity. The data is preprocessed in the input layers (function ``model_input_layer``) using Keras ``Cropping2D`` layer and Keras lambda layer (code lines 17-20). I have had difficulties to load a model which contained both normalization and resizing functions so I haven't resized the input. 

#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers after each fully connected layer in order to reduce overfitting (model.py lines 47 and 50). The dropout rate was chosen to be equal to 0.5.

The model was trained and validated on different data sets to ensure that the model was not overfitting (function ``import_and_split_csv_data`` code lines 56 - 76). The data was splitted and into training and validation test using ``sklearn.model_selection.train_test_split``. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 152).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the data set provided by Udacity as combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a simplified version of the NVidia network recommended in the lectures.

My first step was to use a convolution neural network model similar to the ["deep learning model in 99 lines of code"](https://blog.coast.ai/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a). I thought this model might be appropriate because it appeared to have a structure which seemd to fit the problem well - not too many and too complex convolution layers for image processing and a dense "trail" suitable for this regression problem.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The model trained and validated well in 3 epochs taking approximately 15 seconds each on a Windows 10 Pro laptop with GeForce GTX 1050Ti.

The final step was to run the simulator to see how well the car was driving around track one. During the first few seconds the vehicle drove smoothly and continuously. However, there were two spots where the vehicle fell off the track:

1. The sharp left turn after the bridge
2. The sharp right turn afterwards

To improve the driving behavior in these cases, I started to modify and augment the data. I assumed that the distribution of steering angles in the training data (see fig. below) caused a bias towards driving in straight direction and started to randomly filter 50% of records where the steering angle was less than 0.1 radian. I also added the images from the left and right camera but only if the steering angles exceeded 0.1 radians. I have applied the correction factor of 0.2 radian for steering angle which made the vehicle go well through the sharp turns although the driving in the straight direction became less smooth.

![Distribution of steering angles in the training data set][steering-data-distrib]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network wit the layers and layer sizes shown in figure below. Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![Visualization of the architecture][model-visualization]

#### 3. Creation of the Training Set & Training Process

After the collection process, I had approx. 12K number of data points which were split into training and validation using the test size = 0.8 and training size = 0.2. I then preprocessed this data by applying a Cropping, Resizing and Normalization as a part of the model in order to prevent modifications in the ``drive.py`` module. However I have encountered difficulties with loading a model which used the ``resize`` function in the Lambda layer. Keras could not load the model although the training went quicker and smoother. I finally randomly shuffled the data set and put 20% of the data into a validation set, see function ``import_and_split_csv_data``.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The good number of epochs was 3 as evidenced by the loss curves shown in the figure below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![Loss curve after 3 epochs during the training of the model][loss-curves]