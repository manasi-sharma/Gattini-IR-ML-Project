# run this program in Google colab


# linking to images in Google drive
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}


# linking to Google drive (contd.)
!mkdir -p drive
!google-drive-ocamlfuse drive


# checking contents of the directories in the Drive
!ls /content/drive/Summer_19/jpeg_trial
!ls /content/drive/Summer_19/jpeg_trial/real
!ls /content/drive/Summer_19/jpeg_trial/bogus


# importing correct directories
import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

import os
import cv2
import random
import cv2


# initial configuration for image parameters
#The number of pixels in each dimension of an image.
img_size = IMG_SIZE

#The images are stored in one-dimensional arrays of this length.
img_size_flat = img_size*img_size

#Tuple with height and width of images used to reshape arrays.
img_shape = [img_size, img_size]

#Number of classes, one class for each of 10 digits.
num_classes = len(CATEGORIES)

#Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

CATEGORIES = ["bogus", "real"]

IMG_SIZE= 28


# creating training and testing sets (type <list>) from directories
def create_dataset(DATADIR):
    dataset= []
    for category in CATEGORIES:  # do bogus and real

        path = os.path.join(DATADIR,category)  # create path to bogus and real
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=bogus 1=real

        for img in os.listdir(path):  # iterate over each image per bogus and real
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
                #print(img_array.shape)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                dataset.append([new_array, class_num])  # add this to our training_data
                #dataset.append([new_array, class_num, num_channels])  # add this to our training_data
        #print(len(training_data))
        return dataset

training_set= create_dataset('/content/drive/Summer_19/jpeg_trial/')
testing_set= create_dataset('/content/drive/Summer_19/testing')


# function to separate the images and labels as separate labels from the training/testing sets
def dataset_splitter(dataset):
    images= []
    labels= []
    for entry in dataset:
        images.append(entry[0])
        labels.append(entry[1])
        #print(7)
    return images, labels
    

# creating image and label <lists>
images, labels= dataset_splitter(training_set)
new_images_1 = np.array(images)
new_labels = np.array(labels)


# re-shaping image dataset to add num_channels, so that it's in the form [number of images, image size, image size, number of channels] instead of [number of images, image size, image size]
from numpy import newaxis
new_images= new_images_1[:, :, :, newaxis]


# creating training and validation sets
random_state=2
#split data into training and validation data
from sklearn.model_selection import train_test_split
images_train, images_val, labels_train, labels_val=train_test_split(new_images,new_labels,test_size=0.1,random_state=random_state)
print(images_train.shape) # should be in the form [number of images, image size, image size, number of channels]: eg. (32, 28, 28, 1)


# import required keras classes for running the CNN
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import plot_model


# running the actual CNN
#Create a sequential model
model_CNN= models.Sequential()

#Conv Layer 1
model_CNN.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_size, img_size, num_channels)))
model_CNN.add(layers.MaxPooling2D((2, 2)))

#Conv Layer 2
model_CNN.add(layers.Conv2D(36, (3, 3), activation='relu'))
model_CNN.add(layers.MaxPooling2D((2, 2)))

#Fully-Connected Layer
model_CNN.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_CNN.add(layers.MaxPooling2D((2, 2)))

#Flatten Layer
model_CNN.add(layers.Flatten())

#Loss function
model_CNN.add(layers.Dropout(0.4)) #For regularization (to reduce overfitting)
model_CNN.add(layers.Dense(512, activation='relu'))
model_CNN.add(layers.Dense(1, activation='sigmoid')) #For binary classification

model_CNN.summary()
model_CNN.compile(loss='binary_crossentropy', optimizer=optimizers.adam(lr=1e-3), metrics=['acc'])

#plot_model(model_CNN, to_file='model_CNN.jpg')-------------- WHAT IS THIS SUPPOSED TO DO?



# running the model for 10 epochs
model_CNN.fit(images_train, labels_train, validation_data=(images_val, labels_val), epochs= 10, batch_size=10)
