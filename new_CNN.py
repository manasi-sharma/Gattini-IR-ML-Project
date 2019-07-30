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
img_size = IMG_SIZE = 61

#The images are stored in one-dimensional arrays of this length.
img_size_flat = img_size*img_size

#Tuple with height and width of images used to reshape arrays.
img_shape = [img_size, img_size]

#Number of classes, one class for each of 10 digits.
num_classes = len(CATEGORIES)

#Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1


# creating training and testing sets (type <list>) from directories
CATEGORIES = ["bogus", "real"]

def create_dataset(DATADIR):
    dataset= []
    real_count= 0
    bogus_count= 0
    flag= 0
      
    for category in CATEGORIES:  # do bogus and real

        path = os.path.join(DATADIR,category)  # create path to bogus and real
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=bogus 1=real

        for img in os.listdir(path):  # iterate over each image per bogus and real
                img_array= np.load(os.path.join(path,img))
                if(img_array.shape!=(61,61)):
                    continue
                else:
                    dataset.append([img_array, class_num])  # add this to our training_data
                    
                    if(class_num==0): # 0 for bogus, 1 for real
                        bogus_count= bogus_count+1
                    else:
                        real_count= real_count+1
        
        #return dataset, bogus_count, real_count 
        if(flag==0):
            flag= flag+1
        else:
            return dataset, bogus_count, real_count

training_set, bogus_count, real_count= create_dataset('/content/drive/Summer_19/nparrays_training_set/')
print("DATASET SHAPE:")
print(len(training_set))
print("REAL COUNT:")
print(real_count)
print("BOGUS COUNT:")
print(bogus_count)


# function to separate the images and labels as separate labels from the training/testing sets
def dataset_splitter(dataset):
    images= []
    labels= []
    for entry in dataset:
        images.append(entry[0])
        labels.append(entry[1])
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


# function to plot sample real and bogus images
def plot_images(images, cls_true, cls_pred=None):

    """print('length of images: '+str(len(images)))
    print('length of cls: '+str(len(cls_true)))"""

    assert len(images) == len(cls_true)

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    

# function to produce separate real and bogus training sets (for plotting)
def RB_image_splitter(images, labels):
    real_im= []
    real_l= []
    bogus_im= []
    bogus_l= []
    count= 0
    for entry in labels:
        if(entry==0):
            bogus_im.append(images[count])
            bogus_l.append(entry)
        else:
            real_im.append(images[count])
            real_l.append(entry)
        count= count+1
    return real_im, real_l, bogus_im, bogus_l


# plotting sample real and bogus images
real_im, real_l, bogus_im, bogus_l= RB_image_splitter(images, labels)
plot_images(images=bogus_im, cls_true=bogus_l)
plot_images(images=real_im, cls_true=real_l)


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
model_CNN.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, num_channels)))
model_CNN.add(layers.MaxPooling2D((2, 2)))

#Conv Layer 2
model_CNN.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_CNN.add(layers.MaxPooling2D((2, 2)))

#Fully-Connected Layer
model_CNN.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_CNN.add(layers.MaxPooling2D((2, 2)))

#Flatten Layer
model_CNN.add(layers.Flatten())

#Loss function
model_CNN.add(layers.Dropout(0.25)) #For regularization (to reduce overfitting)
model_CNN.add(layers.Dense(2048, activation='relu'))
model_CNN.add(layers.Dense(1, activation='sigmoid')) #For binary classification

model_CNN.summary()
model_CNN.compile(loss='binary_crossentropy', optimizer=optimizers.adam(lr=1e-3), metrics=['acc'])


# to save picture of flowchart of model to drive 
plot_model(model_CNN, to_file='/content/drive/Summer_19/model_plots/2_layer_model_CNN.png')


# loading TensorBoard in Notebook
logdir= os.path.join("/content/drive/Summer_19/nparrays_tensor_board", datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)

#%load_ext tensorboard
%reload_ext tensorboard
%tensorboard --logdir /content/drive/Summer_19/nparrays_tensor_board


# fitting and saving model for data
history = model_CNN.fit(images_train, labels_train, validation_data=(images_val, labels_val), epochs= 500, batch_size=30, callbacks=[tensorboard_callback])

# printing average loss and accuracy values
print("Average validation loss: ", np.average(history.history['val_loss']))
print("Average validation accuracy: ", np.average(history.history['val_acc']))

# plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# predicting classes for validation set
predictions= model_CNN.predict(images_val, batch_size=30, verbose=0, steps=None, callbacks=[tensorboard_callback])

# creating confusion matrix
cm= confusion_matrix(labels_val, predictions)

# running the model for 10 epochs
model_CNN.fit(images_train, labels_train, validation_data=(images_val, labels_val), epochs= 10, batch_size=10)
