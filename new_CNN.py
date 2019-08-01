# Run this program in Google colab


# Linking to images in Google drive
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


# Linking to Google drive (contd.)
!mkdir -p drive
!google-drive-ocamlfuse drive


# Checking contents of the directories in the Drive
!ls /content/drive/Summer_19/jpeg_trial
!ls /content/drive/Summer_19/jpeg_trial/real
!ls /content/drive/Summer_19/jpeg_trial/bogus


# Importing correct directories
import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from scipy import ndimage

import os
import cv2
import random
import cv2


# Initial configuration for image parameters
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


# Creating training and testing sets (type <list>) from directories, and doing data augmentation by rotating images by 90, 180, 270 degrees and flipping on the x and y axes
CATEGORIES = ["bogus", "real"]

def create_dataset(DATADIR):
    dataset= []
    real_count= 0
    bogus_count= 0
    flag= 0
    count= 0
      
    for category in CATEGORIES:  # do bogus and real

        path = os.path.join(DATADIR,category)  # create path to bogus and real
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=bogus 1=real

        for img in os.listdir(path):  # iterate over each image per bogus and real
          
                print(count)
                count= count+1
            
                img_array= np.load(os.path.join(path,img))
                if(img_array.shape!=(61,61)):
                    continue
                else:                  
                    # create all the transformations
                    
                    # 90 degrees rotation
                    img_90 = ndimage.rotate(img_array, 90)
                    
                    # 180 degrees rotation
                    img_180 = ndimage.rotate(img_array, 180)
                    
                    # 270 degrees rotation
                    img_270 = ndimage.rotate(img_array, 270)
                    
                    # flip in up-down (vertial) direction
                    img_v= np.flipud(img_array)
                    
                    # flip in left-right (horizontal) direction
                    img_h= np.fliplr(img_array)
                    
                    """fig = plt.figure(figsize=(10, 3))
                    ax1, ax2, ax3, ax4, ax5, ax6 = fig.subplots(1, 6)
                    ax1.imshow(img_array, cmap='gray')
                    ax1.set_axis_off()
                    ax2.imshow(img_90, cmap='gray')
                    ax2.set_axis_off()
                    ax3.imshow(img_180, cmap='gray')
                    ax3.set_axis_off()
                    ax4.imshow(img_270, cmap='gray')
                    ax4.set_axis_off()
                    ax5.imshow(img_v, cmap='gray')
                    ax5.set_axis_off()
                    ax6.imshow(img_h, cmap='gray')
                    ax6.set_axis_off()
                    fig.set_tight_layout(True)
                    plt.show()
                    
                    break"""                    
                    
                    dataset.append([img_array, class_num])  # add this to our training_data
                    dataset.append([img_90, class_num])  # add 90 degrees rotation to our training_data
                    dataset.append([img_180, class_num])  # add 180 degrees rotation to our training_data
                    dataset.append([img_270, class_num])  # add 270 degrees rotation to our training_data
                    dataset.append([img_v, class_num])  # add vertical flip to our training_data
                    dataset.append([img_h, class_num])  # add horizontal flip to our training_data
                    
                    
                    if(class_num==0): # 0 for bogus, 1 for real
                        bogus_count= bogus_count+6
                    else:
                        real_count= real_count+6
        
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


# Function to separate the images and labels as separate labels from the training/testing sets
def dataset_splitter(dataset):
    images= []
    labels= []
    for entry in dataset:
        images.append(entry[0])
        labels.append(entry[1])
    return images, labels
    

# Creating image and label <lists>
images, labels= dataset_splitter(training_set)
new_images_1 = np.array(images)
new_labels = np.array(labels)


# Re-shaping image dataset to add num_channels, so that it's in the form [number of images, image size, image size, number of channels] instead of [number of images, image size, image size]
from numpy import newaxis
new_images= new_images_1[:, :, :, newaxis]


# Creating training and testing sets
random_state=2
#split data into training and test data
from sklearn.model_selection import train_test_split
images_train, images_test, labels_train, labels_test=train_test_split(new_images,new_labels,test_size=0.2,random_state=random_state)
print(images_train.shape) # should be in the form [number of images, image size, image size, number of channels]: eg. (32, 28, 28, 1)


# Function to plot sample real and bogus images
def plot_images(images, cls_true, cls_pred=None):

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
    

# Function to produce separate real and bogus training sets (for plotting)
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


# Plotting sample real and bogus images
real_im, real_l, bogus_im, bogus_l= RB_image_splitter(images, labels)
plot_images(images=bogus_im, cls_true=bogus_l)
plot_images(images=real_im, cls_true=real_l)


# Import required keras classes for running the CNN
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import plot_model


# Running the actual CNN
#Create a sequential model
model_CNN= models.Sequential()

#Conv Layer 1
model_CNN.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, num_channels)))
model_CNN.add(layers.MaxPooling2D((2, 2)))

#Conv Layer 2
model_CNN.add(layers.Conv2D(64, (5, 5), activation='relu')) #can also be 3x3
model_CNN.add(layers.MaxPooling2D((2, 2)))

"""#Fully-Connected Layer
model_CNN.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_CNN.add(layers.MaxPooling2D((2, 2)))"""

#Flatten Layer
model_CNN.add(layers.Flatten())

#Loss function
model_CNN.add(layers.Dropout(0.25)) #For regularization (to reduce overfitting)
model_CNN.add(layers.Dense(2048, activation='relu'))
model_CNN.add(layers.Dense(1, activation='sigmoid')) #For binary classification

model_CNN.summary()
model_CNN.compile(loss='binary_crossentropy', optimizer=optimizers.adam(lr=1e-3), metrics=['acc'])


# To save picture of flowchart of model to drive 
plot_model(model_CNN, to_file='/content/drive/Summer_19/model_plots/2_layer_model_CNN.png')


# Loading TensorBoard in Notebook
logdir= os.path.join("/content/drive/Summer_19/nparrays_tensor_board", datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)

#%load_ext tensorboard
%reload_ext tensorboard
%tensorboard --logdir /content/drive/Summer_19/nparrays_tensor_board


# Fitting and saving model for data
#history = model_CNN.fit(images_train, labels_train, validation_data=(images_val, labels_val), epochs= 500, batch_size=30, callbacks=[tensorboard_callback])
history = model_CNN.fit(images_train, labels_train, validation_split=0.2, epochs= 100, batch_size=30, callbacks=[tensorboard_callback])

# Printing average loss and accuracy values
print("Average validation loss: ", np.average(history.history['val_loss']))
print("Average validation accuracy: ", np.average(history.history['val_acc']))

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Predicting classes for validation set
predictions= model_CNN.predict(images_val, batch_size=30, verbose=0, steps=None, callbacks=[tensorboard_callback])

# Creating confusion matrix
length= len(predictions)
labels_test_resized= np.reshape(labels_test,(length, 1)) # converting the 1D labels_test[] array to a 2D array, to make it the same shape as predictions[]
cm= confusion_matrix(labels_test_resized, predictions)

# Plotting the confusion matrix and normalized confusion matrix
# taken from https://www.kaggle.com/grfiv4/plot-a-confusion-matrix

import itertools

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    

 #running the function
 plot_confusion_matrix(cm = cm, normalize = False, target_names = ['Bogus', 'Real'], title = "Confusion Matrix")
 plot_confusion_matrix(cm = cm, normalize = True, target_names = ['Bogus', 'Real'], title = "Confusion Matrix")


# Evaluating accuracy and loss on the test set
results = model_CNN.evaluate(images_test, labels_test, batch_size=30, verbose=1)
loss = float(results[0])
accuracy = float(results[1])
print("Loss = " + str(loss))
print("Test Accuracy = " + str(accuracy))


# Saving model in JSON format to Drive 
# taken from https://machinelearningmastery.com/save-load-keras-deep-learning-models/
from keras.models import model_from_json

# evaluate the model
scores = model_CNN.evaluate(images_test, labels_test, verbose=0)
print("%s: %.2f%%" % (model_CNN.metrics_names[1], scores[1]*100))
 
# serialize model to JSON
model_json = model_CNN.to_json()
with open("/content/drive/Summer_19/JSON_files/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_CNN.save_weights("/content/drive/Summer_19/JSON_files/model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('/content/drive/Summer_19/JSON_files/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/content/drive/Summer_19/JSON_files/model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = model_CNN.evaluate(images_test, labels_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# Predicting probabilities for each of the values in the test set
probs = model_CNN.predict_proba(images_test)
probs_reshaped= np.reshape(probs,(length))

# Making an array to index probabilities in the pandas Dataframe, to then plot histograms for the confusion matrix values
index= np.arange(length)
print(index)

# Plotting histograms for all four confusion matrix options (TP, TN, FP, FN)
# taken from https://github.com/DistrictDataLabs/yellowbrick/issues/749
import pandas as pd

df_predictions = pd.DataFrame({'label': labels_test, 'probs': probs_reshaped, 'index': index})

print(type('label'[0]))

fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
# show true-pos 

bins = np.arange(0, 1.01, 0.1)

def show_quarter(df, query, col, title, ax, bins, x_label=None, y_label=None, autoscale=False):
    results = df.query(query)
    results[col].hist(ax=ax, bins=bins); 
    if y_label:
        ax.set_ylabel(y_label)
    if x_label:
        ax.set_xlabel(x_label)
    ax.set_title(title + " ({})".format(results.shape[0])) #IANBOB
    if(autoscale==True):
        print("yo2")      
        #ax.set_ylim([0,50])
    else:
        print("yo")
        #plt.show()
show_quarter(df_predictions, "label==0 and probs < 0.5", "probs", "True Negative", axs[0][0], bins, y_label="Bogus")
show_quarter(df_predictions, "label==0 and probs >= 0.5", "probs", "False Positive", axs[0][1], bins, autoscale=True)
show_quarter(df_predictions, "label==1 and probs >= 0.5", "probs", "True Positive", axs[1][1], bins, x_label="Real")
show_quarter(df_predictions, "label==1 and probs < 0.5", "probs", "False Negative", axs[1][0], bins, x_label="Bogus", y_label="Real", autoscale=True)
fig.suptitle("Probabilities per Confusion Matrix cell");


# Finding extreme outliers
query= "label==1 and probs==0.0"
results = df_predictions.query(query)
real_outliers= results['index'].values
real_outliers_len= len(real_outliers)
print(real_outliers)

query= "label==0 and probs==1.0"
results = df_predictions.query(query)
bogus_outliers= results['index'].values
bogus_outliers_len= len(bogus_outliers)
print(bogus_outliers)


# Plotting images of 4 extreme outliers (TN and FP)
def plot_images_outliers(images, cls_true, cls_pred=None):

    """print('length of images: '+str(len(images)))
    print('length of cls: '+str(len(cls_true)))"""

    assert len(images) == len(cls_true)

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(2,2)
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
    
    
    # Running the above plotting function
    print("These images are labelled as real but predicted as bogus:")

real_images= []
real_labels= np.full((real_outliers_len), 1)
for i in real_outliers:
    img= images_test[i-1]
    real_images.append(img)
    
plot_images_outliers(images=real_images[0:4], cls_true=real_labels[0:4])
#plot_images_individual(images=real_images[4:5], cls_true=real_labels[4:5])


print("These images are labelled as bogus but predicted as real:")

bogus_images= []
bogus_labels= np.full(bogus_outliers_len, 0)  
for i in bogus_outliers:
    img= images_test[i-1]
    bogus_images.append(img)
    
plot_images_outliers(images=bogus_images[0:4], cls_true=bogus_labels[0:4])
