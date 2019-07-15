import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
#from tqdm import tqdm

training_data = []


DATADIR = "jpeg_trial"

CATEGORIES = ["bogus", "real"]

IMG_SIZE= 372

def create_training_data():
    for category in CATEGORIES:  # do bogus and real

        path = os.path.join(DATADIR,category)  # create path to bogus and real
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=bogus 1=real

        for img in os.listdir(path):  # iterate over each image per bogus and real
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
                #print(img_array.shape)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
        #print(len(training_data))

create_training_data()

print(len(training_data))

random.shuffle(training_data)

"""for sample in training_data[:10]:
    print(sample[1])"""

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


#outputting model
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

#loading model back in
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
