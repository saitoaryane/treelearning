import cv2
import numpy as np
from matplotlib import pyplot as plt
import mahotas
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.neighbors import KNeighborsClassifier

# function to extract hist from an image

def extract_features(image):

    color = ('b','g','r')

    #cv2.imshow('image', image)
    for i,col in enumerate(color):
        histr = cv2.calcHist([image],[i],None,[256],[0,256])
        hist= np.concatenate(histr)

    return hist

# load the training dataset
train_path  = "./Fotos/Fotos/Dataset2/train/"
train_names = os.listdir(train_path)
print (train_names)

# empty list to hold feature vectors and train labels
train_features = []
train_labels   = []

# loop over the training dataset
print ("[STATUS] Started extracting features..")
for train_name in train_names:
        cur_path = train_path + "/" + train_name
        cur_label = train_name
        i = 1
        for file in glob.glob(cur_path + "/*.jpeg"):
                #print "Processing Image - {} in {}".format(i, cur_label)
                # read the training image
                image = cv2.imread(file)              

              
                # extract haralick texture from the image
                features = extract_features(image)
                #print features

                # append the feature vector and label
                train_features.append(features)
                #print features
                train_labels.append(cur_label)

                # show loop update
                i += 1
# have a look at the size of our feature vector and labels
print ("Training features: {}".format(np.array(train_features).shape))
print ("Training labels: {}".format(np.array(train_labels).shape))


# create the classifier
print ("[STATUS] Creating the classifier..")
modelN= KNeighborsClassifier()

# fit the training data and labels
print ("[STATUS] Fitting data/label to model..")
modelN.fit(train_features, train_labels)
predict= []
# loop over the test images

for file in glob.glob("./Fotos/Fotos/Dataset2/*.jpeg"):
    # read the input image
    image = cv2.imread(file)

    features = extract_features(image)
 
	# evaluate the model and predict label
    prediction = modelN.predict(features.reshape(1, -1))[0]

	# show the label
    cv2.putText(image, prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
    print ("Prediction - {}".format(prediction))
    predict.append(prediction)

	 
print (predict)