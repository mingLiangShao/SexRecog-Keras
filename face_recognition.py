#!/usr/bin/env python

import cv2
import numpy as np
import os,sys
from skimage import io
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

sys.path.insert(0, '/Users/shaomingliang/Downloads/BossSensor-master')
from boss_input import extract_data,resize_with_pad

#DatasetPath = []
#for i in os.listdir("yalefaces"):
#    DatasetPath.append(os.path.join("yalefaces", i))

imageData = []
imageLabels = []

imageData1, imageLabels= extract_data('/Users/shaomingliang/Downloads/BossSensor-master/data')
imageData = []
labels = []
for index,img in enumerate(imageData1):
    if img is None:
        continue
    frame = cv2.resize(img, (64, 64))
    imageData.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    labels.append(imageLabels[index])
imageDataFin = imageData
imageLabels = labels

#for i in DatasetPath:
#    imgRead = io.imread(i,as_grey=True)
#    imageData.append(imgRead)
#    
#    labelRead = int(os.path.split(i)[1].split(".")[0].replace("subject", "")) - 1
#    imageLabels.append(labelRead)

#faceDetectClassifier = cv2.CascadeClassifier("/Users/shaomingliang/sml/core/opencv-2.4.13/data/haarcascades/haarcascade_frontalface_default.xml")
#imageDataFin = []
#for i in imageData:
#    #cv2.imshow('sml', np.array(i))
#    #cv2.waitKey(100)
#    facePoints = faceDetectClassifier.detectMultiScale(np.array(i), scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
#    #print('len:%s' %len(facePoints))
#    if len(facePoints)>0:
#        x, y = facePoints[0][0:2]
#        width, height = facePoints[0][2:4]
#        image = i[y - 10: y + height, x: x + width]
#        cv2.imshow('sml', image)
#        cv2.waitKey(100)
#        imageDataFin.append(image)
#        #x,y = facePoints[0][:2]
#        #cropped = i[y: y + 150, x: x + 150]
#        #imageDataFin.append(cropped)
#for img in imageDataFin:
#    print(img.shape) 
#    cv2.imshow('sml', img)
#    cv2.waitKey(100)

c = np.array(imageDataFin)
#print(c.shape)
#print(np.array(imageLabels).shape)

X_train, X_test, y_train, y_test = train_test_split(np.array(imageDataFin),np.array(imageLabels), train_size=0.9, random_state = 20)
X_train = np.array(X_train)
X_test = np.array(X_test)
print('xx')
print(X_train.shape)
print(X_test.shape)
print('xx')


nb_classes = 2
y_train = np.array(y_train) 
y_test = np.array(y_test)


Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print(X_train.shape)
print(Y_test.shape)
train_samples = 2480
X_train = X_train.reshape(train_samples, 64*64)
X_test = X_test.reshape(276, 64*64)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

model = Sequential()
model.add(Dense(512,input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=64, nb_epoch=50, verbose=1, validation_data=(X_test, Y_test))
model.save('/Users/shaomingliang/Downloads/BossSensor-master/store/sex.h5')

loss, accuracy = model.evaluate(X_test,Y_test, verbose=0)
print "loss:%s" %loss
print "accuracy:%s" %accuracy








