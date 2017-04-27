import sys,os
import cv2
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
sys.path.insert(0, '/Users/shaomingliang/Downloads/BossSensor-master')
from boss_input import extract_data,resize_with_pad

color = (255, 255, 255)
cascade_path = "/Users/shaomingliang/sml/core/opencv-2.4.13/data/haarcascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)
img = cv2.imread('/Users/shaomingliang/Downloads/BossSensor-master/data.bak/boss/220.jpg')
model = load_model('/Users/shaomingliang/Downloads/BossSensor-master/store/model.h5')
cap = cv2.VideoCapture(0)
while True:    
    s, frame = cap.read()
    if not s:
        continue
    img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facerect = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(20, 20))
    if len(facerect) <= 0:
        cv2.imshow("capture", frame)
        cv2.waitKey(100)
        continue
    x, y = facerect[0][0:2]
    width, height = facerect[0][2:4]
    if width < 64 and height < 64:
        continue
    image = img[y - 10: y + height, x: x + width]
    image = cv2.resize(image, (64, 64))
    image = image.reshape(1, 64*64)
    image = image.astype('float32')
    image /= 255
    
    result = model.predict_classes(image)

    if result == 1:  # boss
        rect = facerect[0]
        cv2.putText(frame,"sml", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=2)
        cv2.imshow("capture", frame)
        cv2.waitKey(100)
        print('Boss is approaching')
        #show_image()
    else:
        rect = facerect[0]
        cv2.putText(frame,"other", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), (0, 0 , 0), thickness=2)
        cv2.imshow("capture", frame)
        cv2.waitKey(100)
        print('Not boss')







