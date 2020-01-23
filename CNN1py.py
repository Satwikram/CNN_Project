# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 21:15:50 2019

@author: SATWIK RAM K
"""
#importing libraries
import numpy as np
import pandas as pd

#importing Tensorflow keras libraries
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

#Intilizing the CNN
model = Sequential()

#Step 1 Convoultion
model.add(Conv2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu' ))

#Step 2 MaxPOoling
model.add(MaxPooling2D(pool_size = (2,2)))

#Step 3 Flatten
model.add(Flatten())

#Step 4 Full Connection
model.add(Dense(activation = 'relu', units = 128 ))
model.add(Dense(activation = 'sigmoid', units = 1 ))

#Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting Images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=1,
        validation_data = test_set,
        validation_steps=2000)

#Making New Predictions
import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',  target_size=(64, 64))
test_image = image.image_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

if result[0][0] == 0:
    prediction = 'Cat'
    
else:
    prediction = 'Dog'
  
    
test_image1 = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg',  target_size=(64, 64))
test_image1 = image.image_to_array(test_image1)
test_image1 = np.expand_dims(test_image1, axis = 0)
result1 = model.predict(test_image1)

if result1[0][0] == 0:
    prediction1 = 'Cat'
    
else:
    prediction1 = 'Dog'
  
