from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

import cv2

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image

import os,json
import numpy as np

from keras.preprocessing import image
from keras.optimizers import SGD, RMSprop, Adam

#import tensorflow as tf

path = "ocr/by_class"
#path = "data/dogscats/train/"

def ConvBlock(layers, model, filters):
    for i in range(layers): 
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(filters, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Dropout(0.25))

def FCBlock(model):
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

# Mean of each channel as provided by VGG researchers
vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))

def vgg_preprocess(x):
    x = x - vgg_mean     # subtract mean
    return x[:, ::-1]    # reverse axis bgr->rgb

def OCR():
    model = Sequential()
    model.add(Lambda(vgg_preprocess, input_shape=(3,32,32)))
    print ("in ocr")
    ConvBlock(2, model, 16)
    ConvBlock(2, model, 16)
    model.add(Flatten())
    model.add(Dense(192, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(62, activation='softmax'))
    print ("outside OCR")
    return model

model = OCR()

#giving too small a batch size would result in the gradient being calculated only for a few classes
batch_size = 128
def get_batches(path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
        return gen.flow_from_directory(path, target_size=(32,32),
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

print ("starting compilation")
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
batches = get_batches(path+'train', batch_size=batch_size)
val_batches = get_batches(path+'valid', batch_size=batch_size)
print((batches))
#model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=1, validation_data=val_batches, nb_val_samples=val_batches.nb_sample)
model.fit_generator(batches, samples_per_epoch=batch_size, nb_epoch=10000, validation_data=None=val_batches, nb_val_samples=val_batches.nb_sample)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
