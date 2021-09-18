import pandas as pd
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD, RMSprop

import h5py

np.random.seed(2222)

#Load the scaled data
X_train = np.load('Scaled.bin.npy')
Y_train = np.load('labels.bin.npy')

#reshape the given pixels into 48 X 48 matrix
x , y = 48, 48
X_train = X_train.reshape(X_train.shape[0] ,  x , y,1)

#convert categorical labels to one-hot-encoding. This is binarization of the labels
Y_train = np_utils.to_categorical(Y_train)

#Define the model as sequential stack of layers
model = Sequential()
#32 filter of 3x3
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128,init='lecun_uniform'))
model.add(Dropout(0.4))
model.add(Activation('relu'))
model.add(Dense(7))
model.add(Activation('softmax'))

#gradient descent optimizer
sgd = SGD(lr=0.055, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X_train,Y_train, batch_size=150 , nb_epoch=15)
json_string = model.to_json()
model.save_weights('model_weights.h5')
open('model_architecture.json', 'w').write(json_string)
model.save_weights('model_weights.h5')
