import sidekit
import numpy as np
import pickle
from scipy.stats import multivariate_normal

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Flatten, Dropout
from keras.layers.pooling import AveragePooling1D
from keras.layers.recurrent import LSTM
from sklearn.metrics import classification_report
from keras.preprocessing import sequence
from sklearn.preprocessing import scale
from keras.layers import Masking
from keras.callbacks import ModelCheckpoint


train_features = np.load("train_features.npy")
train_labels = np.load("train_labels.npy")

test_features = np.load("test_features.npy")
test_labels = np.load("test_labels.npy")


print("starting classifer")


model = Sequential()
model.add(LSTM(200, input_dim = train_features.shape[2],input_length =train_features.shape[1],activation="relu"))
# model.add(Dropout(0.5))
# model.add(LSTM(100))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.summary()
chkpointer = ModelCheckpoint('weights_1_layer_dropout.hdf5', monitor='val_categorical_accuracy', verbose=1, save_best_only=1, save_weights_only=False, mode='auto')
model.fit(train_features, np_utils.to_categorical(train_labels),
		 validation_data=(test_features, np_utils.to_categorical(test_labels)),
		  nb_epoch=50, batch_size=64,
		  verbose = 1,
		  callbacks = [chkpointer])





