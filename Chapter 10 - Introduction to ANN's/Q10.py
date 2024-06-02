#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 20:09:24 2024

@author: nitaishah
"""

import numpy as np
import pandas as pd
import tensorflow
import keras

MNIST = keras.datasets.mnist.load_data()

(X_train_full, y_train_full) , (X_test, y_test) = MNIST

X_valid, X_train = X_train_full[:10000]/255.0 , X_train_full[10000:]/255.0
y_valid, y_train = y_train_full[:10000], y_train_full[10000:]

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation= 'relu'))
model.add(keras.layers.Dense(10, activation= 'softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

X_train.shape

model.evaluate(X_test, y_test)

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

model_new = keras.models.Sequential()
model_new.add(keras.layers.Flatten(input_shape=[28,28]))
model_new.add(keras.layers.Dense(300, activation='relu'))
model_new.add(keras.layers.Dense(100, activation= 'relu'))
model_new.add(keras.layers.Dense(10, activation= 'softmax'))

model_new.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

history_new = model_new.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),  callbacks=[early_stopping_cb])   

model_new.evaluate(X_test, y_test)

from keras.optimizers import SGD
import tensorflow as tf
import os

K = keras.backend

class ExponentialLearningRate(keras.callbacks.Callback):
    
    def __init__(self, factor):
        self.factor = factor 
        self.rates = []
        self.losses = []
    
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.learning_rate))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.learning_rate, self.model.optimizer.learning_rate * self.factor)

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
    

model_2 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model_2.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(learning_rate=1e-3), metrics=['accuracy'])
expon_lr = ExponentialLearningRate(factor=1.005)
history = model_2.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid), callbacks=[expon_lr])

plt.plot(expon_lr.rates, expon_lr.losses)
plt.gca().set_xscale('log')
plt.hlines(min(expon_lr.losses), min(expon_lr.rates), max(expon_lr.rates))
plt.axis([min(expon_lr.rates), max(expon_lr.rates), 0, expon_lr.losses[0]])
plt.grid()
plt.xlabel("Learning rate")
plt.ylabel("Loss")

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model_3 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model_3.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=3e-1),
              metrics=["accuracy"])

run_index = 1 # increment this at every run
run_logdir = os.path.join(os.curdir, "my_mnist_logs", "run_{:03d}".format(run_index))
run_logdir

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
checkpoint_cb = keras.callbacks.ModelCheckpoint('my_mnist.h5', save_best_only=True)
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

history = model_3.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])

model_4 = keras.models.load_model("my_mnist.h5")
model_4.evaluate(X_test, y_test)
