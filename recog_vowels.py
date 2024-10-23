# -*- coding: utf-8 -*-
"""
recog_vowels.py - recognition of vowels


"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

# PARAMETER(S)

TRAIN_FILES = ["./train_a.txt", "./train_i.txt","./train_u.txt",
               "./train_e.txt", "./train_o.txt"]

TEST_FILES = ["./test_a.txt", "./test_i.txt","./test_u.txt",
              "./test_e.txt", "./test_o.txt"]

N_MFCC   = 12

#
# MAIN BODY
#

# load training and test data (vowel mfcc coefs) (*1)

x_train = np.empty( (0, N_MFCC), dtype="float64")
y_train = np.empty( (0, 1)     , dtype="int16"  )

for i in range(len(TRAIN_FILES)) :
    vectors = np.loadtxt(TRAIN_FILES[i], dtype="float64")
    x_train = np.append(x_train, vectors, axis=0)   # input signal
    supers  = np.ones( (vectors.shape[0], 1), dtype="int16")
    y_train = np.append(y_train, supers*i , axis=0) # supervisery signal
    #print(x_train.shape, y_train.shape)

x_test = np.empty( (0, N_MFCC), dtype="float64")
y_test = np.empty( (0, 1)     , dtype="int16"  )

for i in range(len(TEST_FILES)) :
    vectors = np.loadtxt(TEST_FILES[i], dtype="float64")
    x_test  = np.append(x_test, vectors, axis=0)   # input signal
    supers  = np.ones( (vectors.shape[0], 1), dtype="int16")
    y_test  = np.append(y_test, supers*i , axis=0) # supervisery signal
    #print(x_test.shape, y_test.shape)

# keras_models_sequential (*2)
print("\n### Make NN using Keras Sequential model\n")

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(12,) ))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(  5, activation="softmax"))

print(model.summary())

# compile model
print("\n### Compile the model\n")

model.compile(
    loss="sparse_categorical_crossentropy", 
    optimizer="sgd",        # Statistical Gradient Decent
    metrics=["accuracy"],   # for classification task
    )

print("Done!\n")

# NN training) (*3)
print("\n### NN training\n")

history = model.fit(
    x_train,
    y_train,
    epochs=50,
    validation_data=(x_test, y_test)
    )

# NN test (*4)
print("\n### NN test\n")

model.evaluate(x_test, y_test)

#for i in range(len(x_train)) :
#    z = model.predict(x_train[i:i+1,:], verbose=0)
#    print(i, z) 

# show learning curve (*5)

import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)

plt.show()
print("CAUTION:: pass through the plt.show()")
