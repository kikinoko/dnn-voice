# -*- coding: utf-8 -*-
"""
learn_step_sigm_2.py - approximation of step function


"""
from tensorflow import keras
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import random

# PARAMETER(S)

SEED           =7    #(1)     random seed

HIDDEN_UNITS   = 1      #(2)     num of units in a hidden layer

EPOCHS         = 5000   #(5000)  num of training iterations in a check period
CHECKS         = 20     #(20)    num of check points

LEARNING_RATE  = 0.1    #(0.1)   learning rate
MOMENTUM       = 0.0    #(0.0)   momentum

EPSIRON        = 1.e-4  #(1.e-4) stop when loss < eps
EPS_DELTA      = 0.0    #(0.0)   stop when delta_loss < eps_delta 

GRAPHICS       = 2      #(1)     =1 graphics at last, =2 at all check points
X_STEPS        = 11     #(11)    x steps

#
# MAIN BODY
#
# STEP : datasets of step function

x_train = np.array([[0.0],[0.3],[0.4],[0.5],[1.0]])

y_train = np.array([[0.0],[0.2],[0.5],[0.8],[1.0]])

#print(x_train.shape, x_train.dtype)
#print(y_train.shape, y_train.dtype)

print("\ndataset:")
for i in range(len(x_train)) :
    print(f"{x_train[i]}\t{y_train[i]}")  

# 1x(n)x1 : structure of neural network

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[1]))
model.add(keras.layers.Dense(HIDDEN_UNITS, activation="relu"))
model.add(keras.layers.Dense(1,            activation="relu"))

#print(model.summary())

random.seed(SEED)
print(f"\nseed: {SEED}")

for i in range(1, len(model.layers)) :
    lay = model.layers[i]
    weights, biases = lay.get_weights()
    #print(i,weights.shape, weights, biases.shape, biases)
    for j in range(weights.shape[0]) :
        for k in range(weights.shape[1]) :
            weights[j][k] = random.uniform(-20.0, 20.0)

    for j in range(biases.shape[0]) :
        sum_k = 0.0
        for k in range(weights.shape[0]) :
            sum_k += 0.5 * weights[k][j]
        biases[j] = 0.5 - sum_k

    lay.set_weights([weights, biases])  # OK! that's it
    #print(f"init: {weights} {biases}")

sgd = optimizers.SGD(
    learning_rate= LEARNING_RATE,   # learning rate
    momentum=      MOMENTUM,        # momentum
    nesterov=      False)           # apply Nesterov momentum

model.compile(
    loss="mean_squared_error", 
    optimizer="sgd",   # Statistical Gradient Decent
    )

# setup graphics

if GRAPHICS > 1 :
    fig  = plt.figure()
    axis = fig.add_subplot(111)
    #axis = fig.add_subplot(111, projection='3d')

if GRAPHICS > 0 :
    x_mesh = np.linspace(0.0, 1.0, X_STEPS)
    y_mesh = np.empty_like(x_mesh)

    x_sup  = np.empty(x_train.shape[0], dtype="float")
    y_sup  = np.empty_like(x_sup)
    for i in range(x_train.shape[0]) :
        x_sup[i] = x_train[i][0]
        y_sup[i] = y_train[i]   
    #print(x_sup, y_sup)

# drawing function(s) 

def draw_graph() :    
    for i in range(X_STEPS) :
        #print([[x_mesh[i])]
        y_mesh[i] = model.predict([[x_mesh[i]]], verbose=0)

    #print(y_mesh.shape, y_mesh)

    axis.cla()   # clear the current axes state
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_xticks([0.0, 0.5, 1.0])
    axis.set_yticks([0.0, 0.5, 1.0]) 
    
    axis.plot(x_mesh, y_mesh)
    axis.scatter(x_sup, y_sup, s=50, c="pink", edgecolor="r")
    
    if GRAPHICS > 1 :
        plt.pause(1.e-1)

# model fit (NN training)
print("\n### NN training\n")

last_loss = 0.0

batch_size = len(x_train)

for kai in range(CHECKS) :
    
    history = model.fit(x_train,y_train, epochs=EPOCHS,batch_size= batch_size, verbose=0)
    
    #print(f"out: {model.predict(x_train)}")

    loss = model.evaluate(x_train, y_train, verbose=0)
    print(f"epoch: {kai * EPOCHS}\tloss: {loss :.5g}")

    for i in range(1, len(model.layers)) :
        lay = model.layers[i]
        weights, biases = lay.get_weights()
        #print(i,weights.shape, weights, biases.shape, biases)
    
    if GRAPHICS > 1 :
        draw_graph()

    if kai != 0 :
        if loss < EPSIRON or abs(last_loss - loss) < EPS_DELTA :
            break    
    last_loss = loss

#print(f"final_out:\n{model.predict(x_train)}")
print("\nfinal_out:")
for i in range(len(x_train)) :
    z = model.predict(x_train[i:i+1,:], verbose=0)
    print(f"{x_train[i]}\t{y_train[i]}\t{z[0][0] :.5f}")  
 
# model evaluate (NN test)
print("\n### NN test\n")

model.evaluate(x_train, y_train)

# final event loop

if GRAPHICS == 1 :
    fig  = plt.figure()
    axis = fig.add_subplot(111)
    
if GRAPHICS > 0 :
    draw_graph()
    plt.show()
    print("CAUTION:: pass through the plt.show()")
