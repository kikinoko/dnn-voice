# -*- coding: utf-8 -*-
"""
learn_and_3d.py - approximation of logical function (and)

"""
from tensorflow import keras
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import random

# PARAMETER(S)

SEED           = 1      #(1)     random seed

HIDDEN_UNITS   = 2      #(2)     num of units in a hidden layer

EPOCHS         = 5000   #(5000)  num of training iterations in a check period
CHECKS         = 20     #(20)    num of check points

LEARNING_RATE  = 0.1    #(0.1)   learning rate
MOMENTUM       = 0.0    #(0.0)   momentum

EPSIRON        = 1.e-4  #(1.e-4) stop when loss < eps
EPS_DELTA      = 0.0    #(0.0)   stop when delta_loss < eps_delta

GRAPHICS       = 2      #(1)     =1 graphics at last, =2 at all check points
XY_STEPS       = 11     #(11)    3D x, y steps

#
# MAIN BODY
#

# AND : datasets of logical function (and)
x_train = np.array([[0.],[1.]])

y_train = np.array([[1.],[0.]])
#print(xy_train.shape, xy_train.dtype)
#print(z_train.shape, z_train.dtype)

print("\ndataset:")
for i in range(len(xy_train)) :
    print(f"{xy_train[i]}\t{z_train[i]}")  

# 2x2x1 : structure of neural network

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[2]))
model.add(keras.layers.Dense(HIDDEN_UNITS, activation="sigmoid"))
model.add(keras.layers.Dense(1,            activation="linear"))

#print(model.summary())

random.seed(SEED)
print(f"\nseed: {SEED}")

for i in range(1, len(model.layers)) :
    lay = model.layers[i]
    weights, biases = lay.get_weights()
    #print(i,weights.shape, weights, biases.shape, biases)
    for j in range(weights.shape[0]) :
        for k in range(weights.shape[1]) :
            weights[j][k] = random.uniform(-1.0, 1.0)

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
    axis = fig.add_subplot(111, projection='3d')

if GRAPHICS > 0 :
    x = np.linspace(0.0, 1.0, XY_STEPS)
    y = np.linspace(0.0, 1.0, XY_STEPS)
    x_mesh, y_mesh = np.meshgrid(x,y)
    
    z_mesh = np.empty_like(x_mesh)
    #print(z_mesh.shape, z_mesh)
    
    x_sup = np.empty((xy_train.shape[0]), dtype="float")
    y_sup = np.empty((xy_train.shape[0]), dtype="float")
    z_sup  = np.empty((z_train.shape[0]), dtype="float")

    for i in range(xy_train.shape[0]) :
        x_sup[i] = xy_train[i][0]
        y_sup[i] = xy_train[i][1]
        z_sup[i]  = z_train[i][0]
    #print(x_sup, y_sup, z_sup)

# drawing function(s) 

def draw_graph() :    
    for i in range(XY_STEPS) :
        for j in range(XY_STEPS) :
            #print([[x_mesh[i][j],y_mesh[i][j]]])
            z_mesh[i][j] = model.predict(
                [[x_mesh[i][j],y_mesh[i][j]]], verbose=0)

    #print(z_mesh.shape, z_mesh)

    axis.cla()   # clear the current axes state
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_xticks([0.0, 0.5, 1.0])
    axis.set_yticks([0.0, 0.5, 1.0])
    axis.set_zticks([0.0, 0.5, 1.0])   
    
    axis.plot_surface(x_mesh, y_mesh, z_mesh, cmap="summer")
    axis.scatter(x_sup, y_sup, z_sup, s=50, c="pink", edgecolor="r")
    
    if GRAPHICS > 1 :
        plt.pause(1.e-1)

# model fit (NN training)
print("\n### NN training\n")

if GRAPHICS > 1 :
    draw_graph()
    plt.pause(0.1)

last_loss = 0.0

for kai in range(CHECKS) :
    
    history = model.fit(xy_train, z_train, epochs=EPOCHS, verbose=0)
    
    #print(f"out: {model.predict(xy_train)}")

    loss = model.evaluate(xy_train, z_train, verbose=0)
    print(f"epoch: {kai * EPOCHS}\tloss: {loss :.5g}")
    
    if GRAPHICS > 1 :
        draw_graph()

    if kai != 0 :
        if loss < EPSIRON or abs(last_loss - loss) < EPS_DELTA :
            break    
    last_loss = loss

#print(f"final_out:\n{model.predict(xy_train)}")
print("\nfinal_out:")
for i in range(len(xy_train)) :
    z = model.predict(xy_train[i:i+1,:], verbose=0)
    print(f"{xy_train[i]}\t{z_train[i]}\t{z[0][0] :.5f}")  
 
# model evaluate (NN test)
print("\n### NN test\n")

model.evaluate(xy_train, z_train)

# final event loop

if GRAPHICS == 1 :
    fig  = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    
if GRAPHICS > 0 :
    draw_graph()
    plt.show()
    print("CAUTION:: pass through the plt.show()")
