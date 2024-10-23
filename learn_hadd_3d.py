
# -*- coding: utf-8 -*-
"""
learn_hadd_3d.py - approximation of logical function (half_adder)

Aug. 20, 2022 3D graphics

"""
from tensorflow import keras
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import random

# PARAMETER(S)
SEED           = 78     #(1)     random seed

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

ab_train = np.array([[0,0],[0,1],[1,0],[1,1]])

cs_train = np.array([[0,0],[0,1],[0,1],[1,0]])

#print(ab_train.shape, ab_train.dtype)
#print(cs_train.shape, cs_train.dtype)

print("\ndataset:")
for i in range(len(ab_train)) :
    print(f"{ab_train[i]}\t{cs_train[i]}")  

# 2x2x2 : structure of neural network

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[2]))
model.add(keras.layers.Dense(HIDDEN_UNITS, activation="sigmoid"))
model.add(keras.layers.Dense(2,            activation="linear"))

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
    fig  = plt.figure(figsize=(8,4))
    axis1 = fig.add_subplot(121, projection='3d')
    axis2 = fig.add_subplot(122, projection='3d')

if GRAPHICS > 0 :
    a = np.linspace(0.0, 1.0, XY_STEPS)
    b = np.linspace(0.0, 1.0, XY_STEPS)
    a_mesh, b_mesh = np.meshgrid(a, b)
    
    c_mesh = np.empty_like(a_mesh)
    #print(c_mesh.shape, c_mesh)
    s_mesh = np.empty_like(a_mesh)
    #print(s_mesh.shape, s_mesh)
    
    a_sup = np.empty((ab_train.shape[0]), dtype="float")
    b_sup = np.empty((ab_train.shape[0]), dtype="float")
    c_sup = np.empty((cs_train.shape[0]), dtype="float")
    s_sup = np.empty((cs_train.shape[0]), dtype="float")

    for i in range(ab_train.shape[0]) :
        a_sup[i] = ab_train[i][0]
        b_sup[i] = ab_train[i][1]
        c_sup[i] = cs_train[i][0]
        s_sup[i] = cs_train[i][1]
    #print(a_sup, b_sup, c_sup, s_sup)

# drawing function(s) 

def draw_graph() :    
    for i in range(XY_STEPS) :
        for j in range(XY_STEPS) :
            #print([[a_mesh[i][j],b_mesh[i][j]]])

            tmp = model.predict(
                [[a_mesh[i][j],b_mesh[i][j]]], verbose=0)
            c_mesh[i][j] = tmp[0][0]
            s_mesh[i][j] = tmp[0][1]

    #print(c_mesh.shape, c_mesh)
    #print(s_mesh.shape, s_mesh)

    axis1.cla()   # clear the current axes state
    axis1.set_xlabel("a")
    axis1.set_ylabel("b")
    axis1.set_zlabel("c")
    axis1.set_xticks([0.0, 0.5, 1.0])
    axis1.set_yticks([0.0, 0.5, 1.0])
    axis1.set_zticks([0.0, 0.5, 1.0])   
    
    axis1.plot_surface(a_mesh, b_mesh, c_mesh, cmap="summer")
    axis1.scatter(a_sup, b_sup, c_sup, s=50, c="pink", edgecolor="r")

    axis2.cla()   # clear the current axes state
    axis2.set_xlabel("a")
    axis2.set_ylabel("b")
    axis2.set_zlabel("s")
    axis2.set_xticks([0.0, 0.5, 1.0])
    axis2.set_yticks([0.0, 0.5, 1.0])
    axis2.set_zticks([0.0, 0.5, 1.0])   
    
    axis2.plot_surface(a_mesh, b_mesh, s_mesh, cmap="summer")
    axis2.scatter(a_sup, b_sup, s_sup, s=50, c="pink", edgecolor="r")
    
    if GRAPHICS > 1 :
        plt.pause(1.e-1)

# model fit (NN training)
print("\n### NN training\n")

if GRAPHICS > 1 :
    draw_graph()
    plt.pause(0.1)

last_loss = 0.0

for kai in range(CHECKS) :
    
    history = model.fit(ab_train, cs_train, epochs=EPOCHS, verbose=0)
    
    #print(f"out: {model.predict(ab_train)}")

    loss = model.evaluate(ab_train, cs_train, verbose=0)
    print(f"epoch: {kai * EPOCHS}\tloss: {loss :.5g}")
    
    if GRAPHICS > 1 :
        draw_graph()

    if kai != 0 :
        if loss < EPSIRON or abs(last_loss - loss) < EPS_DELTA :
            break    
    last_loss = loss

#print(f"final_out:\n{model.predict(ab_train)}")
print("\nfinal_out:")
for i in range(len(ab_train)) :
    c = model.predict(ab_train[i:i+1,:], verbose=0)
    print(f"{ab_train[i]}\t{cs_train[i]}\t{c[0]}")

# print weights and biases

for i in range(1, len(model.layers)) :
    lay = model.layers[i]
    weights, biases = lay.get_weights()
    print("\nweights of layer ",i,weights.shape, weights)
    print("\nbiases of layer  ",i,biases.shape, biases)

# model evaluate (NN test)
print("\n### NN test\n")

model.evaluate(ab_train, cs_train)

# final event loop

if GRAPHICS == 1 :
    fig  = plt.figure()
    axis1 = fig.add_subplot(111, projection='3d')
    
if GRAPHICS > 0 :
    draw_graph()
    plt.show()
    print("CAUTION:: pass through the plt.show()")
