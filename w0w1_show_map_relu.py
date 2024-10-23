# -*- coding: utf-8 -*-
"""
w0w1_show_map_relu.py - approximation of step function


"""
from tensorflow import keras
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

# PARAMETER(S)

W1_LOW      = -20       #(-20)
W1_HIGH     = 60        #(60)
W1_STEPS    = 11        #(21)

W0_LOW      = -30       #(-30)
W0_HIGH     = 20        #(20)
W0_STEPS    = 11        #(21)

#W1          =  2.0
#W0          = -0.5

HIDDEN_UNITS   = 1      #(1)     num of units in a hidden layer

LEARNING_RATE  = 0.001  #(0.001) learning rate
MOMENTUM       = 0.0    #(0.0)   momentum
GRAPHICS       = 1      #(1)     =1 graphics at last

#
# MAIN BODY
#

# STEP : datasets of step function

x_train = np.array([[0.0],[0.3],[0.4],[0.5],[1.0]])

y_train = np.array([[0.0],[0.2],[0.5],[0.8],[1.0]])

print("\ndataset:")
for i in range(len(x_train)) :
    print(f"{x_train[i]}\t{y_train[i]}")  

# 1x1x1 : structure of neural network

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[1]))
model.add(keras.layers.Dense(1, activation="relu"))
model.add(keras.layers.Dense(HIDDEN_UNITS,
                             trainable=False, 
                             activation="relu"))
model.add(keras.layers.Dense(1,
                             trainable=False,           # Fix weights, biases
                             activation="linear"))
#print(model.summary())

### begin - loop for w0 and w1

w0_variation = np.linspace(W0_LOW, W0_HIGH, W0_STEPS)
w1_variation = np.linspace(W1_LOW, W1_HIGH, W1_STEPS)
w0_mesh, w1_mesh = np.meshgrid(w0_variation, w1_variation)
z_mesh = np.empty_like(w0_mesh)

#print(z_mesh.shape)

for iw0 in range(len(w0_variation)) :
    w0 = w0_variation[iw0]
    #print(f"w0=\t{w0 :.5f}", end="")
    for iw1 in range(len(w1_variation)) :
        w1 = w1_variation[iw1]
        print(f"w0=\t{w0 :.5f}\tw1=\t{w1 :.5f}", end="")

        ### middle layer 1
        
        lay = model.layers[1]
        weights, biases = lay.get_weights()
        #print(i,weights.shape, weights, biases.shape, biases)
        
        weights[0][0] = w1 # set w1
        biases[0]     = w0 # set w0

        lay.set_weights([weights, biases])
        #print(f"init: {weights} {biases}")

        ### middle layer 2 (fixed)
        
        lay = model.layers[2]
        weights, biases = lay.get_weights()
        #print(i,weights.shape, weights, biases.shape, biases)
        
        weights[0][0] = -1.5 # set w1
        biases[0]     =  1.0 # set w0

        lay.set_weights([weights, biases])
        #print(f"init: {weights} {biases}")

        ### output layer
        
        last_layer_no = len(model.layers) - 1   
        lay = model.layers[last_layer_no]
        weights, biases = lay.get_weights()

        weights[0][0] = -1.0 # set w1
        biases[0]     =  1.0 # set w0
            
        lay.set_weights([weights, biases])  # OK! that's it
        #print(f"last_layer: {weights} {biases}")
        
        ### end - Fix wegths, biases at last layer
        
        sgd = optimizers.SGD(
            learning_rate= LEARNING_RATE,   # learning rate
            momentum=      MOMENTUM,        # momentum
            nesterov=      False)           # apply Nesterov momentum
        
        model.compile(
            loss="mean_squared_error", 
            optimizer="sgd",   # Statistical Gradient Decent
            )
        
        #for i in range(1, len(model.layers)) :
        #    weights, biases = model.layers[i].get_weights()
        #    print(f"compile: {weights} {biases}")        
        
        loss = model.evaluate(x_train, y_train, verbose=0)
        print(f"\tloss: {loss :.5g}")
        
        z_mesh[iw1][iw0] = loss

### end - loop for w0 and w1
print("\n")

print(z_mesh)

# final event loop

if GRAPHICS == 1 :
    fig  = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    axis.set_xlabel("w0")
    axis.set_ylabel("w1")
    axis.set_zlabel("sq_err")
    
if GRAPHICS > 0 :
    axis.plot_surface(w0_mesh, w1_mesh, z_mesh, cmap = "winter")
    plt.show()
    print("CAUTION:: pass through the plt.show()")
