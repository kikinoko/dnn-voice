# -*- coding: utf-8 -*-
"""
adin_click_wave.py - adin, show wave, click to zoom in

"""
import numpy as np
import matplotlib.pyplot as plt
from dnn_modules.adin import rec
import math

# PARAMETER(S)

SAMPLE_RATE = 12000
BLOCK_SIZE  = 600
FFT_POINTS  = 512
N_CHANNELS  = 1
MAX_TIME    = 3.5

# call back function

def onclick(event) :
    draw_graph1(int(event.xdata))

# drawing functions

def setup_graph() :
    global fig, axes 
    
    fig, axes = plt.subplots(
        2,              # n_rows
        1,              # n_columns
        figsize=(8,5),  # width, height (inches)
        tight_layout=True, 
        facecolor="whitesmoke")

def draw_graph0() :
    
    axis = axes[0]
    axis.cla()   # clear the current axes state
    
    x0 = np.arange(     # evenly spaced values within given interval
        0,              # start
        n_points,       # stop
        step=1)         # step

    #print(x0)

    axis.plot(x0, wave[:,0], "b-")   # -  : solid line, show channel-1
    
    axis.set_title("Speech waveform")
    axis.set_xlabel("points")
    axis.set_ylim(-30000, 30000)

    plt.draw()

def draw_graph1(ix) :
     
    zoom = wave[ix:ix+FFT_POINTS, 0].astype(np.float64) # pick the zoomed waveform

### insert signal processing here ###



### end of signal processing ########

    axis = axes[1]
    axis.cla()          # clear the current axes state
   
    x1 = np.arange(     # evenly spaced values within given interval
        ix,             # start
        ix+FFT_POINTS,  # stop
        step=1)         # step
    #print(x1)
    
    axis.plot(x1, zoom, "b-")   # -  : solid line, show channel-1

    axis.set_title("Speech waveform (Zoom in)")
    axis.set_xlabel("points")
    axis.set_ylim(-30000., 30000.)
    
    plt.draw()  
#
# MAIN BODY
#
global fig, axes

# A/D conversion

wave = rec(
    sample_rate=SAMPLE_RATE,
    block_size=BLOCK_SIZE,
    n_channels=N_CHANNELS,
    max_time=MAX_TIME
    )

n_points = wave.shape[0]

print(wave.shape)
#print(wave)

# show graphs and enter event loop

setup_graph()

draw_graph0()
draw_graph1(0)

plt.connect('button_press_event', onclick)  # activate MOUSE event
plt.show()  # event loop
