
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
        3,              # n_rows
        1,              # n_columns
        figsize=(8,7),  # width, height (inches)
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

    # apply Hanning window to picked waveform

    wind = np.hanning(FFT_POINTS)    # Hanning window
    #wind = np.hamming(FFT_POINTS)   # Hamming window
    #wind = np.blackman(FFT_POINTS)  # Blackman window
    #print(f"\nwind.shape={wind.shape} wind.dtype={wind.dtype}")

    zoom = zoom * wind               # windowing
    real = zoom * 1.e-3              # to avoid overflow
    imag = np.zeros_like(real)
    fft_in = real + imag * 1j        # make complex ndarray
    
    # apply FFT
    
    fft_out = np.fft.fft(fft_in)     # fast Fourier transformation
    #print(f"\nfft_out.shape={fft_out.shape} fft_out.dtype={fft_out.dtype}")
    
    # power spectrum and log power spectrum

    pspec = np.empty_like(real)
    for i in range(FFT_POINTS) :
        value  = 0.0
        value += fft_out.real[i] * fft_out.real[i]
        value += fft_out.imag[i] * fft_out.imag[i]
        pspec[i] = value

    #print(f"\npspec.shape={pspec.shape} pspec.dtype={pspec.dtype}")

    lspec = np.empty_like(pspec) # (*2) Natural log power spectrum
    for i in range(FFT_POINTS) :
        value = pspec[i]
        if  value < 1.e-6 :
            value = 1.e-6
        lspec[i] = math.log(value)

    #print(f"\nlspec.shape={lspec.shape} lspec.dtype={lspec.dtype}")

    # cepstrum (*3)

    real = np.copy(lspec)
    imag = np.zeros(lspec.shape, dtype=float)
    fft_in = real + imag * 1j      # make complex ndarray  

    fft_out = np.fft.fft(fft_in)   # fast Fourier transformation
    
    cep  = np.copy(fft_out.real)
    
    #print(f"\ncep.shape={cep.shape} cep.dtype={cep.dtype}")

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

    axis = axes[2]
    axis.cla()          # clear the current axes state
    
    x2 = np.arange(     # evenly spaced values within given interval
        0,              # start
        FFT_POINTS,     # stop
        step=1)         # step
    #print(x2)
    
    axis.plot(x2, cep, "b-")

    axis.set_title("Cepstrum")
    axis.set_xlabel("quefuency (FFT index)")
    axis.set_ylabel("amplitude")
    axis.set_ylim(-600., 600.) 
    
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
