# -*- coding: utf-8 -*-
"""
adin_click_wave.py - adin, show wave, click to zoom in


"""
import numpy as np
import matplotlib.pyplot as plt
import librosa
from dnn_modules.adin import rec  # Assuming rec function is imported correctly

# Parameters
OUT_FILE = "./train_a.txt"
N_MFCC = 12

SAMPLE_RATE = 12000
BLOCK_SIZE = 600
FFT_POINTS = 512
N_CHANNELS = 1
MAX_TIME = 3.5

# Callback function
def onclick(event):
    draw_graph1(int(event.xdata))

# Drawing functions
def setup_graph():
    global fig, axes
    fig, axes = plt.subplots(
        2, figsize=(8, 5), tight_layout=True, facecolor="whitesmoke")

def draw_graph0():
    axis = axes[0]
    axis.cla()   # Clear the current axes state
    x0 = np.arange(0, n_points, step=1)
    axis.plot(x0, wave[:, 0], "b-")
    axis.set_title("Speech waveform")
    axis.set_xlabel("points")
    axis.set_ylim(-30000, 30000)
    plt.draw()

def draw_graph1(ix):
    zoom = wave[ix:ix+FFT_POINTS, 0].astype(np.float64)
    mfcc = librosa.feature.mfcc(y=zoom, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                n_fft=FFT_POINTS, hop_length=BLOCK_SIZE)
    mfcc[0] = 0.0  # Ignore voice volume

    with open(OUT_FILE, 'a') as f:
        for i in range(N_MFCC):
            f.write(str(mfcc[i][0]) + " ")
        f.write("\n")

    axis = axes[1]
    axis.cla()
    x1 = np.arange(N_MFCC)
    axis.plot(x1, mfcc[0], "b-")  # Plot the first coefficient for simplicity
    axis.set_title("Mel frequency cepstrum coefficients")
    axis.set_xlabel("MFCC order")
    axis.set_ylim(-100., 200.)
    plt.draw()

# Main body
wave = rec(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE,
           n_channels=N_CHANNELS, max_time=MAX_TIME)
n_points = wave.shape[0]

setup_graph()
draw_graph0()
draw_graph1(0)

plt.connect('button_press_event', onclick)
plt.show()
