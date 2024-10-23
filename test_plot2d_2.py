# -*- coding: utf_8 *-*
"""
test_plot2d_2.py - plot speech waveform


"""
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# PARAMETER(S)

N_ROW    = 2
N_COL    = 1
WIDTH    = 8
HEIGHT   = 5
MAX_TIME = 3.0   # record audio signals during 3 seconds
SAMPLING = 12000 # sampling rate
CHANNELS = 1     # monoaural recording
DTYPE = 'int16'  # data_type

#
# MAIN BODY
#
fig, axes = plt.subplots(
    N_ROW,                   # the number of rows
    N_COL,                   # the number of columns
    figsize=(WIDTH, HEIGHT), # width, height (inches)
    tight_layout=True)

#if N_ROW * N_COL == 1: axis = axes
#else                 : axis = axes[0]

# dataset

time = np.arange(MAX_TIME * SAMPLING)

sd.default.device = [0, 2] # change default devices to Sound Mappers

print("sd.rec : A/D start ...")
wave = sd.rec(
  int(MAX_TIME * SAMPLING), # samples to record
  samplerate=SAMPLING,      # sampling rate (Hz)
  channels=CHANNELS,        # =1 : mono, =2 : stereo
  dtype=DTYPE
  )
sd.wait() # wait

print("Finished.")

wave_f = np.float64(wave)

# draw graph 1 on axes[0]

axis = axes[0]
axis.set_title("speech waveform")

axis.set_xlabel("time (samples)")
axis.set_xlim(0.0, MAX_TIME * SAMPLING)

axis.set_ylabel("amplitude")
axis.set_ylim(-32768.0, 32767.0)

axis.plot(time, wave)

# draw graph 2 on axes[1]

axis = axes[1]
axis.set_title("")

axis.set_xlabel("time (samples)")
axis.set_xlim(0.0, MAX_TIME * SAMPLING)

axis.set_ylabel("normalized")
axis.set_ylim(-32768.0, 32767.0)

axis.plot(time, wave_f)
 

plt.draw()
plt.show() 
