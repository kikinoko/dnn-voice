# -*- coding: utf-8 -*-
"""
test_ndarray.py - sound recording to ndarray


"""
import sounddevice as sd
import numpy as np

# PARAMETER(S)

MAX_TIME = 3.0   # record audio signals during 3 seconds
SAMPLING = 44100 # sampling rate
CHANNELS = 2     # stereo recording
DTYPE = 'int16'  # data_type

#
# MAIN BODY
#
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
