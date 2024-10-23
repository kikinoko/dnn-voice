# -*- coding: utf-8 -*-
"""
rec_save_wav.py - adin, save .wav file


"""
import sounddevice as sd        # (*1)
import soundfile as sf
import numpy as np

# PARAMETER(S)                  # (*2)

MAX_TIME = 3                    # record 3 seconds
SAMPLING = 12000                # sampling rate (Hz)
WAV_FILE = "./vowel_12k.wav"    # filenme.wav

#
# MAIN BODY
#
sd.default.device = [0, 2] # change default devices to Sound Mappers

sampling_rate = SAMPLING

print("sampling_rate= ", sampling_rate)

print("sd.rec : A/D start ...") # (*3)
wave = sd.rec(int(MAX_TIME * sampling_rate),    # n_samples
              samplerate=sampling_rate,         # sampling_rate
              channels=2                        # n_channels
              )
sd.wait() # wait 

print("Finished.")
print(wave.shape) # (MAX_TIME * sampling_rate, n_channels)

# save waveform data as wav file
sf.write(WAV_FILE, wave, sampling_rate) #(*4)
