# -*- coding: utf-8 -*-
"""
load_wav_play.py -


"""
import sounddevice as sd        # (*1)
import soundfile as sf
import numpy as np

# PARAMETER(S)                  # (*2)

SAMPLING = 12000                    # sampling rate
WAV_FILE = "./vowel_12k.wav"        # filename.wav

#
# MAIN BODY
#
sd.default.device = [0, 2] # change default devices to Sound Mappers

wave, sampling_rate = sf.read(WAV_FILE) # (*3)
# wave : ndarray, sampling_rate : int

#print("original_rate= ", sampling_rate)

sampling_rate = SAMPLING

print("sampling_rate= ", sampling_rate)

sd.play(wave, sampling_rate) # (*4)

sd.wait() # wait
