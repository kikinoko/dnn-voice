# -*- coding: utf-8 -*-
"""
test_record.py - test of sound recording

"""
import sounddevice as sd
import soundfile as sf
import numpy as np

# PARAMETER(S)

MAX_TIME = 3 # record audio signals during 3 seconds
WAV_FILE = "./speech.wav"

#
# MAIN BODY
#
sd.default.device = [0, 2] # change default devices to Sound Mappers

# use default sampling rate
input_device_info = sd.query_devices(device=sd.default.device[0])
sampling_rate = int(input_device_info["default_samplerate"])
print("sampling_rate= ", sampling_rate) # 44100 Hz

print("sd.rec : A/D start ...")
wave = sd.rec(int(MAX_TIME * sampling_rate),
  samplerate=sampling_rate,
  channels=2
  )
sd.wait() # wait

print("Finished.")
print(wave.shape) # (MAX_TIME * sampling_rate, n_channels)

# save waveform data as wav file
sf.write(WAV_FILE, wave, sampling_rate)
