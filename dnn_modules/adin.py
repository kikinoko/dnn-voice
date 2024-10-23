# -*- coding: utf-8 -*-
# adin.py - module function(s) for A/D conversion
"""
Jul. 12, 2022 by Takeshi Kawabata
"""
import sounddevice as sd
import numpy as np
import sys

# GLOBAL VARIABLES IN THIS MODULE

BEGINNING_STUB = 0.55

speech_period = False

# MODULE FUNCTION(S)

def rec(
        sample_rate=11000,
        block_size=550,
        n_channels=1,
        max_time=3.0) :
    
    global speech_period, wave
    
    def callback(wave_adin, frames, time, status):
        global speech_period, wave
        
        #print("shape = ", wave_adin.shape, file=sys.stderr)  # (n_samples, n_channels)
        #print("dtype = ", wave_adin.dtype, file=sys.stderr)  # data type
        #print("frames= ", frames         , file=sys.stderr)  # frames
        #print("time  = ", time           , file=sys.stderr)  # callback time infomation
        #print("status= ", status         , file=sys.stderr)  # ?
   
        #valpow = np.mean(wave**2)  # CAUTION: it does NOT work!
   
        sqsum = 0.0
        for i in range(frames) :
            for j in range(n_channels) :
                sqsum += float(wave_adin[i, j])**2
        valpow = sqsum/frames/n_channels
        
        if valpow < 1.e-6 :
            #print(valpow, file=sys.stderr)
            valpow = 1.e-6
        logpow = 10.0 * np.log10(valpow)
        
        #print(logpow, file=sys.stderr)
        
        if speech_period == False and logpow > 0 :
            speech_period = True
            print("A/D start ...", file=sys.stderr, flush=True)
            
        if speech_period :
            wave = np.append(wave, wave_adin, axis=0) # append to row
            #print(wave.shape)

    print("sample_rate  =", sample_rate, "(Hz)",file=sys.stderr)
    print("block_size   =", block_size,         file=sys.stderr)
    print("n_channels   =", n_channels,         file=sys.stderr)
    print("max_time     =", max_time, "(sec)",  file=sys.stderr)
                 
    sd.default.device = [0, 2] # change default devices to Sound Mappers
    
    wave = np.empty((0,n_channels), int)
    
    speech_period = False
    with sd.InputStream(
            samplerate=sample_rate,     # sampling rate
            blocksize=block_size,       # block size
            channels=n_channels,        # N-channel
            dtype='int16',              # data type
            callback=callback           # callback
        ):
        sd.sleep(int((max_time + BEGINNING_STUB) * 1000)) # wait sound data (ms)
        
    speech_period = False
    print("Finished.",file=sys.stderr, flush=True)
    
    return wave
