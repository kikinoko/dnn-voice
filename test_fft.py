# -*- coding: utf-8 -*-
"""
test_fft.py - fast Fourier transformation (FFT)


"""
import numpy as np
import math

# PARAMETER(S)

TXT_FILE = "./vowel_a.txt"    # filename.txt (*1)

SAMPLE_RATE = 12000
BLOCK_SIZE  = 600
FFT_POINTS  = 512
N_CHANNELS  = 1
MAX_TIME    = 3.5

#
# MAIN BODY
#

# read speech data from .txt into ndarray (*1)

wave = np.empty( (0,) , dtype="float64")

for line in open(TXT_FILE, 'r') :
    wave = np.append(wave, float(line))  # needs assignment
#print(f"\nwave.shape={wave.shape} wave.dtype={wave.dtype}")

# pickup FFT_POINTS from waveform data (*2)

zoom = wave[0:FFT_POINTS].astype(np.float64)     # channel-1
#print(f"\nzoom.shape={zoom.shape} zoom.dtype={zoom.dtype}")

# apply Hanning window to picked waveform (*3)

wind = np.hanning(FFT_POINTS)    # Hanning window
#wind = np.hamming(FFT_POINTS)   # Hamming window
#wind = np.blackman(FFT_POINTS)  # Blackman window
#print(f"\nwind.shape={wind.shape} wind.dtype={wind.dtype}")

zoom = zoom * wind               # windowing
real = zoom * 1.e-3              # to avoid overflow
imag = np.zeros_like(real)
fft_in = real + imag * 1j        # make complex ndarray (*4)

# apply FFT (*5)

fft_out = np.fft.fft(fft_in)     # fast Fourier transformation
#print(f"\nfft_out.shape={fft_out.shape} fft_out.dtype={fft_out.dtype}")

# power spectrum and log power spectrum

pspec = np.empty_like(real) # (*6)
for i in range(FFT_POINTS) :
    value  = 0.0
    value += fft_out.real[i] * fft_out.real[i]
    value += fft_out.imag[i] * fft_out.imag[i]
    pspec[i] = value

#print(f"\npspec.shape={pspec.shape} pspec.dtype={pspec.dtype}")

lspec = np.empty_like(pspec) # (*7)
for i in range(FFT_POINTS) :
    value = pspec[i]
    if  value < 1.e-6 :
        value = 1.e-6
    lspec[i] = 10.0 * math.log10(value)

#print(f"\nlspec.shape={lspec.shape} lspec.dtype={lspec.dtype}")

for i in range(FFT_POINTS) :
    print(lspec[i], " ", end="") # (*8)
