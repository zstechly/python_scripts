# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 22:57:49 2015

@author: zstechly
Trying to make a costas loop in python instead of matlab
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
from scipy import signal

Fs = 200e6;
f = 10e6;
phs = 0;
length = 50000;
sig_rx = np.zeros(length)
wave_cos = np.zeros(length)
wave_sin = np.zeros(length)
theta = np.zeros(length+1)
sig_rx_cos = np.zeros(length)
sig_rx_sin = np.zeros(length)
sig_cos_trail = np.zeros(10);
sig_sin_trail = np.zeros(10);
sig_result_mult = np.zeros(length)
phi=0.8
freq_err = 3e3;
mu = 0.001;
t = np.arange(length)

sig_rx = np.cos(2*np.pi*f/Fs*t+phi) #+ 1j*np.sin(2*np.pi*f/Fs*t+phi)

plt.figure(1)
plt.plot(20*np.log10(abs(fft(sig_rx))))


coeffs = signal.firwin(10,0.1);


# now down convert it with sine / cosine
for x in range(1,length):
    wave_cos[x] = np.cos(2*np.pi*(f+freq_err)/Fs*x+theta[x])
    wave_sin[x] = np.sin(2*np.pi*(f+freq_err)/Fs*x+theta[x])
#   sig_rx_dc = sig_rx[x] * (wave_cos[x] * 1j*wave_sin[x])    
    sig_rx_cos[x] = sig_rx[x] * wave_cos[x];
    sig_rx_sin[x] = sig_rx[x] * wave_sin[x];
    sig_cos_trail[1:9] = sig_cos_trail[0:8];
    sig_sin_trail[1:9] = sig_sin_trail[0:8];
    sig_cos_trail[0] = sig_rx_cos[x]
    sig_sin_trail[0] = sig_rx_sin[x]
#    sig_cos_trail[0] = sig_rx_dc.real
#    sig_cos_trail[0] = sig_rx_dc.imag
    
    sig_result_mult[x] = np.average(sig_cos_trail) * np.average(sig_sin_trail)
    theta[x+1] = theta[x] - mu*sig_result_mult[x]


sig_rx_cos_filt = 2*signal.convolve(coeffs,sig_rx_cos);
sig_rx_sin_filt = 2*signal.convolve(coeffs,sig_rx_sin);

plt.figure(2)
plt.plot(theta)
plt.title("Theta")