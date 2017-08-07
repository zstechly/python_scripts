#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 20:02:06 2017
This is to test the multiprocessing functions
@author: zstechly
"""
import numpy as np
from multiprocessing import Queue, Process
from scipy import fftpack, fft
from matplotlib import pyplot as plt


def doFFT(data_in,queue):
    queue.put(20*np.log10(np.abs(fftpack.fftshift(fft(data_in)))))
    
# let's have 4 signals, and we want to calculate the FFT of each
size = 32768*8
signal_num = 24
time = np.arange(size)
sig_vector = np.zeros([signal_num,size],dtype='complex')

for x in range(signal_num):
    sig_vector[x,:] = np.cos(2*np.pi*x/2000*time) + 1j*np.sin(2*np.pi*x/2000*time)



#sig1_fft = 20*np.log10(np.abs(fftpack.fftshift(fft(sig_vector[0,:]))))
#sig2_fft = 20*np.log10(np.abs(fftpack.fftshift(fft(sig_vector[1,:]))))
#sig3_fft = 20*np.log10(np.abs(fftpack.fftshift(fft(sig_vector[2,:]))))
#sig4_fft = 20*np.log10(np.abs(fftpack.fftshift(fft(sig_vector[3,:]))))
#plt.plot(sig1_fft)

# do things in parallel
queues = [Queue() for i in range(signal_num)]
processes = [Process(target=doFFT,args=(sig_vector[i,:],queues[i])) for i \
                                        in range(signal_num)]
[p.start() for p in processes]
results = [q.get() for q in queues]
[p.terminate() for p in processes]
plt.plot(results[0])