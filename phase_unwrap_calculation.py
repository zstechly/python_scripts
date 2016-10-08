# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 07:42:49 2016

@author: zstechly
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

Fs = 250
ts = 1 / Fs
f = 5

t = np.arange(1000)

sig = np.cos(2*np.pi*f/Fs*t) + 1j*np.sin(2*np.pi*f/Fs*t)
sig_angle = np.angle(sig)

angle_unwrapped = np.unwrap(sig_angle)
xaxis = np.arange(angle_unwrapped.size) * 1 / Fs
plt.figure(1)
plt.plot(xaxis,angle_unwrapped)
plt.title('Unwrapped phase angle')
plt.ylabel('Radians')
plt.xlabel('Seconds')

delta = angle_unwrapped[-1] - angle_unwrapped[0]
rad_sec = delta / (ts*angle_unwrapped.size)
freq_calc = rad_sec/(2*np.pi)