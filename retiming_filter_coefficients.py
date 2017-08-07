# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:05:29 2016

@author: zstechly
"""

import numpy as np
from scipy import fft,fftpack, signal
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
sns.set(style="ticks", context="talk")


sym_aperature  = 25;
tap_resolution = 1024; # dividing the sample by 
tap_scale      = 45000 / 1.55;

#Need a narrow response per every 25 coefficients, meaning a VERY tight total picture
coeffs_input_linear = np.floor(signal.firwin(sym_aperature*tap_resolution,.99/tap_resolution) * tap_resolution * tap_scale);
coeff_input = np.zeros([sym_aperature,tap_resolution]);

# Break into symbol aperature sizeed blocks for FPGA implementation
for x in range(sym_aperature):
    coeff_input[x] = coeffs_input_linear[x*tap_resolution:tap_resolution*(x+1)] * 1.1296;

# plot the frequency response of one phase
w, h = signal.freqz(coeffs_input_linear[0:-1:1024])
fig = plt.figure()
plt.title('Digital filter frequency response')
ax1 = fig.add_subplot(111)

plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.ylabel('Amplitude [dB]', color='b')
plt.xlabel('Frequency [rad/sample]')

ax2 = ax1.twinx()
angles = np.unwrap(np.angle(h))
plt.plot(w, angles, 'g')
plt.ylabel('Angle (radians)', color='g')
plt.grid()
plt.axis('tight')


# create a signal to test retiming with
Fs = 640e6;
F = 100e6;
t = np.arange(16384);

sig_in = np.cos(F/Fs*t) + 1j*np.sin(F/Fs*t)
sig_out0 = signal.convolve(sig_in,coeffs_input_linear[0:-1:1024])    / 29000;
sig_out1 = signal.convolve(sig_in,coeffs_input_linear[512:-1:1024])  / 29000;
sig_out2 = signal.convolve(sig_in,coeffs_input_linear[1024:-1:1024]) / 29000;
plt.figure()
plt.plot(sig_in[100:200])
plt.hold
plt.plot(sig_out0[100:200],'r')
plt.hold
plt.plot(sig_out1[100:200],'g')
plt.hold
plt.plot(sig_out2[100:200],'y')
plt.title('Input, output')
plt.show()


# now write all files in mif format
for x in range(sym_aperature):
    filename = "coeff_rom" + str(x) + ".mif";
    f = open(filename,"w")
    f.write('DEPTH = 1024;\n')
    f.write('WIDTH = 18;\n\n')
    f.write('ADDRESS_RADIX=DEC;\n')
    f.write('DATA_RADIX=HEX;\n\n')
    f.write('CONTENT\n')
    f.write('  BEGIN\n')
    for y in range(tap_resolution):
        write_str = str(y) + ":\t" + str(hex(int(coeff_input[x,y]))) + "\n"
        f.write(write_str)
    f.write("END;");
    f.close();
