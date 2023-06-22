import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

𝑡 = np.linspace(0 , 3 , 12*1024) # return an array which represent my axis for time.
F = [146.83,164.81,174.64,146.83,493.88,440,392,440] # The frequencies for my music
t0 = [0,0.3,0.6,1,1.3,1.7,2.1,2.5] # The start time of each frequency
T = [0.2,0.5,0.8,1.2,1.6,2,2.4,2.9] # The end time of each frequency 

# x is the function that represent my music
x=0
for N in range(8):
    x += np.where(np.logical_and(t>=t0[N], t<=(T[N])),np.sin(2*np.pi*t*F[N]),0)
    
# fn1,fn1 represent the noise frequencies and n is the function of noise
fn1,fn2= np.random.randint(0, 512, 2)
n = np.sin(2*np.pi*t*fn1) + np.sin(2*np.pi*t*fn2)

# We use this library to perform Fourier Tranform to turn the x to Frequency domain (To be able to separate the noise)
from scipy.fftpack import fft
N = 3*1024
f = np.linspace(0 , 512 , int(𝑁/2))
c = 512 / int(N/2)

X = fft(x)
X = 2/N * np.abs(X[0:int(N/2)]) 
# Out function combined with the noise
xn = x + n
XN = fft(xn)
XN = 2/N * np.abs(XN[0:int(N/2)]) 

# This library helps us get the maximum elements in an array which represent the noise value
# and their indices are the noise frequencies
import heapq
max1Val,max2Val = heapq.nlargest(2, XN)
max1 = round(list(XN).index(max1Val)*c)
max2 = round(list(XN).index(max2Val)*c)

#The filtered function
x_filtered = xn - ( np.sin(2*np.pi*t*max1) + np.sin(2*np.pi*t*max2) )
X_filtered = fft(x_filtered)
X_filtered = 2/N * np.abs(X_filtered[0:int(N/2)]) 

# Plots and music play
plt.figure()
plt.subplot(3,1,1)
plt.plot(t,x)
plt.subplot(3,1,2)
plt.plot(t,xn)
plt.subplot(3,1,3)
plt.plot(t,x_filtered)

plt.figure()
plt.subplot(3,1,1)
plt.plot(f,X)
plt.subplot(3,1,2)
plt.plot(f,XN)
plt.subplot(3,1,3)
plt.plot(f,X_filtered)

sd.play(x_filtered, 3*1024)
