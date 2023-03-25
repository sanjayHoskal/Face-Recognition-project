import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

CHUNK = 1024*2
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)


fig,(ax, ax2) = plt.subplots(2, figsize=(12,6))

x= np.arange(0, 2 * CHUNK, 2)
x_fft = np.linspace(0, RATE, CHUNK)

line, =ax.plot(x, np.random.rand(CHUNK),'-',lw=2)
line_fft, = ax2.plot(x_fft, np.random.rand(CHUNK),'-',lw=2)
ax.set_ylim(-2**15,2**15)
ax.set_xlim(0,CHUNK)
ax2.set_ylim(0,10)
ax2.set_xlim(0,RATE/2)
while True:
    data = stream.read(CHUNK)

    data_int = struct.unpack(str(CHUNK) + 'h', data) #+ 2.5*np.random.randn(CHUNK)#, dtype='b')[::2] + 128
    line.set_ydata(data_int)

    y_fft = fft(data_int,len(data_int))
    PSD = (np.abs(y_fft[0:CHUNK]) * 2 /(256 * CHUNK))#y_fft[0:CHUNK]*np.conj(y_fft[0:CHUNK])/CHUNK
    line_fft.set_ydata(PSD)#(np.abs(y_fft[0:CHUNK]) * 2 /(256 * CHUNK))
    indices = PSD > 2.5
    ind1 = PSD > 0.1
    p = (list(indices).count(False) - list(ind1).count(False)) * 100 / len(indices)
    #print(list(indices).count(True))
    #print(f"{p}%")
    #print(len(y_fft[0:CHUNK]),(256*CHUNK))
    ax.set_title(f"{p}%",fontsize=20)
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show()