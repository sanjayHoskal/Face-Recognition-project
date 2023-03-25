import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=[10, 6]
plt.rcParams.update({'font.size': 18})

dt = 0.001
t = np.arange(0,1,dt)
f = np.sin(2*np.pi*50*t) #+ np.sin(2*np.pi*120*t) + np.cos(2*np.pi*100*t)
f_clean = f
f = f + 2.5*np.random.randn(len(t))

plt.plot(t,f,color='c',linewidth=1.5,label='Noisy')
plt.plot(t,f_clean,color='k',linewidth=2,label='Clean')
plt.legend()

n = len(t)
print(n,len(f))
fhat = np.fft.fft(f,n)
PSD = fhat*np.conj(fhat)/n
freq = (1/(dt*n))*np.arange(n)
L = np.arange(1,np.floor(n/2),dtype='int')

fig,axs = plt.subplots(2,1)

plt.sca(axs[0])
plt.plot(t,f,color='c',linewidth=1.5,label='Noisy')
plt.plot(t,f_clean,color='k',linewidth=2,label='Clean')
plt.xlim(t[0],t[-1])
plt.legend()

plt.sca(axs[1])
plt.plot(freq[L],PSD[L],color='c',linewidth=1.5,label='Power Spectrum')
#plt.plot(t,f_clean,color='k',LineWidth=2,label='Clean')
plt.xlim(freq[L[0]],freq[L[-1]])
plt.legend()

indices = PSD > 100
ind1 = PSD > 10
PSDclean = PSD * indices
p=(list(indices).count(False)-list(ind1).count(False))*100/len(indices)
print(list(indices).count(True))
fhat = indices * fhat
ffilt = np.fft.ifft(fhat)

fig,axs = plt.subplots(3,1)

plt.sca(axs[0])
plt.plot(t,f,color='c',linewidth=1.5,label='Noisy')
plt.plot(t,f_clean,color='k',linewidth=2,label='Clean')
plt.xlim(t[0],t[-1])
plt.legend()

plt.sca(axs[1])
plt.plot(t,ffilt,color='k',linewidth=2,label='Power Spectrum')
plt.xlim(t[0],t[-1])
plt.legend()

plt.sca(axs[2])
plt.plot(freq[L],PSD[L],color='c',linewidth=1.5,label='Noisy')
plt.plot(freq[L],PSDclean[L],color='k',linewidth=2,label='Clean')
plt.xlim(freq[L[0]],freq[L[-1]])
plt.legend()

plt.show()