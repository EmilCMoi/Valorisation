import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from DW_dynamics import plot_dynamics_new
from draw import cmaps
Lflep, Lflep_r, Dflep, Dflep_r=cmaps()

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['figure.constrained_layout.use'] = True

fulldat=np.sum(np.load('polarizations2_0_-10.npy'),axis=1)
fulltime=np.linspace(0,(len(fulldat)-1)/1000,len(fulldat))
dips=np.load('polarizations2_0_-10.npy')
#p=np.polyfit(fulltime[100:1500],fulldat[100:1500],1)
#p2=np.polyfit(fulltime[2500:4500],fulldat[2500:4500],2)
def fit_exp(t,a,b,c):
    return a*np.exp(-b*t)+c
from scipy.optimize import curve_fit
popt, pcov = curve_fit(fit_exp, fulltime[1000:1500], fulldat[1000:1500],maxfev=1000000)
p=popt
p2,_=curve_fit(fit_exp, fulltime[2500:4500], fulldat[2500:4500],maxfev=1000000)
print(len(fulldat))
plt.figure()
plt.xlabel(r"Time [ps]")
plt.ylabel(r"Polarization [a.u.]")
plt.grid(True)
plt.plot(fulltime[:2000],fit_exp(fulltime[:2000],*p),'k--',label='before collision')
plt.plot(fulltime[2000:],fit_exp(fulltime[2000:],*p2),'k--',label='after collision')
plt.plot(fulltime,fulldat,'r-')
plt.legend()
plt.figure()
plt.xlabel(r"Time [ps]")
plt.ylabel(r"Breather Polarization [a.u.]")
plt.grid(True)
bion=fulldat[2000:] - fit_exp(fulltime[2000:],*p2)
plt.plot(fulltime[2000:],bion,'r-')


from matplotlib import colors
divnorm=colors.TwoSlopeNorm(vmin=-0.0006, vcenter=0, vmax=0.0006)

plt.figure()
plt.plot(1000*fftfreq(len(bion[500:])), np.abs(fft(bion[500:])),'r-')
plt.xlim(0,30)
plt.xlabel(r"Frequency [THz]")
plt.ylabel(r"FFT Amplitude [a.u.]")
plt.grid(True)
print("before collision fit: ", p)
print("after collision fit: ", p2)
plt.figure()
plt.contourf(fulltime,np.arange(dips.shape[1])*2.511,dips.T,cmap=Lflep,norm=divnorm,levels=100)
plt.xlabel(r"Time [ps]")
plt.ylabel(r"Position [Å]")
plt.title("Dipole moment evolution [a.u.]")
plt.colorbar()
plt.show()