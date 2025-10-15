import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'font.family': 'serif'})
from ase.io import Trajectory
from tqdm import trange
'''
Lx=4.349179577805451#2.511
Ly=2.511
def get_deformations(system,Nx,Ny,dir):
    if dir=='0' or dir=='60':
        la='y'
        Ns=Ny
        Nt=Nx
    elif dir=='30' or dir == '90':
        la='x'
        Ns=Nx
        Nt=Ny
    if la=='x':
        Lt=Ly
        Ll=Lx
        truth_axis=0
        other_axis=1
    elif la=='y':
        Lt=Lx
        Ll=Ly
        truth_axis=1
        other_axis=0
    """Function to calculate the polarization"""
    #polarizations = np.zeros((Ns,3))
    deformations = np.zeros((Ns,3))
    for i in range(Nx):
        for j in range(Ny):
            if dir=='0' or dir=='60':
                l=j
            elif dir=='30' or dir=='90':
                l=i
            for k in range(4):
                #polarizations[l] += system.get_array("charges_model")[4*(i+j*Nx)+k]*system.positions[4*(i+j*Nt)+k]/Nt
                #polarizations[l]+=system.get_array("charges_model")[4*(i+j*Nx)+k+round(len(system)/2)]*system.positions[4*(i+j*Nt)+k+round(len(system)/2)]/Nt
                deformations[l] += system.positions[4*(i+j*Nx)+k+round(len(system)/2)] - system.positions[4*(i+j*Nx)+k]
    deformations/=4*Nt
    a=2.511
    if dir=='0' or dir=='90':
        SP_v=np.array([0,a*np.sqrt(3)/3])
    elif dir=='60' or dir =='30':
        SP_v=np.array([-a*np.sqrt(3)/3*np.cos(np.pi/3)/2,a*np.sqrt(3)/3*np.sin(np.pi/3)/2])
    phi=np.linalg.norm(deformations[:,:2]-SP_v,axis=1)

    return  phi
print("Reading trajectory")
traj=Trajectory('DW2_dir_0.traj', 'r')
phi=np.zeros((len(traj),300))
for i in trange(len(traj)):
    system=traj[i]
    phi[i]=get_deformations(system,1,300,'0')
'''


#dat1=np.loadtxt("/home/zanko/PDM/W3/raman/linear_response_dz.dat")
#dat2=np.loadtxt("/home/zanko/PDM/W3/raman/linear_response_dz2.dat")
#dat3=np.loadtxt("/home/zanko/PDM/W3/raman/linear_response_dz3.dat")
#dat4=np.loadtxt("/home/zanko/PDM/W3/raman/linear_response_dz4.dat")

#fulldat=np.concatenate((dat0[:,2],dat1,dat2,dat3,dat4))
#fulldat=np.concatenate((dat0[:,2],dat1[:,2],dat2[:,2],dat3[:,2],dat4,dat5,dat6))
fulldat=np.sum(np.load('polarizations2_0_-10.npy'),axis=1)
fulltime=np.linspace(0,(len(fulldat)-1)/1000,len(fulldat))
fig, ax1 = plt.subplots()
ax1.plot(fulltime,fulldat,'r')
plt.xlabel(r"Time [ps]")
plt.ylabel(r"$d_z$ [eÅ]")
plt.grid(True)
plt.tight_layout()

import pywt

#plt.figure()

time=fulltime
chirp=fulldat
# perform CWT
wavelet = "cmor1.5-1.0"
# logarithmic scale for scales, as suggested by Torrence & Compo:
widths = np.geomspace(1, 1024, num=100)
sampling_period = np.diff(time).mean()
cwtmatr, freqs = pywt.cwt(chirp, widths, wavelet, sampling_period=sampling_period)
# absolute take absolute value of complex result
cwtmatr = np.abs(cwtmatr[:-1, :-1])

# plot result using matplotlib's pcolormesh (image with annoted axes)
fig, axs = plt.subplots(2, 1)
pcm = axs[0].pcolormesh(time, freqs, cwtmatr)
axs[0].set_yscale("log")
axs[0].set_xlabel("Time [ps]")
axs[0].set_ylabel("Frequency [THz]")
axs[0].set_title("Continuous Wavelet Transform (Scaleogram)")
fig.colorbar(pcm, ax=axs[0])

# plot fourier transform for comparison
from numpy.fft import rfft, rfftfreq

yf = rfft(chirp)
xf = rfftfreq(len(chirp), sampling_period)
plt.semilogx(xf, np.abs(yf))
axs[1].set_xlabel("Frequency [THz]")
axs[1].set_title("Fourier Transform")
plt.tight_layout()


plt.show()
