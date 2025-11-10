import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from DW_dynamics import run_dw_dynamics_track_energies, continue_track_energies, analyze_lammps_track
from utils_1D import rebuild_system
from draw import cmaps
from scipy.fft import fft
import pywt
Lflep, Lflep_r, Dflep, Dflep_r=cmaps()

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['figure.constrained_layout.use'] = True

def kd_freeze_indices(system,dir,x1,x2,nn):
    
    if dir=='0' or dir=='60':
        tree=KDTree(system.positions[:,1:2])
    else:
        tree=KDTree(system.positions[:,0:1])
    #print((tree.query(x1,nn)[1],tree.query(x2,nn)[1]))
    return np.concatenate((tree.query(x1,nn)[1],tree.query(x2,nn)[1]))

Nx=1
Ny=300
dir='0'
Nsteps=7000
#New_steps=100000-Nsteps
NstepsEfield=300#2 # 2, 5, 10, 15
dV=-10
Ly=2.511
system0=rebuild_system(Nx,Ny,dir)
# Track 32 atoms in different regions
Ntrack=31
track_indices=[]
for i in range(Ntrack):
    x=i*Ly*Ny/(Ntrack+1)
    track_indices.append(np.min(kd_freeze_indices(system0,dir,x,0,2)[:2]))
    #if i==0:
    #    track_indices=np.min(kd_freeze_indices(system0,dir,x,0,2))
    #else:
    #    track_indices=np.concatenate((track_indices,kd_freeze_indices(system0,dir,x,0,1)))

track_indices=np.concatenate((kd_freeze_indices(system0,'0',0,Ly*Ny/8,2),kd_freeze_indices(system0,'0',Ly*Ny/4,Ly*Ny*3/8,2),np.concatenate((kd_freeze_indices(system0,'0',Ly*Ny/2,Ly*Ny*5/8,2),kd_freeze_indices(system0,'0',Ly*Ny*3/4,Ly*Ny*7/8,2)))))
#track_indices=np.array(track_indices)+1
track_xs=np.array([system0.positions[i][1] for i in track_indices])
Nevery=100
#print(len(track_indices))
#print(len(np.unique(track_indices)))
#run_dw_dynamics_track_energies(Nx,Ny,dir,NstepsEfield,Nsteps,dV,track_indices)
#continue_track_energies(New_steps,Nevery,Nx,Ny,dir,NstepsEfield,Nsteps,dV,track_indices)
#analyze_lammps_track(New_steps,Nevery,Nx,Ny,dir,NstepsEfield,Nsteps,dV,track_indices)

#times=np.loadtxt(f'data/times_0_-10_{Nsteps+New_steps}_{NstepsEfield}_1_300.txt')[500:]
times=np.loadtxt(f'data/times_0_-10_{Nsteps}_{NstepsEfield}_1_300.txt')

#velocities=np.load(f'data/velocities_track_0_-10_{Nsteps+New_steps}_{NstepsEfield}_1_300.npy')[500:]
velocities=np.load(f'data/velocities_track_0_-10_{Nsteps}_{NstepsEfield}_1_300.npy')

# Calculate VAF
vaf_dir=np.zeros((len(times),3))
vaf=np.zeros((len(times)))
for i in range(len(track_indices)):
    v0_dir=velocities[NstepsEfield,i]
    v0=velocities[NstepsEfield,i]
    for j in range(len(times)):
        vaf[j]+=velocities[j,i].dot(v0)
        for k in range(3):
            vaf_dir[j][k]+=velocities[j,i][k]*v0_dir[k]
        #print(v0.dot(v0))
vaf=vaf/len(track_indices)
vaf_dir=vaf_dir/len(track_indices)
print(vaf[0])
plt.figure()
plt.plot(times,vaf)
plt.xlabel("Time (fs)")
plt.ylabel("Velocity Autocorrelation Function (a.u.)")
plt.grid()
plt.figure()
freqs=np.fft.fftfreq(len(times),d=(times[1]-times[0]))
vaf_fft=np.abs(fft(vaf-np.mean(vaf)))
plt.plot(freqs[vaf_fft.shape[0]//2:],vaf_fft[vaf_fft.shape[0]//2:])
plt.xlabel("Frequency (THz)")
plt.ylabel("VAF FFT (a.u.)")
plt.grid()
plt.figure()
plt.plot(times,vaf_dir)
plt.xlabel("Time (fs)")
plt.ylabel("Velocity Autocorrelation Function (a.u.)")
plt.grid()
plt.figure()
freqs_dir=np.fft.fftfreq(len(times),d=(times[1]-times[0]))
for i in range(3):
    plt.plot(freqs_dir[freqs_dir.shape[0]//2:],np.abs(fft(vaf_dir[:,i]-np.mean(vaf_dir[:,i])))[freqs_dir.shape[0]//2:],label=f"{['x','y','z'][i]}")
#vaf_fft_dir=np.abs(fft(vaf_dir-np.mean(vaf_dir,axis=0)))
#print(vaf_fft_dir.shape())
#plt.plot(freqs_dir[vaf_fft_dir.shape[0]//2:],vaf_fft_dir[vaf_fft_dir.shape[0]//2:])

plt.xlabel("Frequency (THz)")
plt.ylabel("VAF FFT (a.u.)")
plt.grid()

time=times/1000#fulltime
for i in range(3):
    chirp=vaf_dir[:,i]#fulldat
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
    axs[0].set_title("Continuous Wavelet Transform (Scaleogram) direction "+['x','y','z'][i])
    fig.colorbar(pcm, ax=axs[0])

    # plot fourier transform for comparison
    from numpy.fft import rfft, rfftfreq

    yf = rfft(chirp)
    xf = rfftfreq(len(chirp), sampling_period)
    plt.semilogx(xf, np.abs(yf))
    axs[1].set_xlabel("Frequency [THz]")
    axs[1].set_title("Fourier Transform")
#plt.tight_layout()


#plt.show()



#polarizations=np.load(f'data/polarizations2_0_-10_{Nsteps+New_steps}_{NstepsEfield}_1_300.npy')
polarizations=np.load(f'data/polarizations2_0_-10_{Nsteps}_{NstepsEfield}_1_300.npy')
vc=0.09155
tcol=2000
plt.figure()
plt.plot(times,np.sum(polarizations,axis=1))
plt.figure()
plt.pcolor(times,track_xs[::2],velocities[:,::2,0].T,cmap=Lflep)
plt.colorbar()
plt.figure()
plt.pcolor(times,np.arange(len(track_indices)),velocities[:,:,1].T,cmap=Lflep)
'''
for i in range(len(track_indices)):
    plt.figure()
    plt.grid(True)
    plt.title(f"{track_indices[i]}")
    plt.plot(times,velocities[:,i,0],'r-',label='x')
    plt.plot(times,velocities[:,i,1],'b-',label='y')
    plt.plot(times,velocities[:,i,2],'g-',label='z')
    plt.axvline(2000,linestyle='--',color='k',label='DW_collision')
    dx=np.abs(system0.positions[track_indices[i]][1]-Ny/2*Ly)
    plt.axvline(tcol+dx/vc,linestyle='--',color='k',label='DW_bounceback')
    plt.legend()
'''
plt.show()
