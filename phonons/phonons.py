import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from DW_dynamics import run_dw_dynamics_track_energies
from utils_1D import rebuild_system
def kd_freeze_indices(system,dir,x1,x2,nn):
    
    if dir=='0' or dir=='60':
        tree=KDTree(system.positions[:,1:2])
    else:
        tree=KDTree(system.positions[:,0:1])
    return np.concatenate((tree.query(x1,nn)[1],tree.query(x2,nn)[1]))

Nx=1
Ny=300
dir='0'
Nsteps=3000
NstepsEfield=300
dV=-10
Ly=2.511
system0=rebuild_system(Nx,Ny,dir)
track_indices=np.concatenate((kd_freeze_indices(system0,'0',0,Ly*Ny/2,8),kd_freeze_indices(system0,'0',Ly*Ny/4,Ly*Ny*3/4,8)))

#run_dw_dynamics_track_energies(Nx,Ny,dir,NstepsEfield,Nsteps,dV,track_indices)
times=np.loadtxt('data/times_0_-10_3000_300_1_300.txt')
velocities=np.load('data/velocities_track_0_-10_3000_300_1_300.npy')

polarizations=np.load('data/polarizations2_0_-10_3000_300_1_300.npy')
vc=0.09155
tcol=2000
plt.figure()
plt.plot(times,np.sum(polarizations,axis=1))
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
plt.show()