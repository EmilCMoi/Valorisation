import numpy as np
import matplotlib.pyplot as plt
from draw import cmaps
from DW_dynamics import plot_dynamics_continued

pers=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
Ns=[20, 40, 60, 80, 100, 130, 170, 230, 350]
Nsteps=15500
Nx=1
Ny=300
dV=-10
dir='0'

Lflep, Lflep_r, Dflep, Dflep_r=cmaps()
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['figure.constrained_layout.use'] = True

sampledColors=Dflep(np.linspace(0, 1, len(pers)))

plt.figure()
plt.grid(True)
for i in range(len(pers)):
    data=np.load(f"data/polarizations2_{dir}_{dV}_{Nsteps}_{Ns[i]}_{Nx}_{Ny}.npy")
    dip=np.sum(data,axis=1)
    t=np.loadtxt(f"data/times_{dir}_{dV}_{Nsteps}_{Ns[i]}_{Nx}_{Ny}.txt")
    plt.plot(t,dip,label=f"{pers[i]} c",color=sampledColors[i])
    #print(dip)
    #print(t)
plt.legend()
plt.xlabel("Time [fs]")
plt.ylabel("Total Dipole moment [a.u.]")

for i in range(len(pers)):
    plot_dynamics_continued(Nx,Ny,dir,Ns[i],Nsteps,dV,NVT=False,Temperature=0)
plt.show()