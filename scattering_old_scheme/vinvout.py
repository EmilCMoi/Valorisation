import numpy as np
import matplotlib.pyplot as plt
from DW_dynamics import plot_dynamics_continued

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['figure.constrained_layout.use'] = True

Nx=1
Ny=300
dir='0'
dV=-10
Nsteps=20000

percentage=0.7

#pers=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#Ns=[20, 40, 60, 80, 100, 130, 170, 230, 350]
NstepsEfield=600
data=np.load(f"data/polarizations2_{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.npy")
dip=np.sum(data,axis=1)
t=np.loadtxt(f"data/times_{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.txt")

t1=np.argmin(np.abs(t-500))
t2=np.argmin(np.abs(t-1500))
fit1=np.polyfit(t[t1:t2],9.66655716e-05*dip[t1:t2],1)

t3=np.argmin(np.abs(t-6000))
t4=np.argmin(np.abs(t-10000))
fit2=np.polyfit(t[t3:t4],9.66655716e-05*dip[t3:t4],1)

plt.figure()
plt.grid(True)
plt.plot(t[:550],(fit1[0]*t[:550]+fit1[1])/9.66655716e-05,'b--',label=r'$v_{in}'+rf'={fit1[0]/0.9155*1e8:.2} v_c$')
plt.plot(t[560:750],(fit2[0]*t[560:750]+fit2[1])/9.66655716e-05,'k--',label=r'$v_{out}'+rf'={np.abs(fit2[0]/0.9155*1e8):.2} v_c$')
plt.plot(t,dip,'r-')
plt.xlabel("Time [fs]")
plt.ylabel("Total Dipole moment [a.u.]")
plt.legend()

plot_dynamics_continued(Nx,Ny,dir,NstepsEfield,Nsteps,dV,NVT=False,Temperature=0)
plt.show()