import numpy as np
import matplotlib.pyplot as plt
from DW_dynamics import run_dw_dynamics_2, continue_run, plot_dynamics_new, plot_dynamics_continued, continue_lammps, analyze_lammps
from draw import cmaps
from tqdm import trange
EfieldSteps=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300]
#EfieldSteps=[10,20,290,300]
#EfieldSteps=[30,40,270,280]
#EfieldSteps=[50,60,250,260]
#EfieldSteps=[70,80,230,240]
#EfieldSteps=[90,100,210,220]
#EfieldSteps=[110,120,190,200]
#EfieldSteps=[130,140,170,180]
#EfieldSteps=[150,160]

Nx=1
Ny=300
dir='0'
dV=-10
Nstepsi=500
Nsteps=20000
Nevery=50

# One step at a time
DoEfield=False
DoLammps=True
DoAnalyze=False
DoPlot=False

Lflep, Lflep_r, Dflep, Dflep_r=cmaps()
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['figure.constrained_layout.use'] = True
if DoEfield:
    for i in trange(len(EfieldSteps)):
        run_dw_dynamics_2(Nx,Ny,dir,EfieldSteps[i],Nstepsi,dV,NVT=False,Temperature=0)
if DoLammps:
    for i in trange(len(EfieldSteps)):
        continue_lammps(Nsteps-Nstepsi,Nevery,Nx,Ny,dir,EfieldSteps[i],EfieldSteps[i],dV,NVT=False,Temperature=0)
if DoAnalyze:
    for i in trange(len(EfieldSteps)):
        analyze_lammps(Nsteps-Nstepsi,Nevery,Nx,Ny,dir,EfieldSteps[i],EfieldSteps[i],dV,NVT=False,Temperature=0)
if DoPlot:
    for i in trange(len(EfieldSteps)):
        data=np.load(f"data/polarizations2_{dir}_{dV}_{Nsteps}_{EfieldSteps[i]}_{Nx}_{Ny}.npy")
        dip=np.sum(data,axis=1)
        t=np.loadtxt(f"data/times_{dir}_{dV}_{Nsteps}_{EfieldSteps[i]}_{Nx}_{Ny}.txt")
        if i==0:
            dips=dip
            ts=t
            Es=np.repeat(EfieldSteps[i],len(dip))
            dips0=data[:,len(data[0])//2]

        else:
            dips=np.concatenate((dips,dip))
            ts=np.concatenate((ts,t))
            Es=np.concatenate((Es,np.repeat(EfieldSteps[i],len(dip))))
            dips0=np.concatenate((dips0,data[:,len(data[0])//2]))
        
    plt.figure()
    plt.tricontourf(ts,Es,dips,levels=100,cmap=Lflep)
    plt.xlabel("Time [fs]")
    plt.ylabel("E-field steps [fs]")
    plt.colorbar(label="Total Dipole moment [a.u.]")
    plt.title("Total dipole moment evolution")

    #print(np.size(dips0))
    #print(np.size(ts))
    
    plt.figure()
    plt.tricontourf(ts,Es,dips0,levels=100,cmap=Lflep)
    plt.xlabel("Time [fs]")
    plt.ylabel("E-field steps [fs]")
    plt.colorbar(label="Middle line Dipole moment [a.u.]")
    plt.title("Middle line dipole moment evolution")
    plt.show()
    
    #plt.show()