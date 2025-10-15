from DW_dynamics import run_dw_dynamics_2, continue_run, plot_dynamics_new, plot_dynamics_continued
import numpy as np
import matplotlib.pyplot as plt

Nx=1
Ny=300
dir='0'
dV=-10
Nsteps=10000

#percentage=0.7

#pers=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#Ns=[20, 40, 60, 80, 100, 130, 170, 230, 350]
#i=np.argmin(np.abs(np.array(pers)-percentage))
#NstepsEfield=Ns[i]
NstepsEfield=600

#20 40 60 80 100 130 170 230 350
#NstepsEfield=100
New_steps=10000
Nevery=50
print(NstepsEfield)
#run_dw_dynamics_2(Nx,Ny,dir,NstepsEfield,Nsteps,dV,NVT=False,Temperature=0)
#plot_dynamics_new(Nx,Ny,dir,NstepsEfield,Nsteps,dV,NVT=False,Temperature=0)
continue_run(New_steps,Nevery,Nx,Ny,dir,NstepsEfield,Nsteps,dV,NVT=False,Temperature=0)
plot_dynamics_continued(Nx,Ny,dir,NstepsEfield,Nsteps+New_steps,dV,NVT=False,Temperature=0)

plt.show()