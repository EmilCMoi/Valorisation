import numpy as np
import matplotlib.pyplot as plt
from DW_vacancy import vacancy_dynamics, plot_vacancy_dynamics

Nx=1
Ny=300
dir='0'
dV=-10
Nsteps=500
NstepsEfield=50
N_vacancy=Nx*Ny*4//3

#vacancy_dynamics(Nx,Ny,dir,Nsteps,NstepsEfield,dV,N_vacancy)
plot_vacancy_dynamics(Nx,Ny,dir,NstepsEfield,Nsteps,dV,N_vacancy)