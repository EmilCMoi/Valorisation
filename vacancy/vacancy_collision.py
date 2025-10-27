import numpy as np
import matplotlib.pyplot as plt
from DW_vacancy import vacancy_dynamics, plot_vacancy_dynamics
from ase.visualize import view
from ase.io import read, write
from ase.io.trajectory import Trajectory
Nx=1
Ny=300
dir='0'
dV=-10
Nsteps=500
NstepsEfield=200
N_vacancy=Nx*Ny*4*45//100

vacancy_dynamics(Nx,Ny,dir,Nsteps,NstepsEfield,dV,N_vacancy)
plot_vacancy_dynamics(Nx,Ny,dir,NstepsEfield,Nsteps,dV,N_vacancy)
#traj=Trajectory('data/DW_0_400_500_50_-10.traj','r')
