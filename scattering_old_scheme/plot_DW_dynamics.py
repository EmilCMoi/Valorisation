from DW_dynamics import plot_dynamics, plot_dynamics_new
#from tqdm import trange
'''
if dir=='0' or dir=='60':
            la='y'
            Ns=Ny
            Nt=Nx
    elif dir=='30' or dir == '90':
            la='x'
            Ns=Nx
            Nt=Ny
'''

Nx=1 
Ny=300
dir='0'
Nsteps=7000
NstepsEfield=100
dV=-10

#plot_dynamics(dir,dV,Nx,Ny)
plot_dynamics_new(Nx,Ny,dir,NstepsEfield,Nsteps,dV,NVT=False,Temperature=0)
