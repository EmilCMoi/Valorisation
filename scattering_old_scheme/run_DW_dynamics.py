from DW_dynamics import run_dw_dynamics_2
from tqdm import trange
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
Nsteps=6000 
NstepsEfield=200#200 250 300 350#500 #17, 35, 53, 74, 98, 127, 166, 226, 350 # 20, 40 60 80 100 130 170 230 350
dVs=[-10]

for i in range(len(dVs)):
    dV=dVs[i]
    print(dir)
    run_dw_dynamics_2(Nx,Ny,dir,NstepsEfield,Nsteps,dV)