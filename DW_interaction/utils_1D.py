import os
import numpy as np
from ase.io import read
from model.calculator import LAMMPS
from model.born import born_charges, dipole_moment
from model.build_1D import build_1D, build_1D_charges
from tqdm import trange
from ase.io.trajectory import Trajectory
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.optimize import LBFGS
import matplotlib.pyplot as plt
import nglview as nv
from ase.optimize import BFGSLineSearch
from ase.filters import FrechetCellFilter
from subprocess import run
from scipy.optimize import curve_fit
from ase import Atoms
from ase.io import write
from ase.io.xsf import write_xsf
from model.draw import cmaps

Lflep, Lflep_r, Dflep, Dflep_r = cmaps()
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})


# Faster and more accurate method poss
# ibly
#os.system("mpirun -np 12 lmp -in /home/zanko/PDM/W14/newBorn/input_walls.lammps")
#os.system("source /home/zanko/.bashrc")

def minimize_lammps(Nx,Ny,defo,dir):
    build_1D(Nx=Nx,Ny=Ny,defo=defo,dir=dir,fileName="tmp1D_2.lammps")
    atoms=read('tmp1D_2.lammps',format="lammps-data")
    layer = np.zeros(len(atoms))
    mid = np.mean(atoms.get_positions()[:, 2])
    layer[atoms.get_positions()[:, 2] < mid] = 1
    layer[atoms.get_positions()[:, 2] > mid] = 2
    atoms.set_array("mol-id", layer, dtype=int)
    born, charges_lammps,charges=born_charges(atoms)
    
    build_1D_charges(atoms,charges_lammps,fileName="tmp1D_3.lammps")
    os.system("mpirun -np 12 lmp -in input_walls.lammps ")

    os.system("lmp -restart2data lammps.restart tmp1D_2.lammps > log2.lammps")
    #print(len(atoms),len(layer))
    atoms=read('tmp1D_2.lammps',format="lammps-data")
    atoms.set_array("mol-id", layer, dtype=int)
    born, charges_lammps,charges=born_charges(atoms)
    atoms.set_initial_charges(charges_lammps)
    atoms.set_array("born", born)
    atoms.set_array("charges_2", charges)
    voltage=[0.0,0.0]
    voltages=np.zeros(len(atoms))
    voltages[layer==1] = voltage[0]
    voltages[layer==2] = voltage[1]
    atoms.set_array("voltage", voltages)
    #atoms.calc = LAMMPS()
    

    return atoms
def wall_fit(x,x0,a0,w,b):
    return -a0*np.tanh((x-x0)/w/2)+b

a=2.511
c=6.6612/2
Ly=2.511
Lx=4.349179577805451

v1=a*np.array([np.sqrt(3)/2,1/2,0])
v2=a*np.array([np.sqrt(3)/2,-1/2,0])


def write_atoms(r):
    positions=np.zeros((4,3))
    positions[0]=np.array([0,0,0])
    positions[1]=np.array([0,0,0])+v1/3+v2/3
    positions[2]=np.array([0,0,c])+r[0]*v1 +r[1]*v2+r[2]
    positions[3]=np.array([0,0,c])+v1/3+v2/3+r[0]*v1 +r[1]*v2+r[2]
    labels=["B","N","B","N"]
    atoms=Atoms(labels, positions=positions, cell=[v1, v2, [0, 0, 30]], pbc=True)
    atoms.set_array("mol-id", np.array([1, 1, 2, 2], dtype=int))
    return atoms

def find_V0():
    AB=write_atoms([1/3,1/3,0])
    SP=write_atoms([1/2,1/2,0])
    
    born, charges_lammps, charges=born_charges(AB)
    AB.set_initial_charges(charges_lammps)
    AB.set_array("born", born)
    AB.set_array("charges_2", charges)
    voltage=[0.0,0.0]
    voltages=np.zeros(len(AB))
    voltages[AB.get_array("mol-id")==1] = voltage[0]
    voltages[AB.get_array("mol-id")==2] = voltage[1]
    AB.set_array("voltage", voltages)

    AB.calc = LAMMPS()

    E_AB=AB.get_potential_energy()

    born, charges_lammps, charges=born_charges(SP)
    SP.set_initial_charges(charges_lammps)
    SP.set_array("born", born)
    SP.set_array("charges_2", charges)
    SP.set_array("voltage", voltages)
    SP.calc = LAMMPS()
    E_SP=SP.get_potential_energy()

    V0=(E_SP-E_AB)/v1.dot(v2)

    return V0

def analyse_wall(atoms,gamma,dir):
    if dir=='0' or dir=='60':
        la='y'
    elif dir=='30' or dir == '90':
        la='x'

    if la=='x':
        Lt=Ly
        Ll=Lx
        truth_axis=0
    elif la=='y':
        Lt=Lx
        Ll=Ly
        truth_axis=1
    s=2#Lx*Ly
    n_centers=round(len(atoms)/8)
    polarizations=np.zeros((n_centers,3))
    deformations=np.zeros((n_centers,3))
    centers=np.zeros((n_centers,3))
    borns=np.zeros((n_centers,2))
    #atoms.wrap()
    pos=atoms.positions
    ch=atoms.get_array("charges_2")
    for i in range(round(len(atoms)/8)):
        for j in range(4):
            polarizations[i]+=ch[4*i+j]*pos[4*i+j]
            polarizations[i]+=ch[4*i+j+round(len(atoms)/2)]*pos[4*i+j+round(len(atoms)/2)]
            deformations[i]+=(pos[4*i+j+round(len(atoms)/2)]-pos[4*i+j])/4
        
            centers[i]+=pos[4*i+j]/8
            centers[i]+=pos[4*i+j+round(len(atoms)/2)]/8
            borns[i]+=atoms.get_array("born")[4*i+j,:2]/4
    
    if dir=='0' or dir=='90':
        SP_v=np.array([0,a*np.sqrt(3)/3])
    elif dir=='60' or dir =='30':
        SP_v=np.array([-a*np.sqrt(3)/3*np.cos(np.pi/3)/2,a*np.sqrt(3)/3*np.sin(np.pi/3)/2])
    phi=np.linalg.norm(deformations[:,:2]-SP_v,axis=1)
    
    w_fit,_=curve_fit(wall_fit,centers[:round(len(phi)/2),truth_axis],phi[:round(len(phi)/2)],maxfev=100000,p0=[1000,-1,20,1.5])#x0,a0,w,b
    
    polarizations/=s

    # Polarization quanta
    polarizations[:,0]-=-2.19292327e-02 
    polarizations[:,1]-=-1.51700056e-08

    V0=find_V0()
    a0=a*np.sqrt(3)/3/2

    lamé=gamma**2/V0/a0**2/9*2

    return centers[:,truth_axis], polarizations, phi, w_fit, lamé, deformations, borns, dir, centers[:,2]
        
def plot_polarization(centers, polarizations,la,borns, dir):
    
    if la=='x':
        Lt=Ly
        Ll=Lx
        truth_axis=0
        other_axis=1
    elif la=='y':
        Lt=Lx
        Ll=Ly
        truth_axis=1
        other_axis=0
    plt.figure()
    plt.plot(centers, polarizations[:,2],'r-')
    plt.xlabel(rf"${la}$ [Å]")
    plt.ylabel(r"Out-of-plane polarization [e/Å]")
    plt.title(f"{dir}° Wall")
    plt.xlim(centers[0],centers[round(len(centers)/2)])
    plt.grid(True)
    plt.tight_layout()
    '''
    plt.figure()
    plt.plot(centers, polarizations[:,0],'r-',label=r"$p_x$")
    plt.plot(centers, polarizations[:,1],'b-',label=r"$p_y$")
    plt.xlabel(rf"${la}$ [Å]")
    plt.ylabel(r"In-plane polarization (e/A)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    '''
    #nArrows=20
    #centers=centers[:round(len(centers)/2)]
    #borns=borns[:round(len(borns)/2)]
    #if len(centers)>nArrows:
    #    centers=centers[::int(len(centers)/nArrows)]
    #    borns=borns[::int(len(borns)/nArrows)]
    C=np.linalg.norm(borns[:,:2],axis=1)
    fig, axs=plt.subplots(1,1,figsize=(12,3))
    cntr1=axs.quiver(centers,np.zeros(len(centers)),borns[:,truth_axis]/C,borns[:,other_axis]/C,C,pivot='mid',cmap=Dflep,scale=20)
    cbar1=fig.colorbar(cntr1, ax=axs,shrink=1.5,aspect="3")
    axs.axis('equal')
    axs.set_title(f'{dir}° Wall')
    axs.set_xlabel(rf"${la}$ [Å]")
    axs.set_yticks([])
    axs.set_xticks([])
    
    cbar1.ax.set_xlabel(r"$\Delta q'_{\parallel}$ [$e$/Å]")
    plt.tight_layout()
canard = (0, 116/255, 128/255)


def plot_deformation(centers, phi, deformations, w_fit,la,dir,buckling):
    if la=='x':
        Lt=Ly
        Ll=Lx
        truth_axis=0
        other_axis=1
    elif la=='y':
        Lt=Lx
        Ll=Ly
        truth_axis=1
        other_axis=0
    plt.figure()
    plt.plot(centers, wall_fit(centers, *w_fit), '-',color=canard,label=r"Fit")
    plt.plot(centers[:len(phi)], phi,'r.', label="Deformation")
    plt.axvline(x=w_fit[0]-2*w_fit[2],color='k', linestyle='--',label=r"$x_0\pm2w$")
    plt.axvline(x=w_fit[0]+2*w_fit[2],color='k', linestyle='--')
    plt.xlim(centers[0],centers[round(len(centers)/2)])
    plt.xlabel(rf"${la}$ [Å]")
    plt.ylabel(r"$\|(\phi_x,\phi_y)\|$[Å]")    
    plt.title(rf"{dir}° Wall fit: $w={w_fit[2]:.3f}$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(centers,buckling,'r-')
    plt.xlabel(rf"${la}$ [Å]")
    plt.ylabel(r"Out-of-plane buckling [Å]")
    plt.xlim(centers[0],centers[round(len(centers)/2)])
    plt.title(f"{dir}° Wall")
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.grid(True)
    plt.plot(centers,deformations[:,2],'r-')
    plt.xlabel(rf"${la}$ [Å]")
    plt.xlim(centers[0],centers[round(len(centers)/2)])
    plt.ylabel(r"Interlayer distance [Å]")
    plt.title(f"{dir}° Wall")
    plt.tight_layout()

    plt.figure()
    plt.plot(centers,deformations[:,0],'r-',label=r"$\phi_x$")
    plt.plot(centers,deformations[:,1],'-',color=canard,label=r"$\phi_y$")
    plt.xlabel(rf"${la}$ [Å]")
    plt.ylabel(r"Deformation [Å]")
    plt.xlim(centers[0],centers[round(len(centers)/2)])
    plt.title(f"{dir}° Wall")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.figure()
    burgers=np.zeros(len(centers))
    if la=='x':
        burgers=np.arctan2(deformations[:,1],deformations[:,0])*180/np.pi
    elif la=='y':
        burgers=np.arctan2(deformations[:,0],deformations[:,1])*180/np.pi
    plt.plot(centers,burgers,'r.')
    plt.xlabel(rf"${la}$ [Å]")
    plt.ylabel(r"Deformation Angle [°]")
    plt.xlim(centers[0],centers[round(len(centers)/2)])
    plt.title(f"{dir}° Wall")
    plt.grid(True)
    plt.tight_layout()

    
def full_creation_analysis_convergence(Ns,dir,Nt=1):
    systems=[]
    systems0=[]
    gammas=np.zeros(len(Ns))
    for N in Ns:
        if dir=='0' or dir=='60':
            la='y'  
        elif dir=='30' or dir == '90':
            la='x'

        if la=='x':
            Ll=Lx
            Lt=Ly
            Nx=N
            Ny=Nt
        elif la=='y':
            Ll=Ly
            Lt=Lx
            Nx=Nt
            Ny=N
        system0=minimize_lammps(Nx=Nx,Ny=Ny,defo=False,dir=dir)
        system1=minimize_lammps(Nx=Nx,Ny=Ny,defo=True,dir=dir)
        systems0.append(system0)
        systems.append(system1)
    for i in range(len(Ns)):
        systems0[i].calc = LAMMPS()
        systems[i].calc = LAMMPS()
        E0=systems0[i].get_potential_energy()
        E1=systems[i].get_potential_energy()
        gammas[i]=(E1-E0)/Lt/2
        #print(f"N={Ns[i]}: Energy of wall: {(E1-E0)/Lt/2} eV/A")
    return gammas

def full_creation_analysis(Ns,dir,plot=True,Nt=1):
    # Ns is the number of unit cells in the wall
    # la is 'y' for zigzag, 'x' for armchair

    Lx=2.511
    Ly=4.349179577805451

    if dir=='0' or dir=='60':
        la='y'  
    elif dir=='30' or dir == '90':
        la='x'

    if la=='x':
        Ll=Lx
        Lt=Ly
        Nx=Ns
        Ny=Nt
    elif la=='y':
        Ll=Ly
        Lt=Lx
        Nx=Nt
        Ny=Ns
    system0=minimize_lammps(Nx=Nx,Ny=Ny,defo=False,dir=dir)
    system1=minimize_lammps(Nx=Nx,Ny=Ny,defo=True,dir=dir)

    
    
    system0.calc = LAMMPS()
    system1.calc = LAMMPS()
    E0=system0.get_potential_energy()
    E1=system1.get_potential_energy()
    gamma=(E1-E0)/Lt/2

    centers, polarizations, phi, w_fit, lamé, deformations,borns, dir, buckling = analyse_wall(system1, gamma=gamma, dir=dir)
    #print(centers)
    #print(polarizations)
    #print(phi)
    if plot:
        print(f"Energy of wall: {(E1-E0)/Lt/2} eV/A")
        print(f"Effective Lamé parameter: {lamé} eV/A^2")
        print(f"Reference: lambda=1.779, mu=7.939")
    
        print(f"Wall width: {w_fit[2]} A")
        print(f"Wall fit: {w_fit}")
        plot_deformation(centers, phi, deformations,w_fit, la,dir, buckling)
        plot_polarization(centers, polarizations, la,borns,dir)
    #print(polarizations[0])
    return system1, gamma
'''
if dir=='0' or dir=='60':
        la='y'
    elif dir=='30' or dir == '90':
        la='x'
0 60 y
30 90 x
Lx=sqrt(3)Ly

'''

# Ok, just need to change names, whatever
# Decide once and for all
# zigzag is the direction of 0 and 60, called y from now on
# armchair is the direction of 30 and 90, called x from now on
# 1000 Nx and 1730 Ny
'''
atoms=full_creation_analysis(Ns=1000,dir='90')
#write("test.xsf",atoms,format="xsf")
plt.show()
print(find_V0())
'''
def full_full_creation_analysis():
    # Ns is the number of unit cells in the wall
    # la is 'y' for zigzag, 'x' for armchair

    Lx=4.349179577805451
    Ly=2.511
    '''
    if dir=='0' or dir=='60':
        la='y'  
    elif dir=='30' or dir == '90':
        la='x'

    if la=='x':
        Ll=Lx
        Lt=Ly
        #Nx=Ns
        #Ny=Nt
    elif la=='y':
        Ll=Ly
        Lt=Lx
        #Nx=Nt
        #Ny=Ns
    '''
    # 1000 Nx and 1730 Ny
    system0_0=minimize_lammps(Nx=1,Ny=1730,defo=False,dir='0')
    system_0=minimize_lammps(Nx=1,Ny=1730,defo=True,dir='0')

    system0_30=minimize_lammps(Nx=1000,Ny=1,defo=False,dir='30')
    system_30=minimize_lammps(Nx=1000,Ny=1,defo=True,dir='30')

    system0_60=minimize_lammps(Nx=1,Ny=1730,defo=False,dir='60')
    system_60=minimize_lammps(Nx=1,Ny=1730,defo=True,dir='60')

    system0_90=minimize_lammps(Nx=1000,Ny=1,defo=False,dir='90')
    system_90=minimize_lammps(Nx=1000,Ny=1,defo=True,dir='90')
    
    system0_0.calc = LAMMPS()
    system_0.calc = LAMMPS()
    system0_30.calc = LAMMPS()
    system_30.calc = LAMMPS()
    system0_60.calc = LAMMPS()
    system_60.calc = LAMMPS()
    system0_90.calc = LAMMPS()
    system_90.calc = LAMMPS()
    E0_0=system0_0.get_potential_energy()
    E0_30=system0_30.get_potential_energy()
    E0_60=system0_60.get_potential_energy()
    E0_90=system0_90.get_potential_energy()
    E_0=system_0.get_potential_energy()
    E_30=system_30.get_potential_energy()
    E_60=system_60.get_potential_energy()
    E_90=system_90.get_potential_energy()
    #E0=system0.get_potential_energy()
    #E1=system1.get_potential_energy()
    #gamma=(E1-E0)/Lt/2
    gamma_0=(E_0-E0_0)/Lx/2
    gamma_30=(E_30-E0_30)/Ly/2
    gamma_60=(E_60-E0_60)/Lx/2
    gamma_90=(E_90-E0_90)/Ly/2

    print(gamma_0)
    print(gamma_30)
    print(gamma_60)
    print(gamma_90)

    centers0, polarizations0, phi0, w_fit0, lamé0, deformations0,borns0, dir0, buckling0 = analyse_wall(system_0, gamma=gamma_0, dir='0')
    centers30, polarizations30, phi30, w_fit30, lamé30, deformations30,borns30, dir30, buckling30 = analyse_wall(system_30, gamma=gamma_30, dir='30')
    centers60, polarizations60, phi60, w_fit60, lamé60, deformations60,borns60, dir60, buckling60 = analyse_wall(system_60, gamma=gamma_60, dir='60')
    centers90, polarizations90, phi90, w_fit90, lamé90, deformations90,borns90, dir90, buckling90 = analyse_wall(system_90, gamma=gamma_90, dir='90')
    
    print(lamé0)
    print(lamé30)
    print(lamé60)
    print(lamé90)

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    axs = axs.flatten()
    axs[0].plot(centers0,wall_fit(centers0, *w_fit0), '-',color=canard,label=r"Fit")
    axs[0].plot(centers0[:len(phi0)], phi0,'r.', label="Deformation")
    axs[0].axvline(x=w_fit0[0]-2*w_fit0[2],color='k', linestyle='--',label=r"$x_0\pm2w$")
    axs[0].axvline(x=w_fit0[0]+2*w_fit0[2],color='k', linestyle='--')
    axs[0].set_xlim(centers0[0],centers0[round(len(centers0)/2)])
    axs[0].set_xlabel(rf"$y$ [Å]")
    axs[0].set_ylabel(r"$\|(\phi_x,\phi_y)\|$[Å]")
    axs[0].set_title(rf"{dir0}° Wall fit: $w={w_fit0[2]:.3f}$")
    axs[0].grid(True)
    axs[0].legend()
    axs[1].plot(centers30,wall_fit(centers30, *w_fit30), '-',color=canard,label=r"Fit")
    axs[1].plot(centers30[:len(phi30)], phi30,'r.', label="Deformation")
    axs[1].axvline(x=w_fit30[0]-2*w_fit30[2],color='k', linestyle='--',label=r"$x_0\pm2w$")
    axs[1].axvline(x=w_fit30[0]+2*w_fit30[2],color='k', linestyle='--')
    axs[1].set_xlim(centers30[0],centers30[round(len(centers30)/2)])
    axs[1].set_xlabel(rf"${dir30}$ [Å]")
    axs[1].set_ylabel(r"$\|(\phi_x,\phi_y)\|$[Å]")
    axs[1].set_title(rf"x° Wall fit: $w={w_fit30[2]:.3f}$")
    axs[1].grid(True)
    axs[1].legend()
    axs[2].plot(centers60,wall_fit(centers60, *w_fit60), '-',color=canard,label=r"Fit")
    axs[2].plot(centers60[:len(phi60)], phi60,'r.', label="Deformation")
    axs[2].axvline(x=w_fit60[0]-2*w_fit60[2],color='k', linestyle='--',label=r"$x_0\pm2w$")
    axs[2].axvline(x=w_fit60[0]+2*w_fit60[2],color='k', linestyle='--')
    axs[2].set_xlim(centers60[0],centers60[round(len(centers60)/2)])
    axs[2].set_xlabel(rf"$y$ [Å]")
    axs[2].set_ylabel(r"$\|(\phi_x,\phi_y)\|$[Å]")
    axs[2].set_title(rf"{dir60}° Wall fit: $w={w_fit60[2]:.3f}$")
    axs[2].grid(True)
    axs[2].legend()
    axs[3].plot(centers90,wall_fit(centers90, *w_fit90), '-',color=canard,label=r"Fit")
    axs[3].plot(centers90[:len(phi90)], phi90,'r.', label="Deformation")
    axs[3].axvline(x=w_fit90[0]-2*w_fit90[2],color='k', linestyle='--',label=r"$x_0\pm2w$")
    axs[3].axvline(x=w_fit90[0]+2*w_fit90[2],color='k', linestyle='--')
    axs[3].set_xlim(centers90[0],centers90[round(len(centers90)/2)])
    axs[3].set_xlabel(rf"$x$ [Å]")
    axs[3].set_ylabel(r"$\|(\phi_x,\phi_y)\|$[Å]")
    axs[3].set_title(rf"{dir90}° Wall fit: $w={w_fit90[2]:.3f}$")
    axs[3].grid(True)
    axs[3].legend()
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    axs = axs.flatten()
    axs[0].plot(centers0,deformations0[:,2],'r-')
    axs[0].set_xlabel(rf"$y$ [Å]")
    axs[0].set_xlim(centers0[0],centers0[round(len(centers0)/2)])
    axs[0].set_ylabel(r"Interlayer distance [Å]")
    axs[0].set_title(f"{dir0}° Wall")
    axs[0].grid(True)
    axs[1].plot(centers30,deformations30[:,2],'r-')
    axs[1].set_xlabel(rf"${dir30}$ [Å]")
    axs[1].set_xlim(centers30[0],centers30[round(len(centers30)/2)])
    axs[1].set_ylabel(r"Interlayer distance [Å]")
    axs[1].set_title(f"{dir30}° Wall")
    axs[1].grid(True)
    axs[2].plot(centers60,deformations60[:,2],'r-')
    axs[2].set_xlabel(rf"$y$ [Å]")
    axs[2].set_xlim(centers60[0],centers60[round(len(centers60)/2)])
    axs[2].set_ylabel(r"Interlayer distance [Å]")
    axs[2].set_title(f"{dir60}° Wall")
    axs[2].grid(True)
    axs[3].plot(centers90,deformations90[:,2],'r-')
    axs[3].set_xlabel(rf"$x$ [Å]")
    axs[3].set_xlim(centers90[0],centers90[round(len(centers90)/2)])
    axs[3].set_ylabel(r"Interlayer distance [Å]")
    axs[3].set_title(f"{dir90}° Wall")
    axs[3].grid(True)
    plt.tight_layout()
    plt.show()

    minpolarization=np.min(np.concatenate((polarizations0,polarizations30,polarizations60,polarizations90)))
    maxpolarization=np.max(np.concatenate((polarizations0,polarizations30,polarizations60,polarizations90)))
    dpolarization=maxpolarization-minpolarization

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    axs = axs.flatten()
    axs[0].plot(centers0,polarizations0[:,2],'r-')
    axs[0].set_xlabel(rf"$y$ [Å]")
    axs[0].set_ylabel(r"Out-of-plane polarization [e/Å]")
    axs[0].set_title(f"{dir0}° Wall")
    axs[0].set_xlim(centers0[0],centers0[round(len(centers0)/2)])
    axs[0].set_ylim(minpolarization-0.05*dpolarization,maxpolarization+0.05*dpolarization)
    axs[0].grid(True)
    axs[1].plot(centers30,polarizations30[:,2],'r-')
    axs[1].set_xlabel(rf"${dir30}$ [Å]")
    axs[1].set_ylabel(r"Out-of-plane polarization [e/Å]")
    axs[1].set_title(f"{dir30}° Wall")
    axs[1].set_xlim(centers30[0],centers30[round(len(centers30)/2)])
    axs[1].set_ylim(minpolarization-0.05*dpolarization,maxpolarization+0.05*dpolarization)
    axs[1].grid(True)
    axs[2].plot(centers60,polarizations60[:,2],'r-')
    axs[2].set_xlabel(rf"$y$ [Å]")
    axs[2].set_ylabel(r"Out-of-plane polarization [e/Å]")
    axs[2].set_title(f"{dir60}° Wall")
    axs[2].set_xlim(centers60[0],centers60[round(len(centers60)/2)])
    axs[2].set_ylim(minpolarization-0.05*dpolarization,maxpolarization+0.05*dpolarization)
    axs[2].grid(True)
    axs[3].plot(centers90,polarizations90[:,2],'r-')
    axs[3].set_xlabel(rf"$x$ [Å]")
    axs[3].set_ylabel(r"Out-of-plane polarization [e/Å]")
    axs[3].set_title(f"{dir90}° Wall")
    axs[3].set_xlim(centers90[0],centers90[round(len(centers90)/2)])
    axs[3].set_ylim(minpolarization-0.05*dpolarization,maxpolarization+0.05*dpolarization)
    axs[3].grid(True)
    plt.tight_layout()
    plt.show()

    buckling0-=50+6.6612/4
    buckling30-=50+6.6612/4
    buckling60-=50+6.6612/4
    buckling90-=50+6.6612/4

    minbuckling=np.min(np.concatenate((buckling0,buckling30,buckling60,buckling90)))
    maxbuckling=np.max(np.concatenate((buckling0,buckling30,buckling60,buckling90)))
    dbuckling=maxbuckling-minbuckling

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    axs = axs.flatten()
    axs[0].plot(centers0,buckling0,'r-')
    axs[0].set_xlabel(rf"$y$ [Å]")
    axs[0].set_ylabel(r"Out-of-plane buckling [Å]")
    axs[0].set_xlim(centers0[0],centers0[round(len(centers0)/2)])
    axs[0].set_ylim(minbuckling-0.1*dbuckling,maxbuckling+0.1*dbuckling)
    axs[0].set_title(f"{dir0}° Wall")
    axs[0].grid(True)
    axs[1].plot(centers30,buckling30,'r-')
    axs[1].set_xlabel(rf"${dir30}$ [Å]")
    axs[1].set_ylabel(r"Out-of-plane buckling [Å]")
    axs[1].set_xlim(centers30[0],centers30[round(len(centers30)/2)])
    axs[1].set_title(f"{dir30}° Wall")
    axs[1].grid(True)
    axs[1].set_ylim(minbuckling-0.1*dbuckling,maxbuckling+0.1*dbuckling)
    axs[2].plot(centers60,buckling60,'r-')
    axs[2].set_xlabel(rf"$y$ [Å]")
    axs[2].set_ylabel(r"Out-of-plane buckling [Å]")
    axs[2].set_xlim(centers60[0],centers60[round(len(centers60)/2)])
    axs[2].set_ylim(minbuckling-0.1*dbuckling,maxbuckling+0.1*dbuckling)
    axs[2].set_title(f"{dir60}° Wall")
    axs[2].grid(True)
    axs[3].plot(centers90,buckling90,'r-')
    axs[3].set_xlabel(rf"$x$ [Å]")
    axs[3].set_ylabel(r"Out-of-plane buckling [Å]")
    axs[3].set_xlim(centers90[0],centers90[round(len(centers90)/2)])
    axs[3].set_ylim(minbuckling-0.1*dbuckling,maxbuckling+0.1*dbuckling)
    axs[3].set_title(f"{dir90}° Wall")
    axs[3].grid(True)
    plt.tight_layout()
    plt.show()
    '''
    plt.plot(centers, wall_fit(centers, *w_fit), '-',color=canard,label=r"Fit")
    plt.plot(centers[:len(phi)], phi,'r.', label="Deformation")
    plt.axvline(x=w_fit[0]-2*w_fit[2],color='k', linestyle='--',label=r"$x_0\pm2w$")
    plt.axvline(x=w_fit[0]+2*w_fit[2],color='k', linestyle='--')
    plt.xlim(centers[0],centers[round(len(centers)/2)])
    plt.xlabel(rf"${la}$ [Å]")
    plt.ylabel(r"$\|(\phi_x,\phi_y)\|$[Å]")    
    plt.title(rf"{dir}° Wall fit: $w={w_fit[2]:.3f}$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    '''
#full_full_creation_analysis()

def rebuild_system(Nx,Ny,dir,dV=0):
    #layer=np.loadtxt(f"layer_{dir}.txt")
    layer=np.zeros(8*Nx*Ny)
    layer+=1
    layer[Nx*Ny*4:]+=layer[Nx*Ny*4:] # second half is layer 2
    system=read(f"DW_{dir}.lammps",format="lammps-data")
    #system=read(f"tmp1D_2.lammps",format="lammps-data")
    system.set_array("mol-id", layer, dtype=int)
    born, charges_lammps,charges=born_charges(system)

    system.set_initial_charges(charges_lammps)
    system.set_array("born", born)
    system.set_array("charges_2", charges)
    system.set_array("charges_model", charges)
    voltages = [dV/2,-dV/2]

    if dir=='0' or dir=='60':
            la='y'
            Ns=Ny
            Nt=Nx
    elif dir=='30' or dir == '90':
            la='x'
            Ns=Nx
            Nt=Ny
    if la=='x':
            Lt=Ly
            Ll=Lx
            truth_axis=0
            other_axis=1
    elif la=='y':
            Lt=Lx
            Ll=Ly
            truth_axis=1
            other_axis=0

    layer=system.get_array("mol-id")
    voltage=np.zeros(len(system))
    for i in range(len(system)):
        voltage[i]= voltages[int(layer[i])-1]
    system.set_array("voltage", voltage, dtype=float)
    #system.calc=LAMMPS()
    #system.get_potential_energy()

    return system

def rebuild_system_def(Nx,Ny,dir,defo,freeze_indices=None,dV=0,relax=False):
    #layer=np.loadtxt(f"layer_{dir}.txt")
    layer=np.zeros(8*Nx*Ny)
    layer+=1
    layer[Nx*Ny*4:]+=layer[Nx*Ny*4:] # second half is layer 2

    if relax:
        print("WARNING: Calculation can only be done if no LAMMPS Python instance is running.")
        with open("input_def.lammps","r") as f:
            input_lines=f.readlines()
        if freeze_indices is not None:
            freeze_string="group freeze id "
            for index in freeze_indices:
                freeze_string+=f"{index} "
            freeze_string+="\nfix 1 freeze setforce 0.0 NULL NULL\n"
            input_lines[24]=freeze_string
            with open("input_def.lammps","w") as f:
                f.writelines(input_lines)
        os.system(f"cp displaced_DW_{defo:.2f}.lmp defo_temp.lammps")
        os.system("mpirun -np 16 lmp -in input_def.lammps > log_def.lammps")
        os.system("lmp -restart2data lammps.restart tmp1D_2.lammps > log2.lammps")
        system=read(f"tmp1D_2.lammps",format="lammps-data")
    else:
        system=read(f"displaced_DW_{defo:.2f}.lmp",format="lammps-data")
    system.set_array("mol-id", layer, dtype=int)
    born, charges_lammps,charges=born_charges(system)

    system.set_initial_charges(charges_lammps)
    system.set_array("born", born)
    system.set_array("charges_2", charges)
    system.set_array("charges_model", charges)

    voltages = [dV/2,-dV/2]

    if dir=='0' or dir=='60':
            la='y'
            Ns=Ny
            Nt=Nx
    elif dir=='30' or dir == '90':
            la='x'
            Ns=Nx
            Nt=Ny
    if la=='x':
            Lt=Ly
            Ll=Lx
            truth_axis=0
            other_axis=1
    elif la=='y':
            Lt=Lx
            Ll=Ly
            truth_axis=1
            other_axis=0

    layer=system.get_array("mol-id")
    voltage=np.zeros(len(system))
    for i in range(len(system)):
        voltage[i]= voltages[int(layer[i])-1]
    system.set_array("voltage", voltage, dtype=float)
    #system.calc=LAMMPS()
    #system.get_potential_energy()

    return system