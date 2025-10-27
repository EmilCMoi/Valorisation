import numpy as np
import matplotlib.pyplot as plt
from model.build_1D import build_1D
from utils_1D import minimize_lammps
from model.calculator import LAMMPS
from ase.md.verlet import VelocityVerlet
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.units import fs, kB
from model.born import dipole_moment, KDborn
from tqdm import trange
from ase.io.trajectory import Trajectory
from scipy.optimize import curve_fit
from model.draw import cmaps
from ase.io import read
from utils_1D import build_1D_charges
import os
from model.born import KDborn
from scipy.fft import ifftshift

Lflep, Lflep_r, Dflep, Dflep_r=cmaps()
def printenergy(a):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    #print(a.velocities)
    print(
        'Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
        'Etot = %.3feV d_z = %.3f' % (epot, ekin, ekin / (1.5 * kB), epot + ekin,dipole_moment(a)[2])
    )
Lx=4.349179577805451#2.511
Ly=2.511

def rebuild_system(Nx,Ny,dir,dV=0):
    layer=np.loadtxt(f"layer_{dir}.txt")
    system=read(f"DW_{dir}.lammps",format="lammps-data")
    system.set_array("mol-id", layer, dtype=int)
    born, charges_lammps,charges=KDborn(system)

    system.set_initial_charges(charges_lammps)
    system.set_array("born", born)
    system.set_array("charges_2", charges)

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
def set_voltage(system,dV):
    voltages = [dV/2,-dV/2]
    layer=system.get_array("mol-id")
    voltage=np.zeros(len(system))
    for i in range(len(system)):
        voltage[i]= voltages[int(layer[i])-1]
    system.set_array("voltage", voltage, dtype=float)

def get_polarizations_vacancy(system,Nx,Ny,dir,N_vacancy):
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
    """Function to calculate the polarization"""
    polarizations = np.zeros((Ns,3))
    deformations = np.zeros((Ns,3))
    for i in range(Nx):
        for j in range(Ny):
            if dir=='0' or dir=='60':
                l=j
            elif dir=='30' or dir=='90':
                l=i
            if (4*(i+j*Nx)<=N_vacancy<4*(i+j*Nx)+4 or 4*(i+j*Nx)+round((len(system)+1)/2)<=N_vacancy<4*(i+j*Nx)+round((len(system)+1)/2)+4):
                continue
            # We need to remove 1 from the index if the vacancy is before it
            ib1=0
            it1=0
            if 4*(i+j*Nx)>N_vacancy:
                ib1=1
                it1=1
            if 4*(i+j*Nx)+round((len(system)+1)/2)>N_vacancy:
                it1=1
            for k in range(4):
                polarizations[l] += system.get_array("charges_model")[4*(i+j*Nx)+k-ib1]*system.positions[4*(i+j*Nt)+k-ib1]/Nt
                polarizations[l]+=system.get_array("charges_model")[4*(i+j*Nx)+k+round((len(system)+1)/2)-it1]*system.positions[4*(i+j*Nt)+k+round((len(system)+1)/2)-it1]/Nt
                deformations[l] += system.positions[4*(i+j*Nx)+k+round((len(system)+1)/2)-it1] - system.positions[4*(i+j*Nx)+k-ib1]
    deformations/=4*Nt
    a=2.511
    if dir=='0' or dir=='90':
        SP_v=np.array([0,a*np.sqrt(3)/3])
    elif dir=='60' or dir =='30':
        SP_v=np.array([-a*np.sqrt(3)/3*np.cos(np.pi/3)/2,a*np.sqrt(3)/3*np.sin(np.pi/3)/2])
    phi=np.linalg.norm(deformations[:,:2]-SP_v,axis=1)

    polarizations /= (Ll*Lt)
    return polarizations, phi

def vacancy_energy(system,dir,N_vacancy):
    assert N_vacancy<len(system), "Vacancy index out of range"
    if dir=='0' or dir=='60':
            la='y'
            Ns=len(system)//4
            Nt=1
    elif dir=='30' or dir == '90':
            la='x'
            Ns=1
            Nt=len(system)//4
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

    system_vac=system.copy()
    del system_vac[N_vacancy]
    _, charges_lammps,_=KDborn(system_vac)
    #build_1D_charges(system_vac,charges_lammps,fileName="tmp1D_3.lammps")
    #os.system("lmp -in input_walls.lammps > log2.lammps ")
    #os.system(f"lmp -restart2data lammps.restart DW_vac_{dir}.lammps > log3.lammps")
    
    #print(len(system_vac))
    #system_relaxed=read(f"DW_vac_{dir}.lammps",format="lammps-data")
    #print(len(system_relaxed))
    #system_relaxed.set_array("mol-id", system_vac.get_array("mol-id"), dtype=int)
    #system_relaxed.set_array("born", system_vac.get_array("born"))
    #system_relaxed.set_array("charges_2", system_vac.get_array("charges_2"))
    #system_relaxed.set_initial_charges(system_vac.get_initial_charges())
    #set_voltage(system_relaxed,dV=0)
    set_voltage(system_vac,dV=0)
    #comm = MPI.COMM_WORLD
    #nprocs = comm.Get_size()
    #print(nprocs)
    lmp=LAMMPS()
    #system_relaxed.calc=lmp
    #E_vac=system_relaxed.get_potential_energy()
    system_vac.calc=lmp
    E_vac=system_vac.get_potential_energy()
    #lmp.close()
    return E_vac

def minimize_cavity(system):
    _,charges_lammps,_=KDborn(system)
    build_1D_charges(system,charges_lammps,fileName="tmp1D_3.lammps")
    os.system("lmp -in input_walls.lammps ")
    os.system(f"lmp -restart2data lammps.restart DW_vac_min.lammps > log3.lammps")
    system_relaxed=read(f"DW_vac_min.lammps",format="lammps-data")
    system_relaxed.set_array("mol-id", system.get_array("mol-id"), dtype=int)
    system_relaxed.set_array("born", system.get_array("born"))
    system_relaxed.set_array("charges_2", system.get_array("charges_2"))
    system_relaxed.set_initial_charges(system.get_initial_charges())
    set_voltage(system_relaxed,dV=0)
    return system_relaxed
def vacancy_dynamics(Nx,Ny,dir,Nsteps,NstepsEfield,dV,N_vacancy):

    voltages = [-dV/2,dV/2]

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

    
    system=rebuild_system(Nx,Ny,dir,dV=0)
    del system[N_vacancy]
    system= minimize_cavity(system)
    system.calc=LAMMPS()
    timestep = 1 * fs
    layer=system.get_array("mol-id")
    voltage=np.zeros(len(system))
    #Only apply to the part not containing the vacancy
    for i in range(len(system)):
        if (i>len(system)//4 and layer[i]==1) or (i>len(system)//2 and layer[i]==2):
            voltage[i]= voltages[int(layer[i])-1]
    system.set_array("voltage", voltage, dtype=float)

    times=np.arange(Nsteps)*timestep/fs
    xs=Ll*np.arange(Ns)
                    
    polarizations=np.zeros((Nsteps,Ns,3))
    deformations=np.zeros((Nsteps,Ns))
    dipoles=np.zeros(Nsteps)
    mes=f"{dir}_{N_vacancy}_{Nsteps}_{NstepsEfield}_{dV}"
    traj=Trajectory(f'data/DW_{mes}.traj', 'w', system)
    dyn = VelocityVerlet(system, timestep=timestep)
    dyn.attach(traj.write, interval=10)
    for i in trange(Nsteps):
        dyn.run(1)
        dipoles[i] = dipole_moment(system)[2]
        polarizations[i,:,:],deformations[i]= get_polarizations_vacancy(system,Nx=Nx,Ny=Ny,dir=dir,N_vacancy=N_vacancy)
        if i == NstepsEfield:
            voltage=np.zeros(len(system))
            system.set_array("voltage", voltage, dtype=float)
        if i%100==0:
            printenergy(system)
            np.save(f"data/polarizations_{mes}.npy",polarizations[:,:,2])
            np.save(f"data/deformations_{mes}.npy",deformations)
    traj.close()

    np.save(f"data/polarizations_{mes}.npy",polarizations[:,:,2])
    np.save(f"data/deformations_{mes}.npy",deformations)

def plot_vacancy_dynamics(Nx,Ny,dir,NstepsEfield,Nsteps,dV,N_vacancy):

    mes=f"{dir}_{N_vacancy}_{Nsteps}_{NstepsEfield}_{dV}"
    polarizations=np.load(f"data/polarizations_{mes}.npy")
    if dV>0:
        polarizations=ifftshift(polarizations,axes=1)
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

    times=np.arange(Nsteps)*1*fs/fs
    xs=Ll*np.arange(Ns)
    from matplotlib import colors
    print(np.min(polarizations[:,:]))
    #divnorm=colors.TwoSlopeNorm(vmin=np.min(polarizations[:,:]), vcenter=0, vmax=np.max(polarizations[:,:]))
    divnorm=colors.TwoSlopeNorm(vmin=-0.0005, vcenter=0, vmax=0.0005)

    T, X= np.meshgrid(times, xs)
    plt.figure()
    plt.contourf(T,X,polarizations[:,:].T,norm=divnorm,cmap=Lflep,levels=5000)
    plt.axhline(Ll*N_vacancy/4/Nt,color='k',linestyle='--')
    plt.colorbar()
    plt.title(f"Vacancy at index {N_vacancy}, {dir}°, $\Delta V=${dV} V")
    plt.xlabel("Time [ps]")
    plt.ylabel(r"$x$ [Å]")

    plt.show()
