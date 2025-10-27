import numpy as np
import matplotlib.pyplot as plt
from model.build_1D import build_1D
from utils_1D import minimize_lammps
from model.calculator import LAMMPS
from ase.md.verlet import VelocityVerlet
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.units import fs, kB
from model.born import dipole_moment, KDborn, born_charges
from tqdm import trange
from ase.io.trajectory import Trajectory
from scipy.optimize import curve_fit
from model.draw import cmaps
from ase.io import read
from utils_1D import build_1D_charges
import os
from model.born import KDborn
from scipy.fft import ifftshift
from ase.io import write

Lflep, Lflep_r, Dflep, Dflep_r=cmaps()
def printenergy(a):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    #print(a.velocities)
    print(
        'Epot = %.5feV  Ekin = %.5feV (T=%3.0fK)  '
        'Etot = %.5feV d_z = %.5f' % (epot, ekin, ekin / (1.5 * kB), epot + ekin,dipole_moment(a)[2])
    )
Lx=4.349179577805451#2.511
Ly=2.511

def wall_fit(x,x1,x2,w,a0,c): # Double wall fit
     return a0*np.tanh(2*(x-x1)/abs(w))-a0*np.tanh(2*(x-x2)/abs(w))+c

def atoms2lammps(system,fileName):
    """Function to convert an ASE trajectory to a LAMMPS data file"""
    print(fileName)
    write(fileName, system, format='lammps-data',velocities=True,atom_style='full')


def create_minimize(Nx,Ny,dir):
    defo=True
    build_1D(Nx=Nx,Ny=Ny,defo=defo,dir=dir,fileName="tmp1D_2.lammps")
    atoms=read('tmp1D_2.lammps',format="lammps-data")
    layer = np.zeros(len(atoms))
    mid = np.mean(atoms.get_positions()[:, 2])
    layer[atoms.get_positions()[:, 2] < mid] = 1
    layer[atoms.get_positions()[:, 2] > mid] = 2
    atoms.set_array("mol-id", layer, dtype=int)
    born, charges_lammps,charges=born_charges(atoms)
    
    build_1D_charges(atoms,charges_lammps,fileName="tmp1D_3.lammps")
    os.system("mpirun -np 16 lmp -in input_walls.lammps ")

    os.system(f"lmp -restart2data lammps.restart DW_{dir}.lammps > log2.lammps")
    np.savetxt(f"layer_{dir}.txt",layer)

def get_polarizations(system,Nx,Ny,dir):
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
            for k in range(4):
                polarizations[l] += system.get_array("charges_model")[4*(i+j*Nx)+k]*system.positions[4*(i+j*Nt)+k]/Nt
                polarizations[l]+=system.get_array("charges_model")[4*(i+j*Nx)+k+round(len(system)/2)]*system.positions[4*(i+j*Nt)+k+round(len(system)/2)]/Nt
                deformations[l] += system.positions[4*(i+j*Nx)+k+round(len(system)/2)] - system.positions[4*(i+j*Nx)+k]
    deformations/=4*Nt
    a=2.511
    if dir=='0' or dir=='90':
        SP_v=np.array([0,a*np.sqrt(3)/3])
    elif dir=='60' or dir =='30':
        SP_v=np.array([-a*np.sqrt(3)/3*np.cos(np.pi/3)/2,a*np.sqrt(3)/3*np.sin(np.pi/3)/2])
    phi=np.linalg.norm(deformations[:,:2],axis=1)#-SP_v,axis=1)

    polarizations /= (Ll*Lt)
    return polarizations, phi


def run_dw_dynamics(Nx,Ny,dir,Nsteps,dV,system):

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

    
    print(len(system))
    system.calc=LAMMPS()
    system.get_potential_energy()
    timestep = 1 * fs
    layer=system.get_array("mol-id")
    voltage=np.zeros(len(system))
    for i in range(len(system)):
        voltage[i]= voltages[int(layer[i])-1]
    system.set_array("voltage", voltage, dtype=float)

    times=np.arange(Nsteps)*timestep/fs
    xs=Ll*np.arange(Ns)
                    
    polarizations=np.zeros((Nsteps,Ns,3))
    deformations=np.zeros((Nsteps,Ns))
    dipoles=np.zeros(Nsteps)
    traj=Trajectory(f'DW2_dir_{dir}.traj', 'w', system)
    dyn = VelocityVerlet(system, timestep=timestep)
    dyn.attach(traj.write, interval=10)
    fits=np.zeros((Nsteps,5))
    current_p0=np.array([xs[round(len(xs)/3)],xs[round(len(xs)*2/3)],1,100,1])  # Initial guess for the fit parameters
    for i in trange(Nsteps):
        dyn.run(1)
        dipoles[i] = dipole_moment(system)[2]
        polarizations[i,:,:],deformations[i]= get_polarizations(system,Nx=Nx,Ny=Ny,dir=dir)
        fits[i,:],_=curve_fit(wall_fit,xs,deformations[i],p0=current_p0,maxfev=10000)
        current_p0=fits[i,:] # This makes it easier to perform the fit on the next step
        if i%10==0:
            printenergy(system)

    print("Fit values ",fits[-1])
    from matplotlib import colors
    print(np.min(polarizations[:,:,2]))
    divnorm=colors.TwoSlopeNorm(vmin=np.min(polarizations[:,:,2]), vcenter=0, vmax=np.max(polarizations[:,:,2]))

    plt.figure()
    plt.plot(np.arange(Nsteps)*timestep/fs, dipoles, 'r-')
    plt.xlabel("Time [ps]")
    plt.ylabel(r"$d_z$ [eÅ]")
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(xs, deformations[-1], 'rx')
    plt.plot(xs, wall_fit(xs, *fits[-1]), 'b--', label='Fit')
    plt.xlabel(r"$x$ [Å]")
    plt.ylabel(r"$\phi$ [Å]")
    plt.title(f"{dir}° Wall, $\Delta V=${voltages[1]-voltages[0]} V, $t=${times[-1]/1000} ps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(xs, deformations[0], 'rx')
    plt.plot(xs, wall_fit(xs, *fits[-1]), 'b--', label='Fit')
    plt.xlabel(r"$x$ [Å]")
    plt.ylabel(r"$\phi$ [Å]")
    plt.legend()
    plt.title(f"{dir}° Wall, $\Delta V=${voltages[1]-voltages[0]} V, $t=0$")
    plt.grid(True)
    plt.tight_layout()
    # Width
    plt.figure()
    plt.plot(times, fits[:,2], 'r-', label='Width')
    plt.xlabel("Time [ps]")
    plt.ylabel("DW Width [Å]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Gamma Lorentz
    plt.figure()
    plt.plot(times, fits[:,2]/fits[0,2], 'r-')
    plt.xlabel("Time [ps]")
    plt.ylabel(r"$\gamma_L$")
    plt.grid(True)
    plt.tight_layout()

    # Velocity in units of vc
    plt.figure()
    plt.plot(times,-(fits[:,2]/fits[0,2])**2+1, 'r-')
    plt.xlabel("Time [ps]")
    plt.ylabel(r"$v/v_c$")
    plt.grid(True)
    plt.tight_layout()

    # DW positions
    plt.figure()
    plt.plot(times, fits[:,0], 'r-', label='Position 1')
    plt.plot(times, fits[:,1], 'b-', label='Position 2')
    plt.xlabel("Time [ps]")
    plt.ylabel("DW Position [Å]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    T, X= np.meshgrid(times, xs)
    plt.figure()
    plt.contourf(T,X,polarizations[:,:,2].T,cmap=Lflep,norm=divnorm,levels=50)
    plt.colorbar()
    traj.close()


def run_dw_dynamics_2(Nx,Ny,dir,NstepsEfield,Nsteps,dV,NVT=False,Temperature=0):
    layer=np.loadtxt(f"layer_{dir}.txt")
    system=read(f"DW_{dir}.lammps",format="lammps-data")
    system.set_array("mol-id", layer, dtype=int)
    born, charges_lammps,charges=born_charges(system)

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
    print(len(system))
    system.calc=LAMMPS()
    system.get_potential_energy()
    timestep = 1 * fs
    

    times=np.arange(Nsteps)*timestep/fs
    xs=Ll*np.arange(Ns)

    if not NVT:
        dyn = VelocityVerlet(system, timestep=timestep)
        mes='_'
    else:
        dyn = NoseHooverChainNVT(system, timestep=timestep, temperature_K=Temperature)
        dyn.run(100)  # Equilibrate for 100 steps
        mes='_NVT_'+str(Temperature)+'K_'

    polarizations=np.zeros((Nsteps,Ns,3))
    deformations=np.zeros((Nsteps,Ns))
    dipoles=np.zeros(Nsteps)
    traj=Trajectory(f'data/DW2{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.traj', 'w', system)
    
    dyn.attach(traj.write, interval=10)
    fits=np.zeros((Nsteps,5))
    current_p0=np.array([xs[round(len(xs)/3)],xs[round(len(xs)*2/3)],1,100,1])  # Initial guess for the fit parameters
    for i in trange(Nsteps):
        if i == NstepsEfield:
            voltage=np.zeros(len(system))
            #for j in range(len(system)):
            #    voltage[j]= voltages[1-int(layer[j])-1]
            system.set_array("voltage", voltage, dtype=float)
        dyn.run(1)
        dipoles[i] = dipole_moment(system)[2]
        polarizations[i,:,:],deformations[i]= get_polarizations(system,Nx=Nx,Ny=Ny,dir=dir)
        fits[i,:],_=curve_fit(wall_fit,xs,deformations[i],p0=current_p0,maxfev=10000)
        current_p0=fits[i,:] # This makes it easier to perform the fit on the next step
        if i%100==0:
            printenergy(system)
            np.savetxt(f"data/fit2_vals{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.txt",fits)
            np.save(f"data/polarizations2{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.npy",polarizations[:,:,2])
            np.save(f"data/deformations2{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.npy",deformations)
    '''
    print("Fit values ",fits[-1])
    from matplotlib import colors
    print(np.min(polarizations[:,:,2]))
    #divnorm=colors.TwoSlopeNorm(vmin=np.min(polarizations[:,:,2]), vcenter=0, vmax=np.max(polarizations[:,:,2]))
    divnorm=colors.TwoSlopeNorm(vmin=-0.0004, vcenter=0, vmax=0.0004)
    plt.figure()
    plt.plot(np.arange(Nsteps)*timestep/fs, dipoles, 'r-')
    plt.xlabel("Time [ps]")
    plt.ylabel(r"$d_z$ [eÅ]")
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(xs, deformations[-1], 'rx')
    plt.plot(xs, wall_fit(xs, *fits[-1]), 'b--', label='Fit')
    plt.xlabel(r"$x$ [Å]")
    plt.ylabel(r"$\phi$ [Å]")
    plt.title(f"{dir}° Wall, $\Delta V=${voltages[1]-voltages[0]} V, $t=${times[-1]/1000} ps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(xs, deformations[0], 'rx')
    plt.plot(xs, wall_fit(xs, *fits[-1]), 'b--', label='Fit')
    plt.xlabel(r"$x$ [Å]")
    plt.ylabel(r"$\phi$ [Å]")
    plt.legend()
    plt.title(f"{dir}° Wall, $\Delta V=${voltages[1]-voltages[0]} V, $t=0$")
    plt.grid(True)
    plt.tight_layout()
    # Width
    plt.figure()
    plt.plot(times, fits[:,2], 'r-', label='Width')
    plt.xlabel("Time [ps]")
    plt.ylabel("DW Width [Å]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Gamma Lorentz
    plt.figure()
    plt.plot(times, fits[:,2]/fits[0,2], 'r-')
    plt.xlabel("Time [ps]")
    plt.ylabel(r"$\gamma_L$")
    plt.grid(True)
    plt.tight_layout()

    # Velocity in units of vc
    plt.figure()
    plt.plot(times,-(fits[:,2]/fits[0,2])**2+1, 'r-')
    plt.xlabel("Time [ps]")
    plt.ylabel(r"$v/v_c$")
    plt.grid(True)
    plt.tight_layout()

    # DW positions
    plt.figure()
    plt.plot(times, fits[:,0], 'r-', label='Position 1')
    plt.plot(times, fits[:,1], 'b-', label='Position 2')
    plt.xlabel("Time [ps]")
    plt.ylabel("DW Position [Å]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    T, X= np.meshgrid(times, xs)
    plt.figure()
    plt.contourf(T,X,polarizations[:,:,2].T,cmap=Lflep,norm=divnorm,levels=50)
    plt.colorbar()
    '''
    traj.close()
    times=np.arange(Nsteps)
    np.savetxt(f"data/times{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.txt",times)
    np.savetxt(f"data/fit2_vals{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.txt",fits)
    np.save(f"data/polarizations2{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.npy",polarizations[:,:,2])
    np.save(f"data/deformations2{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.npy",deformations)
    atoms2lammps(system,f"data/DW2{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.lammps")

def plot_dynamics(dir,dV,Nx,Ny):
    fits=np.loadtxt(f"fit2_vals_{dir}_{dV}.txt")
    polarizations=np.load(f"polarizations2_{dir}_{dV}.npy")
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

    times=np.arange(len(fits))*1*fs/fs
    xs=Ll*np.arange(Ns)
    from matplotlib import colors
    print(np.min(polarizations[:,:]))
    divnorm=colors.TwoSlopeNorm(vmin=np.min(polarizations[:,:]), vcenter=0, vmax=np.max(polarizations[:,:]))
    #divnorm=colors.TwoSlopeNorm(vmin=-0.0004, vcenter=0, vmax=0.0004)
    plt.figure()
    plt.plot(times, fits[:,2], 'r-', label='Width')
    plt.xlabel("Time [ps]")
    plt.ylabel("DW Width [Å]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Gamma Lorentz
    plt.figure()
    plt.plot(times, fits[:,2]/fits[0,2], 'r-')
    plt.xlabel("Time [ps]")
    plt.ylabel(r"$\gamma_L$")
    plt.grid(True)
    plt.tight_layout()

    # Velocity in units of vc
    plt.figure()
    plt.plot(times,-(fits[:,2]/fits[0,2])**2+1, 'r-')
    plt.xlabel("Time [ps]")
    plt.ylabel(r"$v/v_c$")
    plt.grid(True)
    plt.tight_layout()
    # Acceleration in units of vc/ps
    plt.figure()
    plt.plot(times[1:],np.diff(-(fits[:,2]/fits[0,2])**2+1)/(times[1]-times[0]), 'r-')
    plt.xlabel("Time [ps]")
    plt.ylabel(r"$a/v_c$ [1/ps]")
    plt.grid(True)
    plt.tight_layout()
    # DW positions
    plt.figure()
    plt.plot(times, fits[:,0], 'r-', label='Position 1')
    plt.plot(times, fits[:,1], 'b-', label='Position 2')
    plt.xlabel("Time [ps]")
    plt.ylabel("DW Position [Å]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    T, X= np.meshgrid(times, xs)
    plt.figure()
    plt.contourf(T,X,polarizations[:,:].T,cmap=Lflep,norm=divnorm,levels=50)
    plt.colorbar()

    #fits=np.loadtxt("fit2_vals_0_-5.txt")
    #polarizations=np.load("polarizations2_0_1.npy")

    positions=fits[:,0]
    #widths=fits[:,1]
    #heights=fits[:,2]
    #centers=fits[:,3]

    velocities=np.diff(positions)
    accelerations=np.diff(velocities)

    plt.figure()
    plt.grid(True)
    plt.plot(positions,'r-')
    plt.figure()
    plt.grid(True)
    plt.plot(velocities,'r-')
    plt.figure()
    plt.grid(True)
    plt.plot(accelerations,'r-')


    plt.show()

def plot_dynamics_new(Nx,Ny,dir,NstepsEfield,Nsteps,dV,NVT=False,Temperature=0):

    if not NVT:
        mes='_'
    else:
        mes='_NVT_'+str(Temperature)+'K_'

    
    fits=np.loadtxt(f"data/fit2_vals{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.txt")
    polarizations=np.load(f"data/polarizations2{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.npy")
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

    times=np.arange(len(fits))*1*fs/fs
    xs=Ll*np.arange(Ns)
    from matplotlib import colors
    print(np.min(polarizations[:,:]))
    divnorm=colors.TwoSlopeNorm(vmin=np.min(polarizations[:,:]), vcenter=0, vmax=np.max(polarizations[:,:]))
    #divnorm=colors.TwoSlopeNorm(vmin=-0.0004, vcenter=0, vmax=0.0004)
    plt.figure()
    plt.plot(times, fits[:,2], 'r-', label='Width')
    plt.xlabel("Time [ps]")
    plt.ylabel("DW Width [Å]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Gamma Lorentz
    plt.figure()
    plt.plot(times, fits[:,2]/fits[0,2], 'r-')
    plt.xlabel("Time [ps]")
    plt.ylabel(r"$\gamma_L$")
    plt.grid(True)
    plt.tight_layout()

    # Velocity in units of vc
    plt.figure()
    plt.plot(times,-(fits[:,2]/fits[0,2])**2+1, 'r-')
    plt.xlabel("Time [ps]")
    plt.ylabel(r"$v/v_c$")
    plt.grid(True)
    plt.tight_layout()
    # Acceleration in units of vc/ps
    plt.figure()
    plt.plot(times[1:],np.diff(-(fits[:,2]/fits[0,2])**2+1)/(times[1]-times[0]), 'r-')
    plt.xlabel("Time [ps]")
    plt.ylabel(r"$a/v_c$ [1/ps]")
    plt.grid(True)
    plt.tight_layout()
    # DW positions
    plt.figure()
    plt.plot(times, fits[:,0], 'r-', label='Position 1')
    plt.plot(times, fits[:,1], 'b-', label='Position 2')
    plt.xlabel("Time [ps]")
    plt.ylabel("DW Position [Å]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    T, X= np.meshgrid(times, xs)
    plt.figure()
    plt.contourf(T,X,polarizations[:,:].T,cmap=Lflep,norm=divnorm,levels=50)
    plt.colorbar()

    #fits=np.loadtxt("fit2_vals_0_-5.txt")
    #polarizations=np.load("polarizations2_0_1.npy")

    positions=fits[:,0]
    #widths=fits[:,1]
    #heights=fits[:,2]
    #centers=fits[:,3]

    velocities=np.diff(positions)
    accelerations=np.diff(velocities)

    plt.figure()
    plt.grid(True)
    plt.plot(positions,'r-')
    plt.figure()
    plt.grid(True)
    plt.plot(velocities,'r-')
    plt.figure()
    plt.grid(True)
    plt.plot(accelerations,'r-')


    plt.show()

def plot_dynamics_continued(Nx,Ny,dir,NstepsEfield,Nsteps,dV,NVT=False,Temperature=0):

    if not NVT:
        mes='_'
    else:
        mes='_NVT_'+str(Temperature)+'K_'

    
    #fits=np.loadtxt(f"data/fit2_vals{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.txt")
    polarizations=np.load(f"data/polarizations2{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.npy")
    times=np.loadtxt(f"data/times{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.txt")
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

    #times=np.arange(len(fits))*1*fs/fs
    xs=Ll*np.arange(Ns)
    from matplotlib import colors
    print(np.min(polarizations[:,:]))
    #divnorm=colors.TwoSlopeNorm(vmin=np.min(polarizations[:,:]), vcenter=0, vmax=np.max(polarizations[:,:]))
    divnorm=colors.TwoSlopeNorm(vmin=-0.0006, vcenter=0, vmax=0.0006)
    '''
    plt.figure()
    plt.plot(times, fits[:,2], 'r-', label='Width')
    plt.xlabel("Time [ps]")
    plt.ylabel("DW Width [Å]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Gamma Lorentz
    plt.figure()
    plt.plot(times, fits[:,2]/fits[0,2], 'r-')
    plt.xlabel("Time [ps]")
    plt.ylabel(r"$\gamma_L$")
    plt.grid(True)
    plt.tight_layout()

    # Velocity in units of vc
    plt.figure()
    plt.plot(times,-(fits[:,2]/fits[0,2])**2+1, 'r-')
    plt.xlabel("Time [ps]")
    plt.ylabel(r"$v/v_c$")
    plt.grid(True)
    plt.tight_layout()
    # Acceleration in units of vc/ps
    plt.figure()
    plt.plot(times[1:],np.diff(-(fits[:,2]/fits[0,2])**2+1)/(times[1]-times[0]), 'r-')
    plt.xlabel("Time [ps]")
    plt.ylabel(r"$a/v_c$ [1/ps]")
    plt.grid(True)
    plt.tight_layout()
    # DW positions
    plt.figure()
    plt.plot(times, fits[:,0], 'r-', label='Position 1')
    plt.plot(times, fits[:,1], 'b-', label='Position 2')
    plt.xlabel("Time [ps]")
    plt.ylabel("DW Position [Å]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    '''
    T, X= np.meshgrid(times, xs)
    plt.figure()
    plt.contourf(T,X,polarizations[:,:].T,cmap=Lflep,norm=divnorm,levels=50)
    plt.xlabel(r"$t$ [fs]")
    plt.ylabel(r"$x$ [Å]")
    plt.title(f"Total dipole moment [a.u.]")
    plt.colorbar()

    #fits=np.loadtxt("fit2_vals_0_-5.txt")
    #polarizations=np.load("polarizations2_0_1.npy")
    '''
    positions=fits[:,0]
    #widths=fits[:,1]
    #heights=fits[:,2]
    #centers=fits[:,3]

    velocities=np.diff(positions)
    accelerations=np.diff(velocities)

    plt.figure()
    plt.grid(True)
    plt.plot(positions,'r-')
    plt.figure()
    plt.grid(True)
    plt.plot(velocities,'r-')
    plt.figure()
    plt.grid(True)
    plt.plot(accelerations,'r-')
    '''

    #plt.show()

def continue_run(New_steps,Nevery,Nx,Ny,dir,NstepsEfield,Nsteps,dV,NVT=False,Temperature=0):
    #layer=np.loadtxt(f"layer_{dir}.txt")
    #system=read(f"data/DW2_NVT_300K_{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.lmp",format="lammps-data")
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

    if not NVT:
        mes='_'
    else:
        mes='_NVT_'+str(Temperature)+'K_' # Temperature is included for future proofing but is not currently implemented
    
    # Writing input file
    with open(f"continue.template", "r") as file:
        data = file.readlines()
    data[25]="run "+str(New_steps)+"\n"
    data[23]=f"dump Ddump all atom {Nevery} output_{NstepsEfield}.lammpstrj\n"
    data[27]=f"write_restart lammps_{NstepsEfield}.restart\n"

    with open(f"input{mes}continue.lammps", "w") as file:
        file.writelines(data)

    # Running continuation of dynamics
    os.system(f"cp data/DW2{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.lammps tmp_cont.lammps")
    os.system(f"mpirun -np 16 lmp -in input{mes}continue.lammps > continue.log")
    # Analyzing results
    layer=np.loadtxt(f"layer_{dir}.txt")
    traj=Trajectory(f'data/DW2{mes}{dir}_{dV}_{Nsteps+New_steps}_{NstepsEfield}_{Nx}_{Ny}.traj', 'a')
    ref_config=read(f"data/DW2{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.traj",index=0)
    polar=np.zeros((New_steps//Nevery,Ns,3))
    defo=np.zeros((New_steps//Nevery,Ns))
    #fi=np.zeros((New_steps//Nevery,5))
    ts=np.zeros(New_steps//Nevery)


    #fits=np.loadtxt(f"data/fit2_vals{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.txt")
    polarizations=np.load(f"data/polarizations2{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.npy")
    deformations=np.load(f"data/deformations2{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.npy")
    times=np.loadtxt(f"data/times{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.txt")

    #current_p0=fits[-1,:]
    for i in trange(New_steps//Nevery):
        system=read(f"output_{NstepsEfield}.lammpstrj",format="lammps-dump-text",index=i+1) # +1 as to not include the first step which is already included
        system.set_array("mol-id", layer, dtype=int)
        system.symbols=ref_config.symbols
        born, charges_lammps,charges=born_charges(system)

        system.set_initial_charges(charges_lammps)
        system.set_array("born", born)
        system.set_array("charges_2", charges)
        system.set_array("charges_model", charges)

        voltages = [dV/2,-dV/2]

        

        layer=system.get_array("mol-id")
        # This would only be needed to get the energy of the system
        #voltage=np.zeros(len(system))
        #for i in range(len(system)):
        #    voltage[i]= voltages[int(layer[i])-1]
        #system.set_array("voltage", voltage, dtype=float)
        #xs=Ll*np.arange(Ns)
        traj.write(system)
        polar[i],defo[i]= get_polarizations(system,Nx=Nx,Ny=Ny,dir=dir)
        ts[i]=(i+1)*Nevery+Nsteps
        #fi[i,:],_=curve_fit(wall_fit,xs,defo[i],p0=current_p0,maxfev=10000)
        #current_p0=fi[i,:]
    
    #fits=np.concatenate((fits,fi),axis=0)
    polarizations=np.concatenate((polarizations,polar[:,:,2]),axis=0)
    deformations=np.concatenate((deformations,defo),axis=0)
    times=np.concatenate((times,ts),axis=0)
    traj.close()
    # If everything went well, we store the results
    os.system(f"lmp -restart2data lammps_{NstepsEfield}.restart continue_{NstepsEfield}.lmp > log2.lammps")
    os.system(f"cp continue_{NstepsEfield}.lmp data/DW2{mes}{dir}_{dV}_{Nsteps+New_steps}_{NstepsEfield}_{Nx}_{Ny}.lammps")
    #np.savetxt(f"data/fit2_vals{mes}{dir}_{dV}_{Nsteps+New_steps}_{NstepsEfield}_{Nx}_{Ny}.txt",fits)
    np.save(f"data/polarizations2{mes}{dir}_{dV}_{Nsteps+New_steps}_{NstepsEfield}_{Nx}_{Ny}.npy",polarizations)
    np.save(f"data/deformations2{mes}{dir}_{dV}_{Nsteps+New_steps}_{NstepsEfield}_{Nx}_{Ny}.npy",deformations) 
    np.savetxt(f"data/times{mes}{dir}_{dV}_{Nsteps+New_steps}_{NstepsEfield}_{Nx}_{Ny}.txt",times)
 
    #atoms2lammps(system,f"data/DW2{mes}{dir}_{dV}_{Nsteps+New_steps}_{NstepsEfield}_{Nx}_{Ny}.lammps")

    #np.savetxt(f"fit_test.txt",fits)
    #np.save(f"polarizations_test.npy",polarizations)
    #np.save(f"deformations_test.npy",deformations)

def continue_lammps(New_steps,Nevery,Nx,Ny,dir,NstepsEfield,Nsteps,dV,NVT=False,Temperature=0):
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

    if not NVT:
        mes='_'
    else:
        mes='_NVT_'+str(Temperature)+'K_' # Temperature is included for future proofing but is not currently implemented
    
    # Writing input file
    with open(f"continue.template", "r") as file:
        data = file.readlines()
    data[25]="run "+str(New_steps)+"\n"
    data[23]=f"dump Ddump all atom {Nevery} output_{NstepsEfield}.lammpstrj\n"
    data[27]=f"write_restart lammps_{NstepsEfield}.restart\n"

    with open(f"input{mes}continue.lammps", "w") as file:
        file.writelines(data)

    # Running continuation of dynamics
    os.system(f"cp data/DW2{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.lammps tmp_cont.lammps")
    os.system(f"mpirun -np 16 lmp -in input{mes}continue.lammps > continue.log")

def analyze_lammps(New_steps,Nevery,Nx,Ny,dir,NstepsEfield,Nsteps,dV,NVT=False,Temperature=0):
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
    if not NVT:
        mes='_'
    else:
        mes='_NVT_'+str(Temperature)+'K_' # Temperature is included for future proofing but is not currently implemented

    layer=np.loadtxt(f"layer_{dir}.txt")
    traj=Trajectory(f'data/DW2{mes}{dir}_{dV}_{Nsteps+New_steps}_{NstepsEfield}_{Nx}_{Ny}.traj', 'a')
    ref_config=read(f"data/DW2{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.traj",index=0)
    polar=np.zeros((New_steps//Nevery,Ns,3))
    defo=np.zeros((New_steps//Nevery,Ns))
    #fi=np.zeros((New_steps//Nevery,5))
    ts=np.zeros(New_steps//Nevery)


    #fits=np.loadtxt(f"data/fit2_vals{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.txt")
    polarizations=np.load(f"data/polarizations2{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.npy")
    deformations=np.load(f"data/deformations2{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.npy")
    times=np.loadtxt(f"data/times{mes}{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.txt")

    #current_p0=fits[-1,:]
    for i in trange(New_steps//Nevery):
        system=read(f"output_{NstepsEfield}.lammpstrj",format="lammps-dump-text",index=i+1) # +1 as to not include the first step which is already included
        system.set_array("mol-id", layer, dtype=int)
        system.symbols=ref_config.symbols
        born, charges_lammps,charges=born_charges(system)

        system.set_initial_charges(charges_lammps)
        system.set_array("born", born)
        system.set_array("charges_2", charges)
        system.set_array("charges_model", charges)

        voltages = [dV/2,-dV/2]

        

        layer=system.get_array("mol-id")
        # This would only be needed to get the energy of the system
        #voltage=np.zeros(len(system))
        #for i in range(len(system)):
        #    voltage[i]= voltages[int(layer[i])-1]
        #system.set_array("voltage", voltage, dtype=float)
        #xs=Ll*np.arange(Ns)
        traj.write(system)
        polar[i],defo[i]= get_polarizations(system,Nx=Nx,Ny=Ny,dir=dir)
        ts[i]=(i+1)*Nevery+Nsteps
        #fi[i,:],_=curve_fit(wall_fit,xs,defo[i],p0=current_p0,maxfev=10000)
        #current_p0=fi[i,:]
    
    #fits=np.concatenate((fits,fi),axis=0)
    polarizations=np.concatenate((polarizations,polar[:,:,2]),axis=0)
    deformations=np.concatenate((deformations,defo),axis=0)
    times=np.concatenate((times,ts),axis=0)
    traj.close()
    # If everything went well, we store the results
    os.system(f"lmp -restart2data lammps_{NstepsEfield}.restart continue_{NstepsEfield}.lmp > log2.lammps")
    os.system(f"cp continue_{NstepsEfield}.lmp data/DW2{mes}{dir}_{dV}_{Nsteps+New_steps}_{NstepsEfield}_{Nx}_{Ny}.lammps")
    #np.savetxt(f"data/fit2_vals{mes}{dir}_{dV}_{Nsteps+New_steps}_{NstepsEfield}_{Nx}_{Ny}.txt",fits)
    np.save(f"data/polarizations2{mes}{dir}_{dV}_{Nsteps+New_steps}_{NstepsEfield}_{Nx}_{Ny}.npy",polarizations)
    np.save(f"data/deformations2{mes}{dir}_{dV}_{Nsteps+New_steps}_{NstepsEfield}_{Nx}_{Ny}.npy",deformations) 
    np.savetxt(f"data/times{mes}{dir}_{dV}_{Nsteps+New_steps}_{NstepsEfield}_{Nx}_{Ny}.txt",times)