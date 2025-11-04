import numpy as np
import matplotlib.pyplot as plt
from draw import cmaps
from utils_1D import rebuild_system_def, rebuild_system, file2atoms
from scipy.spatial import KDTree
from DW_dynamics import get_polarizations, wall_fit, atoms2lammps
from scipy.optimize import curve_fit
from model.calculator import LAMMPS
from tqdm import trange
from ase.md.verlet import VelocityVerlet
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from model.born import dipole_moment, born_charges
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.units import fs, kB
Lflep, Lflep_r, Dflep, Dflep_r=cmaps()
Ly=2.511
Lx=4.349179577805451
def printenergy(a):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    #print(a.velocities)
    print(
        'Epot = %.5feV  Ekin = %.5feV (T=%3.0fK)  '
        'Etot = %.5feV d_z = %.5f' % (epot, ekin, ekin / (1.5 * kB), epot + ekin,dipole_moment(a)[2])
    )
    return epot*len(a), ekin*len(a)
def kd_freeze_indices(system,dir,x1,x2,nn):
    
    if dir=='0' or dir=='60':
        tree=KDTree(system.positions[:,1:2])
    else:
        tree=KDTree(system.positions[:,0:1])
    #print(system.positions[:,0:1])
    #print(x1,x2,np.concatenate((tree.query(x1,4)[1],tree.query(x2,4)[1])))
    return np.concatenate((tree.query(x1,nn)[1],tree.query(x2,nn)[1]))
#498 497 697 698 478 477 717 718 1677 1678 1918 1917 1320 1323 2279 2276 1697 1698 1898 1897 1300 1303 2299 2296 478 477 717 718 123 120 1076 1079 
def run_dw_dynamics_frozen(system,Nsteps,V_func,Vparams,NVT=False,Temperature=0):
    layer=np.loadtxt(f"layer_{dir}.txt")
    system=read(f"DW_{dir}.lammps",format="lammps-data")
    system.set_array("mol-id", layer, dtype=int)
    born, charges_lammps,charges=born_charges(system)

    system.set_initial_charges(charges_lammps)
    system.set_array("born", born)
    system.set_array("charges_2", charges)

    voltages = [V_func(0,Nsteps,*Vparams)/2,-V_func(0,Nsteps,*Vparams)/2]

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
        dyn = NoseHooverChainNVT(system, timestep=timestep, temperature_K=Temperature,tdamp=100*fs)
        dyn.run(100)  # Equilibrate for 100 steps
        mes='_NVT_'+str(Temperature)+'K_'

    polarizations=np.zeros((Nsteps,Ns,3))
    deformations=np.zeros((Nsteps,Ns))
    epots=np.zeros(Nsteps//100)
    ekins=np.zeros(Nsteps//100)
    timesE=[]
    dipoles=np.zeros(Nsteps)
    traj=Trajectory(f'hyst.traj', 'w', system)
    
    dyn.attach(traj.write, interval=10)
    fits=np.zeros((Nsteps,5))
    current_p0=np.array([xs[round(len(xs)/3)],xs[round(len(xs)*2/3)],1,100,1])  # Initial guess for the fit parameters
    for i in trange(Nsteps):
        dyn.run(1)
        dipoles[i] = dipole_moment(system)[2]
        polarizations[i,:,:],deformations[i]= get_polarizations(system,Nx=Nx,Ny=Ny,dir=dir)
        fits[i,:],_=curve_fit(wall_fit,xs,deformations[i],p0=current_p0,maxfev=10000)
        current_p0=fits[i,:] # This makes it easier to perform the fit on the next step
        voltages = [V_func(i,Nsteps,*Vparams)/2,-V_func(i,Nsteps,*Vparams)/2]
        voltage=np.zeros(len(system))
        for j in range(len(system)):
            voltage[j]= voltages[int(layer[j])-1]
        system.set_array("voltage", voltage, dtype=float)

        if i%100==0:
            timesE.append(i)
            epot,ekin=printenergy(system)
            np.savetxt(f"data/fit_hyst_{Nsteps}_{Vparams}.txt",fits)
            np.save(f"data/polarizations_hyst_{Nsteps}_{Vparams}.npy",polarizations[:,:,2])
            np.save(f"data/deformations_hyst_{Nsteps}_{Vparams}.npy",deformations)
            np.savez(f"data/energies_hyst_{Nsteps}_{Vparams}.npz",epot=epot,ekin=ekin,times=timesE)
            
    traj.close()
    times=np.arange(Nsteps)
    #timesE=times[::100]
    np.savetxt(f"data/times_hyst_{Nsteps}_{Vparams}.txt",times)
    np.savetxt(f"data/fit_hyst_{Nsteps}_{Vparams}.txt",fits)
    np.save(f"data/polarizations_hyst_{Nsteps}_{Vparams}.npy",polarizations[:,:,2])
    np.save(f"data/deformations_hyst_{Nsteps}_{Vparams}.npy",deformations)
    np.savez(f"data/energies_hyst_{Nsteps}_{Vparams}.npz",epot=epots,ekin=ekins,times=timesE)
    
    atoms2lammps(system,f"hyst_continue_{Nsteps}_{Vparams}.lammps")
Nx=1
Ny=300
dir='0'
defo=0.00
Dz=3.0
#bottom_indices=[498, 497, 697, 698 ,478, 477, 717, 718, 1677, 1678, 1918, 1917, 1320, 1323, 2279, 2276]
#top_indices=[1697, 1698, 1898, 1897, 1300, 1303, 2299, 2296, 478, 477, 717, 718, 123, 120, 1076, 1079]
system=rebuild_system(Nx,Ny,dir)

def initial_fit(system):
    xs=np.arange(Ny)*Ly
    _,phi=get_polarizations(system,Nx,Ny,dir)
    fit,_=curve_fit(wall_fit, xs,phi,p0=[xs[len(xs)//4],3*xs[len(xs)//4],60,2,2])
    return fit

ff=initial_fit(rebuild_system(Nx,Ny,dir))
x10=ff[0]
x20=ff[1]
w0=ff[2]
a0=ff[3]
c0=ff[4]

freeze_indices=kd_freeze_indices(system,dir,x10,x20,8)
# Positions at which interlayer distance will increase
press_indices=np.concatenate((kd_freeze_indices(system,dir,Ny*Ly/8,7*Ny*Ly/8,4),kd_freeze_indices(system,dir,3*Ny*Ly/8,5*Ny*Ly/8,4)))
# Positions at which interlayer distance will decrease
press2_indices=np.concatenate((kd_freeze_indices(system,dir,3*Ny*Ly/16,13*Ny*Ly/16,4),kd_freeze_indices(system,dir,5*Ny*Ly/16,11*Ny*Ly/16,4)))

bottom_indices=np.concatenate((press_indices[press_indices<4*Nx*Ny],press2_indices[press2_indices>=4*Nx*Ny]))
top_indices=np.concatenate((press_indices[press_indices>=4*Nx*Ny],press2_indices[press2_indices<4*Nx*Ny]))
'''
dpress=(x20-x10)/3
        dpress2=0.9*dpress
        press_indices=kd_freeze_indices(system,dir,x10+dpress,x20-dpress,4)
        # The constriction on press2_indices is the opposite of that on press_indices
        press2_indices=kd_freeze_indices(system,dir,x10+dpress2,x20-dpress2,4)
        press_mirrored=kd_freeze_indices(system,dir,x10-dpress,x20+dpress,4)
        press2_mirrored=kd_freeze_indices(system,dir,x10-dpress2,x20+dpress2,4)
        bottom_indices=np.concatenate((press_indices[press_indices<4*Nx*Ny],press2_indices[press2_indices<4*Nx*Ny]))
        top_indices=np.concatenate((press_indices[press_indices>=4*Nx*Ny],press_mirrored[press_mirrored>=4*Nx*Ny]))
        bottom_indices=np.concatenate((bottom_indices,press2_indices[press2_indices>=4*Nx*Ny],press2_mirrored[press2_mirrored>=4*Nx*Ny]))
        top_indices=np.concatenate((top_indices,press2_indices[press2_indices<4*Nx*Ny],press2_mirrored[press2_mirrored<4*Nx*Ny]))
        #setdiff1d could be useful but I won't use it now
        #constrain_bottom_indices=np.setdiff1d(bottom_indices,freeze_indices)
'''

#new_system=rebuild_system_def(Nx,Ny,dir,defo,freeze_indices=freeze_indices,relax=True,constrain=True,Dz=Dz,constrain_bottom_indices=bottom_indices,constrain_top_indices=top_indices)
new_system=file2atoms('hyst_def_basis.lmp')
new_system.calc=LAMMPS()
new_system.pbc=[True,True,False]
fr_dir=[False,False,True]
new_system.calc.freeze_atoms(np.concatenate((press_indices,press2_indices)),np.repeat(fr_dir,len(np.concatenate((press_indices,press2_indices)))))

def V_triangular(t,Nsteps,Vmax):
    if t<Nsteps/4:
        return 4*Vmax*t/Nsteps
    elif t<Nsteps*3/4:
        return 4*Vmax*(1/2 - t/Nsteps)
    else:
        return 4*Vmax*(t/Nsteps -1)
    
Nsteps=10000
Vmax=1
# Nsteps 4000 maybe more later
# Vmax 1 3 5 10
plt.figure()
plt.plot(np.arange(Nsteps),[V_triangular(t,Nsteps,Vmax) for t in range(Nsteps)])
plt.xlabel("Timestep")
plt.ylabel("Applied Voltage (V)")
plt.title("Applied voltage profile")
plt.grid()

z=np.zeros(len(new_system)//2)
for i in range(len(new_system)//2):
    z[i]=new_system.positions[i+len(new_system)//2][2]-new_system.positions[i][2]
plt.figure()
plt.plot(z)
plt.xlabel("Position along DW (A)")
plt.ylabel("Interlayer distance (A)")
plt.show()

plt.show()


run_dw_dynamics_frozen(new_system,Nsteps,V_triangular,[Vmax],NVT=False,Temperature=0)

#np.loadtxt(f"data/fit_hyst.txt")
times=np.loadtxt(f"data/times_hyst_{Nsteps}_[{Vmax}].txt")
polarizations=np.load(f"data/polarizations_hyst_{Nsteps}_[{Vmax}].npy")
#deformations=np.load(f"data/deformations_hyst.npy")
plt.figure()
plt.plot(times,polarizations.mean(axis=1))
plt.figure()
plt.plot([V_triangular(t,Nsteps,Vmax) for t in range(Nsteps)],polarizations.mean(axis=1))
plt.figure()
Ns=Ny
plt.pcolormesh(np.arange(Ns)*Ly,times,polarizations,cmap=Dflep_r)
plt.colorbar()
plt.show()


