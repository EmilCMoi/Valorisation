import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, LBFGS
from model.calculator import LAMMPS
from model.born import born_charges, dipole_moment
from ase.io.trajectory import Trajectory
from ase.mep import NEB, NEBTools
import matplotlib.pyplot as plt
from ase.visualize import view
from model.draw import cmaps

Lflep, Lflep_r, Dflep, Dflep_r = cmaps()
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})

a=2.511
c=6.6612/2

v1=a*np.array([np.sqrt(3)/2,1/2,0])
v2=a*np.array([np.sqrt(3)/2,-1/2,0])

def create_atoms(r,dV=0):
    positions=np.zeros((4,3))
    positions[0]=np.array([0,0,0])
    positions[1]=np.array([0,0,0])+v1/3+v2/3
    positions[2]=np.array([0,0,c])+r[0]*v1 +r[1]*v2+r[2]
    positions[3]=np.array([0,0,c])+v1/3+v2/3+r[0]*v1 +r[1]*v2+r[2]
    labels=["B","N","B","N"]
    atoms=Atoms(labels, positions=positions, cell=[v1, v2, [0, 0, 30]], pbc=True)
    atoms.set_array("mol-id", np.array([1, 1, 2, 2], dtype=int))
    born, charges_lammps, charges=born_charges(atoms)
    atoms.set_initial_charges(charges_lammps)
    atoms.set_array("born", born)
    atoms.set_array("charges_2", charges)
    atoms.set_array("charges_model", charges)
    voltage=[-dV/2,dV/2]
    voltages=np.zeros(len(atoms))
    voltages[atoms.get_array("mol-id")==1]=voltage[0]
    voltages[atoms.get_array("mol-id")==2]=voltage[1]
    atoms.set_array("voltage",voltages)
    atoms.calc=LAMMPS()
    return atoms

def to_model(atoms,dV=0):
    atoms.set_array("mol-id", np.array([1, 1, 2, 2], dtype=int))
    born, charges_lammps, charges=born_charges(atoms)
    atoms.set_initial_charges(charges_lammps)
    atoms.set_array("born", born)
    atoms.set_array("charges_2", charges)
    atoms.set_array("charges_model", charges)
    voltage=[-dV/2,dV/2]
    voltages=np.zeros(len(atoms))
    voltages[atoms.get_array("mol-id")==1]=voltage[0]
    voltages[atoms.get_array("mol-id")==2]=voltage[1]
    atoms.set_array("voltage",voltages)
    atoms.calc=LAMMPS()
    


def AB_BA_neb(replicas,dV):
    AB=create_atoms([1/3,1/3,0],dV=0)
    BA=create_atoms([2/3,2/3,0],dV=0)
    #print(dipole_moment(AB))
    #print(dipole_moment(BA))
    optimizer=BFGS(AB,trajectory="AB_opt.traj", logfile="AB_opt.log")
    optimizer.run(fmax=0.001)

    optimizer=BFGS(BA,trajectory="BA_opt.traj",logfile="BA_opt.log")
    optimizer.run(fmax=0.001)

    AB=Trajectory('AB_opt.traj','r')[-1]
    BA=Trajectory('BA_opt.traj','r')[-1]

    to_model(AB,dV)
    to_model(BA,dV)

    #print(dipole_moment(AB))
    #print(dipole_moment(BA))

    images=[AB]
    images+=[AB.copy() for _ in range(replicas-2)]
    images+=[BA]
    for image in images:
        to_model(image,dV)
    # non-climbing image NEB
    neb=NEB(images,k=1.0,climb=True)
    neb.interpolate()
    optimizer=BFGS(neb, trajectory="neb.traj")
    optimizer.run(fmax=0.01)
    #print(len(Trajectory("neb.traj",'r')))
    mep=Trajectory("neb.traj",'r')[-replicas:]
    '''
    # climbing image NEB
    for i in range(replicas):
        atoms=mep[i]
        to_model(atoms,dV)
        images[i]=atoms
    neb=NEB(images, climb=True, k=1.0)
    optimizer=BFGS(neb, trajectory="neb_climb.traj")
    optimizer.run(fmax=0.001)
    '''
    mep=Trajectory("neb.traj",'r')[-replicas:]
    energies=np.zeros(replicas)
    dipoles=np.zeros(replicas)
    #print(len(mep))
    #print(replicas)
    for i in range(replicas):
        atoms=mep[i]
        to_model(atoms,dV)
        #mep[i].calc=LAMMPS()
        #print(mep[i].calc)
        energies[i]=atoms.get_potential_energy()
        dipoles[i]=dipole_moment(atoms)[2]
    '''
    print(atoms.positions)
    print(BA.positions)
    print(dipole_moment(BA))
    print(dipole_moment(atoms))
    print(BA.get_array("charges_model"))
    print(atoms.get_array("charges_model"))
    view(BA)
    view(atoms)
    '''
    return mep, energies, dipoles

dVs=-np.array([0, 0.3, 0.6, 0.9])*c
styles=['rx--', 'bx--', 'gx--', 'kx--']
for dV in dVs:
    print(f"Running NEB with dV={dV}")
    _, energies, dipoles = AB_BA_neb(replicas=20, dV=dV)
    #print(dipoles)
    np.savetxt(f"neb_{dV}.dat", np.array([energies, dipoles]).T)
f1,ax1=plt.subplots(1,1)
f2,ax2=plt.subplots(1,1)
for dV, style in zip(dVs, styles):
    print(f"Processing NEB with dV={dV}")
    #mep = Trajectory(f"neb_{dV}.traj", 'r')
    energies, dipoles = np.loadtxt(f"neb_{dV}.dat", unpack=True)
    
    energies -= energies[0]  # Normalize energies to the first step
    ax1.plot(energies,style,label=rf"$E_z \approx{-dV/c:.2}$",markersize=10)
    ax1.set_xlabel('NEB step')
    ax1.set_ylabel('Energy [eV]')
    ax1.grid(True)

    ax2.plot(dipoles, energies,style,label=rf"$E_z \approx{-dV/c:.2}$",markersize=10)
    ax2.set_xlabel('Dipole Moment [eÅ]')
    ax2.set_ylabel('Energy [eV]')
    ax2.grid(True)
ax1.legend()
ax1.set_title('ILP + charge exchange')

ax2.legend()
f1.tight_layout()
f2.tight_layout()
plt.show()