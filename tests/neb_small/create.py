import numpy as np
from ase import Atoms

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

def write_lammps_data(filename, r):
    atoms = write_atoms(r)
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)} atoms\n")
        f.write("2 atom types\n\n")
        f.write(f"0.0 {Lx} xlo xhi\n")
        f.write(f"0.0 {Ly} ylo yhi\n")
        f.write(f"0.0 {c} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write("1 10.811\n")
        f.write("2 14.0067\n\n")
        f.write("Atoms # full\n\n")
        for i, atom in enumerate(atoms):
            mol_id = atom.get_array('mol-id')[0]
            charge = 0.42 * (1.5 - i % 2) * 2 if mol_id == 1 else 0.0
            f.write(f"{i + 1} {mol_id} {atom.symbols[i]} {charge} {atom.position[0]} {atom.position[1]} {atom.position[2]}\n")