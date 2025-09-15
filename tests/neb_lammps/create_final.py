import numpy as np

# Writing the initial coordinates for the NEB calculation
'''
natoms=np.loadtxt('output_a.lammpstrj',skiprows=3,max_rows=1)
A=np.loadtxt('output_a.lammpstrj',skiprows=5,max_rows=3)
print(natoms)
datdat=["\n"]
datdat.append(f"{int(natoms)} atoms \n")
datdat.append("3 atom types \n")
datdat.append("\n")
datdat.append("\n")
print(A)
datdat.append(f"{A[0,0]} {A[0,1]} xlo xhi \n")
datdat.append(f"{A[1,0]} {A[0,1]*np.sin(np.pi/3)} ylo yhi \n")
datdat.append(f"{A[2,0]} {A[2,1]} zlo zhi \n")
datdat.append(f"{A[0,2]} {A[1,2]} {A[2,2]} xy xz yz \n ")
datdat.append("\n")
datdat.append("Masses \n")
datdat.append("\n")
datdat.append("1 10.811 \n")
datdat.append("2 14.0067 \n")
datdat.append("3 12.01 \n")
data=np.loadtxt('output_a.lammpstrj',skiprows=9,max_rows=int(natoms))
dX=A[0,1]
dY=A[1,1]
dZ=A[2,1]-A[2,0]

V1=np.array([dX,0,0])
V2=np.array([A[0,2],dY,0])
xy=A[0,2]
print(dZ)
data_clean=np.array([np.round(data[:,0]),data[:,2]*dX+data[:,3]*xy,data[:,3]*dX*np.sin(np.pi/3),data[:,4]*dZ+A[2,0]])
# count the number of carbon atoms
C_count = np.sum(data[:,1] == 3)
print(C_count)
natoms=int(natoms)
print(data_clean[0,1])
datdat.append("Atoms # full \n \n")
for i in range(len(data_clean[0])):
    if data_clean[0,i]>natoms-C_count:
        mol=3
        charge=0.0
    elif data_clean[0,i]<natoms-C_count:
        mol=2-data_clean[0,i]%2
        charge=0.42*(1.5-data[i,1])*2
    datdat.append(f"{int(data_clean[0][i])} {int(mol)} {int(data[i,1])} {charge} {data_clean[1][i]} {data_clean[2][i]} {data_clean[3][i]}\n")
with open('initial.lammps', 'w') as f:
    f.writelines(datdat)

from ase.io import read
from ase.visualize import view
atoms=read('initial.lammps', format='lammps-data')
view(atoms)
'''
# Writing the final coordinates for the NEB calculation
natoms=np.loadtxt('output_b.lammpstrj',skiprows=3,max_rows=1)
natoms=int(natoms)
print(natoms)
data=np.loadtxt('output_b.lammpstrj',skiprows=9,max_rows=int(natoms))
A=np.loadtxt('output_b.lammpstrj',skiprows=5,max_rows=3)
dX=A[0,1]
dY=A[0,1]*np.sin(np.pi/3)
dZ=A[2,1]-A[2,0]
xy=A[0,2]
print(A)
print(dZ)
data_clean=[np.round(data[:,0]),data[:,2]*dX+data[:,3]*xy,data[:,3]*dY,data[:,4]*dZ+A[2,0]]
datdat=[]
datdat.append(f"{int(natoms)}\n")
for i in range(natoms):
    datdat.append(f"{int(data_clean[0][i])} {data_clean[1][i]} {data_clean[2][i]} {data_clean[3][i]}\n")

with open('coords.final', 'w') as f:
    f.writelines(datdat)