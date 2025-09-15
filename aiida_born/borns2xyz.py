import numpy as np
from ase import Atoms


rs=np.load("borns.npz")['rs']
borns=np.load("borns.npz")['borns']

a = 2.511
c = 6.6612 / 2
v1 = a*np.array([1,0,0])
v2 = a*np.array([-1/2,np.sqrt(3)/2,0])
cell=[v1, v2, [0, 0, 30]]

def create_structure(r):
    a = 2.511
    c = 6.6612 / 2
    v1 = a*np.array([1,0,0])
    v2 = a*np.array([-1/2,np.sqrt(3)/2,0])
    
    # Adjusting for the convention of ibrav=4:
    tmp=r[0]
    r[0]=r[1]
    r[1]=-tmp
    # Create positions based on the input r
    positions = np.zeros((4, 3))
    positions[0] = np.array([0, 0, 0])
    positions[1] = np.array([0, 0, 0]) + v1 / 3 - v2 / 3
    positions[2] = np.array([0, 0, c]) + r[0] * v1 + r[1] * v2 + r[2]
    positions[3] = np.array([0, 0, c]) + v1 / 3 - v2 / 3 + r[0] * v1 + r[1] * v2 + r[2]
    
    labels = ["B", "N", "B", "N"]
    struct=Atoms(symbols=labels, positions=positions,cell=[v1, v2, [0, 0, 30]])
    struct.set_array("mol-id", [1, 1, 2, 2],dtype=int)  # Assigning molecule IDs
    struct.wrap()
    struct.center()
    struct.wrap()
    struct.pbc=[True,True,False]
    return struct
#bl=['a','b','c','d']
data=[]
for i in range(len(borns)):
    r=[rs[i][0],rs[i][1],0]
    struct=create_structure(r)
    data.append(str(len(struct))+"\n")
    data.append(f"Lattice=\"{' '.join(map(str,struct.get_cell().flatten()))}\" Properties=\"species:S:1:pos:R:3\" BornA=\"{' '.join(map(str,borns[i][0].ravel()))}\" BornB=\"{' '.join(map(str,borns[i][2].ravel()))}\" BornC=\"{' '.join(map(str,borns[i][1].ravel()))}\" BornD=\"{' '.join(map(str,np.ravel(borns[i][3])))}\" \n")
    for s, p, id in zip(struct.get_chemical_symbols(), struct.get_positions(), struct.get_array("mol-id")):
        data.append(f"{s} {' '.join(map(str,p))}\n")
print(data)
with open("train.xyz",'w') as f:
    f.writelines(data)