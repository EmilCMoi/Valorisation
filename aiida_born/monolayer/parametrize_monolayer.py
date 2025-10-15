import numpy as np
import matplotlib.pyplot as plt
import copy
from ase.neighborlist import NeighborList, PrimitiveNeighborList
from ase import Atoms
from scipy.optimize import curve_fit, minimize
from tqdm import trange
#from draw import cmaps
from scipy.spatial import KDTree
from mpi4py import MPI
from multiprocessing import Pool, Process, Queue
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['figure.constrained_layout.use'] = True

# I'll try to parametrize charges and compare the resulting BECs with outputs from the PhCalculation
# Relevant equations in my MSc thesis: 23, 36

# I'm adding terms related to intralayer charge exchange, this is absent in my MSc thesis

#Lflep, Lflep_r, Dflep, Dflep_r = cmaps()

# Excluding this for now (perfectly flat layers)
def KDnormals(atoms=None):
    # Create a neighbor list using KDTree
    pos = atoms.get_positions()
    v1= atoms.get_cell()[0]
    v2= atoms.get_cell()[1]
    pos_periodic=np.concatenate((pos-v1-v2,pos-v1,pos-v1+v2,pos-v2,pos,pos+v2,pos+v1-v2,pos+v1,pos+v1+v2),axis=0)
    tree_periodic = KDTree(pos_periodic)
    tree = KDTree(pos)
    
    normals = np.zeros((len(pos), 3))
    # For now, 2 layers are implemented
    #tree_layer1=KDTree(pos[atoms.get_array("mol-id")==1])
    #tree_layer2=KDTree(pos[atoms.get_array("mol-id")==2])
    mol_periodic=np.repeat(atoms.get_array("mol-id"), 9)
    tree_periodic_layer1=KDTree(pos_periodic[mol_periodic==1])
    tree_periodic_layer2=KDTree(pos_periodic[mol_periodic==2])

    neighs_layer1 = tree_periodic_layer1.query(pos, k=4)
    neighs_layer2 = tree_periodic_layer2.query(pos, k=4)

    for i in range(len(atoms)):
        if atoms.get_array("mol-id")[i] == 1:
            # Get the neighbors for the first layer
            indices = neighs_layer1[1][i]
            # Calculate the normal vector as the average of the positions of the neighbors
            rs= pos_periodic[indices[1:]] - pos[i]
            n1=np.cross(rs[0], rs[1])
            n2= np.cross(rs[1], rs[2])
            n3= np.cross(rs[2], rs[0])

            n1/= np.linalg.norm(n1)
            n2/= np.linalg.norm(n2)
            n3/= np.linalg.norm(n3)

            index2=neighs_layer2[1][i][0]
            dz=pos_periodic[index2]-pos[i]

            # Ensuring correct orientation
            n1*= np.power(-1,int(dz.dot(n1)<0))
            n2*= np.power(-1,int(dz.dot(n2)<0))
            n3*= np.power(-1,int(dz.dot(n3)<0))
            normals[i] = (n1 + n2 + n3) / 3

        elif atoms.get_array("mol-id")[i] == 2:
            # Get the neighbors for the second layer
            indices = neighs_layer2[1][i]
            # Calculate the normal vector as the average of the positions of the neighbors
            rs= pos_periodic[indices] - pos[i]
            n1=np.cross(rs[0], rs[1])
            n2= np.cross(rs[1], rs[2])
            n3= np.cross(rs[2], rs[0])

            n1/= np.linalg.norm(n1)
            n2/= np.linalg.norm(n2)
            n3/= np.linalg.norm(n3)

            index1=neighs_layer1[1][i][0]
            dz=pos_periodic[index1]-pos[i]

            # Ensuring correct orientation
            n1*= np.power(-1,int(dz.dot(n1)<0))
            n2*= np.power(-1,int(dz.dot(n2)<0))
            n3*= np.power(-1,int(dz.dot(n3)<0))
            normals[i] = (n1 + n2 + n3) / 3

    return normals

def KDnn(atoms=None):
    # Create an intralayer nearest neighbor list using KDTree
    pos = atoms.get_positions()
    v1= atoms.get_cell()[0]
    v2= atoms.get_cell()[1]

    pos_periodic=np.concatenate((pos-v1-v2,pos-v1,pos-v1+v2,pos-v2,pos,pos+v2,pos+v1-v2,pos+v1,pos+v1+v2),axis=0)
    
    
    mol_periodic=np.repeat(atoms.get_array("mol-id"), 9)
    sym_periodic=np.repeat(atoms.symbols, 9)

    # Separating like this is probably heavier but ensures correctness
    neighs_dists=np.zeros((len(atoms), 3,3))
    neighs_indices=np.zeros((len(atoms), 3),dtype=int)
    #print(mol_periodic==1)
    #print(sym_periodic=='B')
    #print(((mol_periodic==1) & (sym_periodic=="B")))
    tree_periodic_B1=KDTree(pos_periodic[(mol_periodic==1) & (sym_periodic=="N")])
    tree_periodic_B2=KDTree(pos_periodic[(mol_periodic==2) & (sym_periodic=="N")])
    tree_periodic_N1=KDTree(pos_periodic[(mol_periodic==1) & (sym_periodic=="B")])
    tree_periodic_N2=KDTree(pos_periodic[(mol_periodic==2) & (sym_periodic=="B")])
    #print(pos)
    #print(atoms.symbols=='N')
    #print(atoms.get_array("mol-id")==2)
    #print((atoms.get_array("mol-id")==2) & (atoms.symbols=="N"))
    #print(len(pos_periodic))
    #print(sym_periodic)
    #print(mol_periodic)
    #print(pos[(atoms.get_array("mol-id")==2) & (atoms.symbols=="N")])
    _,neighs_B1 = tree_periodic_B1.query(pos[(atoms.get_array("mol-id")==1) & (atoms.symbols=="B")], k=3)
    _,neighs_B2 = tree_periodic_B2.query(pos[(atoms.get_array("mol-id")==2) & (atoms.symbols=="B")], k=3)
    _,neighs_N1= tree_periodic_N1.query(pos[(atoms.get_array("mol-id")==1) & (atoms.symbols=="N")], k=3)
    _,neighs_N2= tree_periodic_N2.query(pos[(atoms.get_array("mol-id")==2) & (atoms.symbols=="N")], k=3)

    # Put everything together in the order of atoms
    # This is a mess, a more efficient implementation will not necessarily need this
    countB1=0
    countB2=0
    countN1=0
    countN2=0
    for i in range(len(atoms)):
        for j in range(len(pos_periodic)):
            #print(np.sum(((mol_periodic==1) & (sym_periodic=="B"))[:j]))
            indexB1=np.sum(((mol_periodic==1) & (sym_periodic=="B"))[:j])
            indexB2=np.sum(((mol_periodic==2) & (sym_periodic=="B"))[:j])
            indexN1=np.sum(((mol_periodic==1) & (sym_periodic=="N"))[:j])
            indexN2=np.sum(((mol_periodic==2) & (sym_periodic=="N"))[:j])
            #print(neighs_N1,neighs_B1,neighs_N2,neighs_B2)

            if atoms.get_array("mol-id")[i] == 1 and atoms.symbols[i]=="B" and countB1<np.sum((atoms.symbols=='B')&(atoms.get_array("mol-id")==1)) and indexN1 in neighs_B1[countB1]:
                neighs_dists[i]=pos_periodic[j]-pos[i]
                neighs_indices[i]=j//9
                countB1+=1
                print(j)
            elif atoms.get_array("mol-id")[i] == 2 and atoms.symbols[i]=="B" and countB2<np.sum((atoms.symbols=='B')&(atoms.get_array("mol-id")==2)) and indexN2 in neighs_B2[countB2]:
                neighs_dists[i]=pos_periodic[j]-pos[i]
                neighs_indices[i]=j//9
                countB2+=1
            elif atoms.get_array("mol-id")[i] == 1 and atoms.symbols[i]=="N" and countN1<np.sum((atoms.symbols=='N')&(atoms.get_array("mol-id")==1)) and indexB1 in neighs_N1[countN1]:
                neighs_dists[i]=pos_periodic[j]-pos[i]
                neighs_indices[i]=j//9
                countN1+=1
            elif atoms.get_array("mol-id")[i] == 2 and atoms.symbols[i]=="N" and countN2<np.sum((atoms.symbols=='N')&(atoms.get_array("mol-id")==2)) and indexB2 in neighs_N2[countN2]:
                neighs_dists[i]=pos_periodic[j]-pos[i]
                neighs_indices[i]=j//9
                print(j)
                countN2+=1

    print(neighs_indices)
    return neighs_dists, neighs_indices

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

# Reading data
structs=np.load("borns.npz",allow_pickle=True)['structs']

a = 2.511
c = 6.6612 / 2
v1 = a*np.array([1,0,0])
v2 = a*np.array([-1/2,np.sqrt(3)/2,0])

print(structs[0,0])
borns=np.load("borns.npz")['borns']
print(len(structs))
print(len(borns))
structs_good=[]
borns_good=np.array([])
print(np.shape(borns))
for i in range(len(structs)):
    # Extract BECs
    borns_good=np.concatenate((borns_good, np.ravel(borns[i])))
    # Turn structures into ASE objects
    #print(structs[i,0].position)
    struct=Atoms(cell=[v1, v2, [0, 0, 30]])
    struct.set_pbc([True,True,False])
    struct.append(structs[i,0])
    struct.append( structs[i,1])
    struct.set_array("mol-id", [1, 1],dtype=int)  # Assigning molecule IDs
    structs_good.append(struct)
    
    #print(struct.positions)
from ase.visualize import view
view(structs_good[0])
def born_parametrized(atoms,q0_sp,q1_sp,q2_sp,d_sp,Z):
    """
    Calculate registry-dependent BECs and charges for a given structure.
    """
    
    # Initializing
    na=len(atoms)
    charges=np.zeros(na)
    born=np.zeros((na, 3, 3))
    #Z=0
    born_params={"B":{"N":{'q0_sp':q0_sp,'q1_sp':q1_sp,'q2_sp':q2_sp,'d_sp':1/d_sp},'Z':Z} ,"N":{"B":{}, 'Z':-Z}}
    born_paramss=copy.deepcopy(born_params)
    born_paramss["N"]["B"]["q0_sp"]=-q0_sp
    born_paramss["N"]["B"]["q1_sp"]=-q1_sp
    born_paramss["N"]["B"]["q2_sp"]=-q2_sp
    born_paramss["N"]["B"]["d_sp"]=1/d_sp
    pos=atoms.get_positions()
    cell=atoms.get_cell()
    rcut=5.0
    # Setting up neighbor list
    nlborn=NeighborList(cutoffs=np.zeros(na)+rcut/2,bothways=True,primitive=PrimitiveNeighborList)
    nlborn.update(atoms)

    #neighs_dists, neighs_indices = KDnn(atoms)
    #normals=KDnormals(atoms)
    
    for i in range(na):
        charges[i]+=born_paramss[atoms.symbols[i]]['Z']  # Initializing charges with q0
        # Intralayer contributions
        if atoms.symbols[i]=="B":
            indices, offsets = nlborn.get_neighbors(i)
            for j, offset in zip(indices, offsets):
                if (atoms.get_array("mol-id")[j]==atoms.get_array("mol-id")[i]) and atoms.symbols[j]=="N":
                    #print("Intralayer B-N", i, j)
                    r=pos[j]-pos[i]+offset@cell
                    r_ij=np.linalg.norm(r)
                    q0=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["q0_sp"]
                    q1=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["q1_sp"]
                    q2=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["q2_sp"]
                    d0=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["d_sp"]
                    #Tap=20*(r_ij/rcut)**7 - 70*(r_ij/rcut)**6 + 84*(r_ij/rcut)**5 - 35*(r_ij/rcut)**4+1
                    Tap=1
                    charges[i]+=Tap*np.exp(-(r_ij**2)/d0)*(q2*r_ij**2+q1*r_ij+q0)
                    charges[j]-=Tap*np.exp(-(r_ij**2)/d0)*(q2*r_ij**2+q1*r_ij+q0)


    # The following should adhere to the mathematical expression in my thesis (eq. 36) by explicitly inserting eq. 23
    
    for kappa in range(na):
        # Intralayer contributions
        for i in range(3):
            for j in range(3):
                sum_lambda=0
                for lambd in range(na):
                    sum_sigma=0
                    sigmas, offsets = nlborn.get_neighbors(lambd)
                    for sigma,offset in zip(sigmas, offsets):    
                        if (atoms.symbols[sigma]!=atoms.symbols[lambd]) and (atoms.get_array("mol-id")[sigma]==atoms.get_array("mol-id")[lambd]) and (lambd==kappa or sigma==kappa):
                            assert int(lambd==kappa)+int(sigma==kappa)<2
                            #print("Intralayer", lambd, sigma)
                            r_lambda_sigma=np.linalg.norm(pos[sigma]-pos[lambd]+offset@cell)#-offlambd@cell)
                            #Tap=20*(r_lambda_sigma/rcut)**7 - 70*(r_lambda_sigma/rcut)**6 + 84*(r_lambda_sigma/rcut)**5 - 35*(r_lambda_sigma/rcut)**4+1
                            #dTap=(7*20*(r_lambda_sigma/rcut)**6-6*70*(r_lambda_sigma/rcut)**5+5*84*(r_lambda_sigma/rcut)**4-4*35*(r_lambda_sigma/rcut)**3)/rcut
                            Tap=1
                            dTap=0
                            q0=born_paramss[atoms.symbols[lambd]][atoms.symbols[sigma]]["q0_sp"]
                            q1=born_paramss[atoms.symbols[lambd]][atoms.symbols[sigma]]["q1_sp"]
                            q2=born_paramss[atoms.symbols[lambd]][atoms.symbols[sigma]]["q2_sp"]
                            d0=born_paramss[atoms.symbols[lambd]][atoms.symbols[sigma]]["d_sp"]
                            #Tap*np.exp(-(r_ij)/d0)*(q2*r_ij**2+q1*r_ij+q0)
                            #sum_sigma+=np.power(-1,kappa==lambd)*q0*(Tap*np.exp(-(r_lambda_sigma)/d0)/d0/r_lambda_sigma-Tap*np.exp(-(r_lambda_sigma)/d0)/r_lambda_sigma**2)*(pos[sigma,j]-pos[lambd,j]+offset@cell[j])/ r_lambda_sigma
                            #np.power(-1,kappa==lambd)*
                            sum_sigma+=((2*q2*r_lambda_sigma+q1)*Tap*np.exp(-(r_lambda_sigma**2)/d0) - r_lambda_sigma*(q2*r_lambda_sigma**2+q1*r_lambda_sigma+q0)*Tap*np.exp(-(r_lambda_sigma**2)/d0)/d0)*(pos[sigma,j]-pos[lambd,j]+offset@cell[j])/ r_lambda_sigma
                    sum_lambda+=sum_sigma*(pos[lambd,i])#+offlambd@cell[i])
                born[kappa,i,j]+=sum_lambda#copy.copy(sum_lambda)
    
    #print(born[:,2,1])
    # Add partial charges to the diagonal elements
    for i in range(na):
        for j in range(2):
            born[i,j,j]+=charges[i]#born_paramss[atoms.symbols[i]]['Zi']
        #born[:,2,2]+=born_paramss[atoms.symbols[i]]['Zo']

    # Verify acoustic sum rules
    
    asr=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            asr[i,j]=np.sum(born[:,i,j])
            born[:,i,j]-=asr[i,j]/na
    
    #print("Acoustic sum rule: ", asr)
    #print("Charges: ", charges, np.sum(charges))
    return born
'''
def born_parametrized(atoms,q0_sp,q1_sp,q2_sp,d_sp):
    """
    Calculate registry-dependent BECs and charges for a given structure.
    """
    
    # Initializing
    na=len(atoms)
    charges=np.zeros(na)
    born=np.zeros((na, 3, 3))
    Z=0
    born_params={"B":{"N":{'q0_sp':q0_sp,'q1_sp':q1_sp,'q2_sp':q2_sp,'d_sp':1/d_sp},'Z':Z} ,"N":{"B":{}, 'Z':-Z}}
    born_paramss=copy.deepcopy(born_params)
    born_paramss["N"]["B"]["q0_sp"]=-q0_sp
    born_paramss["N"]["B"]["q1_sp"]=-q1_sp
    born_paramss["N"]["B"]["q2_sp"]=-q2_sp
    born_paramss["N"]["B"]["d_sp"]=1/d_sp
    pos=atoms.get_positions()
    cell=atoms.get_cell()
    rcut=8.0
    # Setting up neighbor list
    nlborn=NeighborList(cutoffs=np.zeros(na)+rcut/2,bothways=True,primitive=PrimitiveNeighborList)
    nlborn.update(atoms)

    #neighs_dists, neighs_indices = KDnn(atoms)
    #normals=KDnormals(atoms)
    
    for i in range(na):
        charges[i]+=born_paramss[atoms.symbols[i]]['Z']  # Initializing charges with q0
        # Intralayer contributions
        if atoms.symbols[i]=="B":
            indices, offsets = nlborn.get_neighbors(i)
            for j, offset in zip(indices, offsets):
                if (atoms.get_array("mol-id")[j]==atoms.get_array("mol-id")[i]) and atoms.symbols[j]=="N":
                    #print("Intralayer B-N", i, j)
                    r=pos[j]-pos[i]+offset@cell
                    r_ij=np.linalg.norm(r)
                    q0=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["q0_sp"]
                    q1=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["q1_sp"]
                    q2=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["q2_sp"]
                    d0=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["d_sp"]
                    #Tap=20*(r_ij/rcut)**7 - 70*(r_ij/rcut)**6 + 84*(r_ij/rcut)**5 - 35*(r_ij/rcut)**4+1
                    Tap=1
                    charges[i]+=Tap*np.exp(-(r_ij)/d0)*(q2*r_ij**2+q1*r_ij+q0)
                    charges[j]-=Tap*np.exp(-(r_ij)/d0)*(q2*r_ij**2+q1*r_ij+q0)


    # The following should adhere to the mathematical expression in my thesis (eq. 36) by explicitly inserting eq. 23
    
    for kappa in range(na):
        # Intralayer contributions
        for i in range(3):
            for j in range(3):
                sum_lambda=0
                for lambd in range(na):
                    sum_sigma=0
                    sigmas, offsets = nlborn.get_neighbors(lambd)
                    for sigma,offset in zip(sigmas, offsets):    
                        if (atoms.symbols[sigma]!=atoms.symbols[lambd]) and (atoms.get_array("mol-id")[sigma]==atoms.get_array("mol-id")[lambd]) and (lambd==kappa or sigma==kappa):
                            assert int(lambd==kappa)+int(sigma==kappa)<2
                            #print("Intralayer", lambd, sigma)
                            r_lambda_sigma=np.linalg.norm(pos[sigma]-pos[lambd]+offset@cell)#-offlambd@cell)
                            #Tap=20*(r_lambda_sigma/rcut)**7 - 70*(r_lambda_sigma/rcut)**6 + 84*(r_lambda_sigma/rcut)**5 - 35*(r_lambda_sigma/rcut)**4+1
                            #dTap=(7*20*(r_lambda_sigma/rcut)**6-6*70*(r_lambda_sigma/rcut)**5+5*84*(r_lambda_sigma/rcut)**4-4*35*(r_lambda_sigma/rcut)**3)/rcut
                            Tap=1
                            dTap=0
                            q0=born_paramss[atoms.symbols[lambd]][atoms.symbols[sigma]]["q0_sp"]
                            q1=born_paramss[atoms.symbols[lambd]][atoms.symbols[sigma]]["q1_sp"]
                            q2=born_paramss[atoms.symbols[lambd]][atoms.symbols[sigma]]["q2_sp"]
                            d0=born_paramss[atoms.symbols[lambd]][atoms.symbols[sigma]]["d_sp"]
                            #Tap*np.exp(-(r_ij)/d0)*(q2*r_ij**2+q1*r_ij+q0)
                            #sum_sigma+=np.power(-1,kappa==lambd)*q0*(Tap*np.exp(-(r_lambda_sigma)/d0)/d0/r_lambda_sigma-Tap*np.exp(-(r_lambda_sigma)/d0)/r_lambda_sigma**2)*(pos[sigma,j]-pos[lambd,j]+offset@cell[j])/ r_lambda_sigma
                            sum_sigma+=np.power(-1,kappa==lambd)*((2*q2*r_lambda_sigma+q1)*Tap*np.exp(-(r_lambda_sigma)/d0) - (q2*r_lambda_sigma**2+q1*r_lambda_sigma+q0)*Tap*np.exp(-(r_lambda_sigma)/d0)/d0)*(pos[sigma,j]-pos[lambd,j]+offset@cell[j])/ r_lambda_sigma
                    sum_lambda+=sum_sigma*(pos[lambd,i])#+offlambd@cell[i])
                born[kappa,i,j]+=sum_lambda#copy.copy(sum_lambda)
    
    # Add partial charges to the diagonal elements
    for i in range(na):
        for j in range(3):
            born[i,j,j]+=charges[i]#born_paramss[atoms.symbols[i]]['Zi']
        #born[:,2,2]+=born_paramss[atoms.symbols[i]]['Zo']

    # Verify acoustic sum rules
    
    asr=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            asr[i,j]=np.sum(born[:,i,j])
            born[:,i,j]-=asr[i,j]/na
    
    #print("Acoustic sum rule: ", asr)
    #print("Charges: ", charges, np.sum(charges))
    return born
'''
#7.256338653090905 [-4.153344000000001, 3.0747330960854087, -4.153344000000001, 2.1352313167259784]
dummyy=range(len(borns_good))
print(len(dummyy))
print(len(borns_good))
print(len(structs))
def fit(dummy,q0_sp,q1_sp,q2_sp,d_sp,Z):
    """
    Fit function for curve fitting.
    We trick scipy into thinking the first argument does anything
    """
    born_out=np.array([])
    for struct in structs_good:
        #struct.wrap()
        #born = np.ravel(born_parametrized(struct, d0,a,b)[::2,2,:2])
        born = np.ravel(born_parametrized(struct,q0_sp,q1_sp,q2_sp,d_sp,Z))
        born_out=np.concatenate((born_out,born))
        x= (q0_sp,q1_sp,q2_sp,d_sp,Z)
    eps=1e-3
    print(np.sum(np.log(eps+np.abs(born_out-borns_good))-np.log(eps))/len(borns_good), x)
    return born_out

def to_min(x):
    born_out=np.array([])
    
    for struct in structs_good:
        struct.wrap()
        #born = np.ravel(born_parametrized(struct, d0,a,b)[::2,2,:2])
        born = np.ravel(born_parametrized(struct,*x))
        born_out=np.concatenate((born_out,born))
    eps=1e-3
    print(np.sum(np.log(eps+np.abs(born_out-borns_good))-np.log(eps))/len(borns_good), x)
    return np.sum(np.log(eps+np.abs(born_out-borns_good))-np.log(eps))/len(borns_good)

'''
def plot_guess(results):
    """
    Plot the guesses for the fit function.
    """
    
    bornss_calc=np.zeros((len(rs), 4, 3, 3))
    for i, struct in enumerate(structs):
        struct.wrap()
        #print(struct.pbc)
        born =born_parametrized(struct,*results)
        bornss_calc[i] = born

    for i in range(4):
        plt.figure()
        b=bornss_calc[:,i,2,:2] # 3,xy components
        #print(len(b))
        C=np.linalg.norm(b,axis=1)
        plt.quiver(rs[:,0],rs[:,1],b[:,0]/C,b[:,1]/C,C,cmap=Dflep,pivot='middle')
        plt.colorbar()
        #print("before")
        plt.axis('equal')

        plt.figure()
        plt.scatter(rs[:,0],rs[:,1],c=bornss_calc[:,i,2,2],cmap=Dflep,s=200)
        plt.colorbar()
        plt.axis('equal')
'''
'''
iters=10
dichotomy_steps=3
p0= [-8.112,  1/0.562, -8.112 , 1/0.562]

for it in range(iters):
    for args in trange(len(p0)):
        for step in range(dichotomy_steps):
            if step==0:
                vals=np.linspace(p0[args]*0.8,p0[args]*1.2,11)
            else:
                #assert minind>0
                if minind!=0 and minind!=len(vals)-1:
                    vals=np.linspace(vals[minind-1],vals[minind+1],11)
                else:
                    vals=np.linspace(p0[args]*0.8,p0[args]*1.2,11)
            epsilons=np.zeros(len(vals))
            for i,val in enumerate(vals):
                ptmp=copy.deepcopy(p0)
                ptmp[args]=val
                #ptmp.insert(0,0.0)
                epsilons[i]=to_min(ptmp)
            minind=np.argmin(epsilons)
            p0[args]=vals[minind]

        plt.figure()
        plt.plot(vals, epsilons, 'o-')
        plt.xlabel(f"Parameter {args}")
        plt.ylabel("Loss function")
        plt.grid(True)
        plt.title(f"Iteration {it}, parameter {args}")    
    print(f"New parameters: {p0}")
    plt.show()

'''


#q0_sp,d_sp,Z
#results=curve_fit(fit, dummyy, borns_good,maxfev=10000,bounds=[[-10,-10,-10,0],[10,10,10,10]])#bounds=([-10,0,-10,0], [0,10,0,10])
#results=minimize(to_min,x0=[0.0, 0.0, 0.0, 5.0],method='BFGS')
#xx=[2.96186407, -4.23496608,  0.86762365,  0.45902289,  4.36691014]
xx=[1.67274665 ,-2.42003986,  0.46991482,  0.46773559,  2.91282671]
#print(results)
#xx=results[0]
#xx=results.x
qs=[1]#np.linspace(5,15,11)
for q in qs:
    #xx=[-0.00685252, 1.07232855]
    #xx=[-0.444, q]
    print("Fitted parameters: ",xx)
    #print(results)
    plt.figure()
    plt.plot([min(borns_good), max(borns_good)], [min(borns_good), max(borns_good)], 'k--')
    plt.plot(borns_good,fit(dummyy, *xx), 'rx',markersize=10)
    plt.title(xx)
    plt.grid(True)
    plt.xlabel("DFT BECs [|e|]")
    plt.ylabel("Fitted BECs [|e|]")
plt.show()

'''
#q0_sp,r0_sp,d_sp,q0_pz,r0_pz,d_pz
#results=curve_fit(fit, dummyy, borns_good, bounds=([-10,0,-10,0,-10], [0,10,0,10,10]) ,maxfev=10000)#bounds=([-10,0,-10,0], [0,10,0,10])

results=minimize(to_min, [-2.47672731,  7.665274  , -4.45484642,  3.94347635,  1.886097], bounds=[(-10, 0), (0, 10), (-10, 0), (0, 10),(-10,10)],method='L-BFGS-B')


#xx=results[0]
xx=results.x
print(results)
print(fit(dummyy, *xx))

print("Fitted parameters: ",xx)
print(results)
plt.figure()
plt.plot([min(borns_good), max(borns_good)], [min(borns_good), max(borns_good)], 'k--')
plt.plot(borns_good,fit(dummyy, *xx), 'rx',markersize=10)
plt.grid(True)
plt.xlabel("DFT BECs [|e|]")
plt.ylabel("Fitted BECs [|e|]")

plot_guess(xx)

print(results)
print(xx)
plt.show()
'''