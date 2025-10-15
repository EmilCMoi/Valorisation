import numpy as np
import matplotlib.pyplot as plt
import copy
from ase.neighborlist import NeighborList, PrimitiveNeighborList
from ase import Atoms
from scipy.optimize import curve_fit, minimize
from tqdm import trange
from draw import cmaps
from scipy.spatial import KDTree
from mpi4py import MPI
from multiprocessing import Pool, Process, Queue
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['figure.constrained_layout.use'] = True

# I'll try to parametrize charges and compare the resulting BECs with outputs from the PhCalculation
# Relevant equations in my MSc thesis: 23, 36
Lflep, Lflep_r, Dflep, Dflep_r = cmaps()

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

def calculate_born(atoms):
    """
    Calculate registry-dependent BECs and charges for a given structure.
    """
    # Initializing
    na=len(atoms)
    charges=np.zeros(na)
    born=np.zeros((na, 3, 3))
    # Setting up parameters
    #born_params={"B":{"N":{"C":-8.11256518,"beta":0.56166857}},"N":{"B":{},"N":{}}}
    born_params={"B":{"N":{"C":3.07823975e-4,"beta":3.82128285e+00}, 'z0': 2.7},"N":{"B":{},"N":{},'z0': -1.0}}
    born_paramss=copy.deepcopy(born_params)
    born_paramss["N"]["B"]["beta"]=copy.deepcopy(born_params["B"]["N"]["beta"])
    born_paramss["N"]["B"]["C"]=-copy.deepcopy(born_params["B"]["N"]["C"])
    pos=atoms.get_positions()
    cell=atoms.get_cell()
    rcut=8
    # Setting up neighbor list
    nlborn=NeighborList(cutoffs=np.zeros(na)+rcut/2,bothways=True,primitive=PrimitiveNeighborList)
    nlborn.update(atoms)
    
    for i in range(na):
        if atoms.get_array("mol-id")[i] == 1:
            indices, offsets = nlborn.get_neighbors(i)
            for j, offset in zip(indices, offsets):
                if atoms.get_array("mol-id")[j]!=1 and atoms.symbols[i]!= atoms.symbols[j] and atoms.symbols[i]!="C" and atoms.symbols[j]!="C":
                    r=pos[j]-pos[i]+offset@cell
                    r_ij=np.linalg.norm(r)
                    r_i=pos[i]
                    r_j=pos[j]+offset@cell
                    Tap=20*(r_ij/rcut)**7 - 70*(r_ij/rcut)**6 + 84*(r_ij/rcut)**5 - 35*(r_ij/rcut)**4+1
                    C=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["C"]
                    beta=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["beta"]

                    charges[i]+=Tap*(np.exp(-(r_ij/beta))*C)
                    charges[j]-=Tap*(np.exp(-(r_ij/beta))*C)
                    #print("Charges: ", charges)

    # The following should adhere to the mathematical expression in my thesis (eq. 36) by explicitly inserting eq. 23
    count=0
    for kappa in range(na):
        for i in range(3):
            for j in range(3):
                sum_lambda=0
                for lambd in range(na):
                    sum_sigma=0

                    sigmas, offsets = nlborn.get_neighbors(lambd)
                    for sigma,offset in zip(sigmas, offsets):
                        if atoms.symbols[sigma]!=atoms.symbols[lambd]:
                            r_lambda_sigma=np.linalg.norm(pos[sigma]-pos[lambd]+offset@cell)
                            q0=born_paramss[atoms.symbols[lambd]][atoms.symbols[sigma]]["C"]
                            d0=born_paramss[atoms.symbols[lambd]][atoms.symbols[sigma]]["beta"]
                            
                            Tap=20*(r_lambda_sigma/rcut)**7 - 70*(r_lambda_sigma/rcut)**6 + 84*(r_lambda_sigma/rcut)**5 - 35*(r_lambda_sigma/rcut)**4+1
                            dTap=(7*20*(r_lambda_sigma/rcut)**6-6*70*(r_lambda_sigma/rcut)**5+5*84*(r_lambda_sigma/rcut)**4-4*35*(r_lambda_sigma/rcut)**3)/rcut



                            sum_sigma+=np.power(-1,atoms.get_array("mol-id")[lambd]-1)*q0* \
                                (dTap*np.exp(-(r_lambda_sigma/d0))-Tap*np.exp(-(r_lambda_sigma/d0))/d0)*(int(lambd==kappa)+int(sigma==kappa))*pos[kappa,j]/ r_lambda_sigma* \
                                int(atoms.get_array("mol-id")[sigma]!=atoms.get_array("mol-id")[lambd]) # if sigma and lambda are not the same molecule and not the same type of atom
                          
                    sum_lambda+=sum_sigma*pos[lambd,i]
                born[kappa,i,j]=sum_lambda
    for i in range(na):
        for j in range(3):
            born[i,j,j]+=charges[i]
    #print("Number of calculations: ", count)
    #print(len(charges))
    #print(np.shape(charges))
    return charges, born

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

rs=np.load("borns.npz")['rs']
borns=np.load("borns.npz")['borns']
which_to_read=np.load("borns.npz")['which_to_read']

charges=np.zeros((len(rs), 4))
borns_calcs=np.zeros((len(rs), 4, 3, 3))
print(len(rs))
print(len(borns))
print(len(which_to_read))
structs=[]
borns_good=np.array([])
print(np.shape(borns))
for i in range(len(rs)):
    #if which_to_read[i]:
        struct=create_structure(np.concatenate((rs[i],[0])))
        charge, born_calc=calculate_born(struct)
        charges[i]=charge
        borns_calcs[i]=born_calc
        structs.append(struct)
        #borns_good=np.concatenate((borns_good, np.ravel(borns[i,::2,2,:2]/2+borns[i,1::2,2,:2]/2)))
        borns_good=np.concatenate((borns_good, np.ravel(borns[i])))
print(borns[0])
print(borns_calcs[0])
print(np.linalg.norm(borns-borns_calcs))

def born_parametrized(atoms,q0,p0,lambd,z0,k):#,dd,eps):
    """
    Calculate registry-dependent BECs and charges for a given structure.
    """
    #z0=0
    # Initializing
    na=len(atoms)
    charges=np.zeros(na)
    born=np.zeros((na, 3, 3))
    # Setting up parameters
    #z0=3
    #born_params={"B":{"N":{"C":C,"beta":beta},'z0':q0},"N":{"B":{},"N":{},'z0':-q0}}
    born_params={"B":{"N":{"lambd":lambd, "z0":z0,"p0":p0,"k":k},'q0':q0},"N":{"B":{},"N":{},'q0':-q0}}
    born_paramss=copy.deepcopy(born_params)
    born_paramss["N"]["B"]["lambd"]=lambd#copy.deepcopy(born_params["B"]["N"]["beta"])
    born_paramss["N"]["B"]["z0"]=z0
    born_paramss["N"]["B"]["k"]=k
    born_paramss["N"]["B"]["p0"]=-p0
    #-copy.deepcopy(born_params["B"]["N"]["C"])
    pos=atoms.get_positions()
    cell=atoms.get_cell()
    rcut=10
    # Setting up neighbor list
    nlborn=NeighborList(cutoffs=np.zeros(na)+rcut/2,bothways=True,primitive=PrimitiveNeighborList)
    nlborn.update(atoms)

    normals=KDnormals(atoms)
    
    for i in range(na):
        charges[i]=born_paramss[atoms.symbols[i]]['q0']  # Initializing charges with q0
        if atoms.get_array("mol-id")[i] == 1:
            indices, offsets = nlborn.get_neighbors(i)
            for j, offset in zip(indices, offsets):
                if atoms.get_array("mol-id")[j]!=1 and atoms.symbols[i]!= atoms.symbols[j] and atoms.symbols[i]!="C" and atoms.symbols[j]!="C":
                    r=pos[j]-pos[i]+offset@cell
                    r_ij=np.linalg.norm(r)
                    #r_i=pos[i]
                    #r_j=pos[j]+offset@cell
                    Tap=20*(r_ij/rcut)**7 - 70*(r_ij/rcut)**6 + 84*(r_ij/rcut)**5 - 35*(r_ij/rcut)**4+1
                    #Tap=1
                    z0=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["z0"]
                    p0=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["p0"]
                    lambd=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["lambd"]
                    k=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["k"]

                    rho_ij=r_ij**2-np.dot(r, normals[i])**2
                    rho_ji= r_ij**2-np.dot(r, normals[j])**2
                    '''
                    z_ij=np.dot(r, normals[i])
                    z_ji=np.dot(r, normals[j])
                    if z_ij<0:
                        z_ij*=-1
                    if z_ji<0:
                        z_ji*=-1
                    
                    z=(z_ij+z_ji)/2
                    '''
                    rll=(rho_ij+rho_ji)/2

                    charges[i]+=Tap*p0*np.exp(-lambd*(r_ij-z0))*np.exp(-k*rll)/r_ij
                    charges[j]-=Tap*p0*np.exp(-lambd*(r_ij-z0))*np.exp(-k*rll)/r_ij
                    #charges[i]+=Tap*((eps+np.exp(-(rho_ij/beta))+np.exp(-(rho_ji/beta)))*C)*np.exp(-np.linalg.norm(r)/dd)
                    #charges[j]-=Tap*((eps+np.exp(-(rho_ij/beta))+np.exp(-(rho_ji/beta)))*C)*np.exp(-np.linalg.norm(r)/dd)

                    '''
                    dr_ij=r/r_ij
                    dTap=dr_ij*(7*20*(r_ij/rcut)**6-6*70*(r_ij/rcut)**5+5*84*(r_ij/rcut)**4-4*35*(r_ij/rcut)**3)/rcut
                    tmp=np.zeros(3)
                    tmp2=np.zeros(3)

                    tmp+=dTap*(np.exp(-(r_ij/beta))*C)
                    tmp+=Tap*(-np.exp(-(r_ij/beta))/beta*C)*dr_ij
                        #tmp+=Tap*(np.exp(-(r_ij/beta))*(df_ij+df_ji))
                    tmp*=2 # This is because r_ij=-r_ji, as we don't calculate the second layer we add this here
                        # Reciprocity/Newton's third law
                    tmp2=tmp

                    born[i,2]+=tmp
                    born[j,2]+=tmp2
                    '''
                    #print("Charges: ", charges)
    
    # The following should adhere to the mathematical expression in my thesis (eq. 36) by explicitly inserting eq. 23
    
    '''
    for kappa in range(na):
        for i in range(3):
            for j in range(3):
                sum_lambda=0
                lambds,offlambds = nlborn.get_neighbors(kappa)
                for lambd,offlambd in zip(lambds,offlambds):
                    sum_sigma=0

                    sigmas, offsets = nlborn.get_neighbors(lambd)
                    for sigma,offset in zip(sigmas, offsets):
                        if atoms.symbols[sigma]!=atoms.symbols[lambd] and int(atoms.get_array("mol-id")[sigma]!=atoms.get_array("mol-id")[lambd]):
                            r_lambda_sigma=np.linalg.norm(pos[sigma]-pos[lambd]+offset@cell-offlambd@cell)
                            #assert r_lambda_sigma<rcut*1.2, f"{r_lambda_sigma}"
                            q0=born_paramss[atoms.symbols[lambd]][atoms.symbols[sigma]]["C"]
                            d0=born_paramss[atoms.symbols[lambd]][atoms.symbols[sigma]]["beta"]
                            
                            Tap=20*(r_lambda_sigma/rcut)**7 - 70*(r_lambda_sigma/rcut)**6 + 84*(r_lambda_sigma/rcut)**5 - 35*(r_lambda_sigma/rcut)**4+1
                            dTap=(7*20*(r_lambda_sigma/rcut)**6-6*70*(r_lambda_sigma/rcut)**5+5*84*(r_lambda_sigma/rcut)**4-4*35*(r_lambda_sigma/rcut)**3)/rcut

                            #print(np.power(-1,atoms.get_array("mol-id")[lambd]-1))
                            assert int(lambd==kappa)+int(sigma==kappa)<2
                            sum_sigma+=np.power(-1,atoms.get_array("mol-id")[lambd]-1)*q0* \
                                (dTap*np.exp(-(r_lambda_sigma/d0))-Tap*np.exp(-(r_lambda_sigma/d0))/d0)*(int(lambd==kappa)+int(sigma==kappa))*pos[kappa,j]/ r_lambda_sigma
                                # if sigma and lambda are not the same molecule and not the same type of atom
                            
                        
                    #print(sum_sigma)   
                    sum_lambda+=sum_sigma*(pos[lambd,i]+offlambd@cell[i])
                #print(sum_lambda)
                #if sum_lambda>1:
                #    print(lambd,kappa,i,j)
                born[kappa,i,j]=sum_lambda
    '''
    for kappa in range(na):
        for i in range(3):
            for j in range(3):
                sum_lambda=0
                #lambds,offlambds = nlborn.get_neighbors(kappa)

                
                for lambd in range(na):#lambd,offlambd in zip(lambds,offlambds):
                    sum_sigma=0

                    sigmas, offsets = nlborn.get_neighbors(lambd)
                    for sigma,offset in zip(sigmas, offsets):
                        if atoms.symbols[sigma]!=atoms.symbols[lambd] and atoms.get_array("mol-id")[sigma]!=atoms.get_array("mol-id")[lambd] and (lambd==kappa or sigma==kappa):
                            r_lambda_sigma=np.linalg.norm(pos[sigma]-pos[lambd]+offset@cell)#-offlambd@cell)
                            #assert r_lambda_sigma<rcut*1.2, f"{r_lambda_sigma}"
                            #q0=born_paramss[atoms.symbols[lambd]][atoms.symbols[sigma]]["C"]
                            #d0=born_paramss[atoms.symbols[lambd]][atoms.symbols[sigma]]["beta"]
                            
                            Tap=20*(r_lambda_sigma/rcut)**7 - 70*(r_lambda_sigma/rcut)**6 + 84*(r_lambda_sigma/rcut)**5 - 35*(r_lambda_sigma/rcut)**4+1
                            dTap=(7*20*(r_lambda_sigma/rcut)**6-6*70*(r_lambda_sigma/rcut)**5+5*84*(r_lambda_sigma/rcut)**4-4*35*(r_lambda_sigma/rcut)**3)/rcut
                            #Tap=1
                            #dTap=0
                            rho_lambda_sigma=r_lambda_sigma**2-np.dot(pos[sigma]-pos[lambd]+offset@cell, normals[lambd])**2
                            rho_sigma_lambda=r_lambda_sigma**2-np.dot(pos[sigma]-pos[lambd]+offset@cell, normals[sigma])**2
                            '''
                            z_lambda_sigma=np.dot(pos[sigma]-pos[lambd]+offset@cell, normals[lambd])
                            z_sigma_lambda=np.dot(pos[sigma]-pos[lambd]+offset@cell, normals[sigma])

                            dz_lambda_sigma=normals[lambd][j]
                            dz_sigma_lambda=normals[sigma][j]

                            if z_lambda_sigma<0:
                                dz_lambda_sigma*=-1
                                z_lambda_sigma*=-1
                            if z_sigma_lambda<0:
                                dz_sigma_lambda*=-1
                                z_sigma_lambda*=-1

                            z=(z_lambda_sigma+z_sigma_lambda)/2
                            dz=(dz_lambda_sigma+dz_sigma_lambda)/2
                            '''
                            # Note: j is the derivative direction
                            drho2_lambda_sigma=2*(pos[sigma]-pos[lambd]+offset@cell)[j]*(1-normals[lambd][j]**2)
                            drho2_sigma_lambda=2*(pos[sigma]-pos[lambd]+offset@cell)[j]*(1-normals[sigma][j]**2)
                            #print(np.power(-1,atoms.get_array("mol-id")[lambd]-1))
                            assert int(lambd==kappa)+int(sigma==kappa)<2
                            #sum_sigma+=np.power(-1,atoms.get_array("mol-id")[lambd]-1)*q0* \
                            #    (dTap*np.exp(-(r_lambda_sigma/d0))-Tap*np.exp(-(r_lambda_sigma/d0))/d0)*(pos[sigma]-pos[lambd]+offset@cell)[j]*(int(sigma==kappa)-int(kappa==lambd))/ r_lambda_sigma
                            z0=born_paramss[atoms.symbols[sigma]][atoms.symbols[lambd]]["z0"]
                            p0=born_paramss[atoms.symbols[sigma]][atoms.symbols[lambd]]["p0"]
                            lambdp=born_paramss[atoms.symbols[sigma]][atoms.symbols[lambd]]["lambd"]
                            k=born_paramss[atoms.symbols[sigma]][atoms.symbols[lambd]]["k"]

                            rll=(rho_lambda_sigma+rho_sigma_lambda)/2

                            '''
                            charges[i]+=Tap*p0*np.exp(-lambd*(r_ij-z0))*np.exp(-k*rll)
                            charges[j]-=Tap*p0*np.exp(-lambd*(r_ij-z0))*np.exp(-k*rll)
                            '''

                            sum_sigma+=dTap*np.exp(-lambdp*(r_lambda_sigma-z0))*np.exp(-k*rll)/r_lambda_sigma
                            sum_sigma+=-lambdp*Tap*np.exp(-lambdp*(r_lambda_sigma-z0))*np.exp(-k*rll)/r_lambda_sigma
                            
                            sum_sigma+=-Tap*np.exp(-lambdp*(r_lambda_sigma-z0))*np.exp(-k*rll)/r_lambda_sigma**2

                            sum_sigma/=r_lambda_sigma*(pos[sigma]-pos[lambd]+offset@cell)[j]

                            #sum_sigma+=Tap*np.exp(-lambdp*(r_lambda_sigma-z0))*np.exp(-k*rll)*dz/r_lambda_sigma

                            sum_sigma+=-k*Tap*np.exp(-lambdp*(r_lambda_sigma-z0))*np.exp(-k*rll)*(drho2_lambda_sigma+drho2_sigma_lambda)/2

                            sum_sigma*=np.power(-1,atoms.get_array("mol-id")[lambd]-1)*p0*(int(sigma==kappa)-int(kappa==lambd))
                            #sum_sigma+=np.power(-1,atoms.get_array("mol-id")[lambd]-1)*q0* \
                            #    (dTap*(np.exp(-(rho_lambda_sigma/d0))+np.exp(-(rho_sigma_lambda/d0)))*(pos[sigma]-pos[lambd]+offset@cell)[j]*(int(sigma==kappa)-int(kappa==lambd))/r_lambda_sigma \
                            #    +Tap*(-np.exp(-(rho_lambda_sigma/d0))/d0*drho2_lambda_sigma-np.exp(-(rho_sigma_lambda/d0))/d0*drho2_sigma_lambda))*(int(sigma==kappa)-int(kappa==lambd))
                            #sum_sigma+=np.power(-1,atoms.get_array("mol-id")[lambd]-1)*q0*( \
                            #-1/dd*np.exp(-np.linalg.norm(r)/dd)*(eps+np.exp(-(rho_lambda_sigma/d0)) + np.exp(-(rho_sigma_lambda/d0)))*(pos[sigma]-pos[lambd]+offset@cell)[j]*(int(sigma==kappa)-int(kappa==lambd))/r_lambda_sigma
                            #+np.exp(-np.linalg.norm(r)/dd)*(np.exp(-(rho_lambda_sigma/d0))/d0*drho2_lambda_sigma + np.exp(-(rho_sigma_lambda/d0))/d0*drho2_sigma_lambda)*(int(sigma==kappa)-int(kappa==lambd)))

                    #print(sum_sigma)
                    sum_lambda+=sum_sigma*(pos[lambd,i])#+offlambd@cell[i])
                    
                
                # Rewritten in a more compact and efficient way
                # First case, kappa == lambda
                '''
                sum_sigma=0
                sigmas, offsets = nlborn.get_neighbors(kappa)
                for sigma,offset in zip(sigmas, offsets):
                    if atoms.symbols[sigma]!=atoms.symbols[kappa] and atoms.get_array("mol-id")[sigma]!=atoms.get_array("mol-id")[kappa]:
                        r_lambda_sigma=np.linalg.norm(pos[sigma]-pos[kappa]+offset@cell)
                        q0=born_paramss[atoms.symbols[kappa]][atoms.symbols[sigma]]["C"]
                        d0=born_paramss[atoms.symbols[kappa]][atoms.symbols[sigma]]["beta"]
                        
                        #Tap=20*(r_lambda_sigma/rcut)**7 - 70*(r_lambda_sigma/rcut)**6 + 84*(r_lambda_sigma/rcut)**5 - 35*(r_lambda_sigma/rcut)**4+1
                        #dTap=(7*20*(r_lambda_sigma/rcut)**6-6*70*(r_lambda_sigma/rcut)**5+5*84*(r_lambda_sigma/rcut)**4-4*35*(r_lambda_sigma/rcut)**3)/rcut
                        Tap=1
                        dTap=0
                        sum_sigma+=np.power(-1,atoms.get_array("mol-id")[kappa]-1)*q0* \
                            (dTap*np.exp(-(r_lambda_sigma/d0))-Tap*np.exp(-(r_lambda_sigma/d0))/d0)*(pos[kappa,j]-pos[sigma,j]-offset@cell[j])/ r_lambda_sigma
                sum_lambda+=sum_sigma*pos[kappa,i]

                # Second case, kappa == sigma
                sum_sigma=0
                #lambds, offsets = nlborn.get_neighbors(kappa)
                #print(list(range(na)))
                lambds = [l for l in range(na) if l != kappa]
                #print(lambds)
                for lambd in lambds:#range(na)[range(na)!=kappa]:#lambd,offset in zip(lambds, offsets):
                    sigmas, offsets = nlborn.get_neighbors(lambd)
                    sigma=kappa
                    offsets=offsets[sigmas==kappa]
                    #print(sigmas==kappa)
                    #print(offsets)
                    for offset in offsets:
                        if atoms.symbols[lambd]!=atoms.symbols[kappa] and atoms.get_array("mol-id")[lambd]!=atoms.get_array("mol-id")[kappa]:
                            r_lambda_sigma=np.linalg.norm(pos[sigma]-pos[lambd]+offset@cell)
                            
                            q0=born_paramss[atoms.symbols[lambd]][atoms.symbols[sigma]]["C"]
                            d0=born_paramss[atoms.symbols[lambd]][atoms.symbols[sigma]]["beta"]
                            #Tap=20*(r_lambda_sigma/rcut)**7 - 70*(r_lambda_sigma/rcut)**6 + 84*(r_lambda_sigma/rcut)**5 - 35*(r_lambda_sigma/rcut)**4+1
                            #dTap=(7*20*(r_lambda_sigma/rcut)**6-6*70*(r_lambda_sigma/rcut)**5+5*84*(r_lambda_sigma/rcut)**4-4*35*(r_lambda_sigma/rcut)**3)/rcut
                            Tap=1
                            dTap=0
                            dr=(pos[sigma]-pos[lambd]+offset@cell)[j]/r_lambda_sigma
                            sum_sigma+=np.power(-1,atoms.get_array("mol-id")[lambd]-1)*q0* \
                                (dTap*np.exp(-(r_lambda_sigma/d0))-Tap*np.exp(-(r_lambda_sigma/d0))/d0)* dr
                    sum_lambda+=sum_sigma*(pos[lambd,i])
                '''

                '''
                    if atoms.symbols[lambd]!=atoms.symbols[kappa] and atoms.get_array("mol-id")[lambd]!=atoms.get_array("mol-id")[kappa]:
                        r_lambda_sigma=np.linalg.norm(pos[lambd]-pos[kappa])#+offset@cell)
                        q0=born_paramss[atoms.symbols[lambd]][atoms.symbols[lambd]]["C"]
                        d0=born_paramss[atoms.symbols[lambd]][atoms.symbols[lambd]]["beta"]
                        
                        Tap=20*(r_lambda_sigma/rcut)**7 - 70*(r_lambda_sigma/rcut)**6 + 84*(r_lambda_sigma/rcut)**5 - 35*(r_lambda_sigma/rcut)**4+1
                        dTap=(7*20*(r_lambda_sigma/rcut)**6-6*70*(r_lambda_sigma/rcut)**5+5*84*(r_lambda_sigma/rcut)**4-4*35*(r_lambda_sigma/rcut)**3)/rcut

                        sum_sigma=np.power(-1,atoms.get_array("mol-id")[lambd]-1)*q0* \
                            (dTap*np.exp(-(r_lambda_sigma/d0))-Tap*np.exp(-(r_lambda_sigma/d0))/d0)*pos[kappa,j]/ r_lambda_sigma
                        sum_lambda+=sum_sigma*(pos[lambd,i]+ offset@cell[i])
                '''
                #print(sum_lambda)
                #if sum_lambda>1:
                #    print(lambd,kappa,i,j)
                born[kappa,i,j]=sum_lambda
    
    '''
    for kappa in range(na):
        lambds, offsets = nlborn.get_neighbors(kappa)
        for lambd, offset in zip(lambds, offsets):
            if atoms.symbols[lambd]!=atoms.symbols[kappa] and atoms.get_array("mol-id")[lambd]!=atoms.get_array("mol-id")[kappa]:
                # Easier parametrization?
                r_norm=np.linalg.norm(pos[kappa]-pos[lambd]+offset@cell)
                r= pos[lambd]-pos[kappa]+offset@cell
                Tap=20*(r_norm/rcut)**7 - 70*(r_norm/rcut)**6 + 84*(r_norm/rcut)**5 - 35*(r_norm/rcut)**4+1
                Tap*=np.power(-1,atoms.get_array("mol-id")[kappa]-1+int(atoms.symbols[kappa]=="B"))
                born[kappa]+=np.power(-1,atoms.get_array("mol-id")[kappa]-1)*Tap*np.exp(-(r_norm/d0))*(a*np.array([[1, 0,0], [0, 1,0],[0, 0,1]]) + b*np.outer(r, r))# + c*np.outer(r, r)/r_norm**2)
                #born[kappa,2,0]+=Tap*np.exp(-(r_norm/d0))*(a + b*np.sin(c*r[0])+ d*np.cos(c*r[0]))
                #born[kappa,2,1]+=Tap*np.exp(-(r_norm/d0))*(a + b*np.sin(c*r[1])+ d*np.cos(c*r[1]))
    '''
    #print(len(charges))
    #print(np.shape(charges))
    for i in range(na):
        for j in range(3):
            born[i,j,j]+=charges[i]

    # Verify acoustic sum rules
    asr=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            asr[i,j]=np.sum(born[:,i,j])
            born[:,i,j]-=asr[i,j]/na
    print("Acoustic sum rule: ", asr)
    #print(np.sum(charges))       
    #print(born_paramss)
    return born#born[::2,2,:2]

dummyy=range(len(borns_good))
print(len(dummyy))
print(len(borns_good))
print(len(structs))
def fit(dummy, q0,p0,lambd,z0,k):
    """
    Fit function for curve fitting.
    We trick scipy into thinking the first argument does anything
    """
    born_out=np.array([])
    for struct in structs:
        struct.wrap()
        #born = np.ravel(born_parametrized(struct, d0,a,b)[::2,2,:2])
        born = np.ravel(born_parametrized(struct,q0,p0,lambd,z0,k))
        born_out=np.concatenate((born_out,born))
        x= (q0,p0,lambd,z0,k)
    eps=1e-4
    print(np.sum(np.log(eps+np.abs(born_out-borns_good))-np.log(eps))/len(borns_good), x)
    return born_out

def to_min(x):
    born_out=np.array([])
    #print(x)
    #a=[]
    #for struct in structs:
    #    a.append((struct,*x))
    #print(a)
    #print(a[0])
    #a=zip(structs,[x[0]],[x[1]],[x[2]],[x[3]],[x[4]])

    #p=Process(target=born_parametrized, args=structs,kwargs={'q0':x[0],'p0':x[1],'lambd':x[2],'z0':x[3],'k':x[4]})
    #p.start()
    #p.join()
    #print(p)
    for struct in structs:
        struct.wrap()
        #born = np.ravel(born_parametrized(struct, d0,a,b)[::2,2,:2])
        born = np.ravel(born_parametrized(struct,*x))
        born_out=np.concatenate((born_out,born))
    eps=1e-4
    print(np.sum(np.log(eps+np.abs(born_out-borns_good))-np.log(eps))/len(borns_good), x)
    return np.sum(np.log(eps+np.abs(born_out-borns_good))-np.log(eps))/len(borns_good)

#rs=rs[which_to_read]
def plot_guess(results):
    """
    Plot the guesses for the fit function.
    """
    
    bornss_calc=np.zeros((len(rs), 4, 3, 3))
    for i, struct in enumerate(structs):
        struct.wrap()
        #print(struct.pbc)
        born =born_parametrized(struct,*xx)
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

#,q0,p0,lambd,z0,k
#results=curve_fit(fit, dummyy, borns_good,maxfev=10000,p0=[2.7,1.61e-2,1.94,3.3,0.217])#,p0=[0.64875374, 1.58706567, 1.88588162, -1.60477028, 1.82270341, 0.45473178])
#with Pool(10) as pwl:
#p0= [2.7,-1/5.43*1.61e-2,1.94,3.3,0.217]

# Search algorithm : start with guess and look within 10% radius for better solution through dichotomy
# perform 3 steps of dichotomy, repeat for n iterations
iters=10
dichotomy_steps=3
#p0s=[2.70064,0.042,1.94,3.3,0.217]
p0= [0.042,2.52,2.984,0.212]
#vals=np.linspace(0.212,0.214,11)
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
                ptmp.insert(0,0.0)
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
results=minimize(to_min, [2.7,-1/5.43*1.61e-2,1.94,3.3,0.217],method='CG',options={'eps':1e-10})


#xx=results[0]
xx=results.x
print(results)
print(fit(dummyy, *xx))

print("Fitted parameters: ",xx)
print(results)
#3.795762257144396 -P0
#3.768413431956908 +P0

#results=np.array([0.005, 1])
plt.figure()
plt.plot([min(borns_good), max(borns_good)], [min(borns_good), max(borns_good)], 'k--')
plt.plot(borns_good,fit(dummyy, *xx), 'rx',markersize=10)
plt.grid(True)
plt.xlabel("DFT BECs [|e|]")
plt.ylabel("Fitted BECs [|e|]")
#plt.tight_layout()
#plt.plot(fit(dummyy, *results),borns_good, 'o')
plot_guess(results)
#plot_guess(-8,0.5,0.2)
#plot_guess(-8,0.5  , 2.7    )
print(results)
plt.show()
'''