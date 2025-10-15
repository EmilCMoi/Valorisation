#from ase.calculators.kim.calculators import _check_conflict_options
#from ase.calculators.calculator import all_changes
#from ase.atoms import Atoms
#from scipy.spatial import cKDTree
from ase.neighborlist import NeighborList
from ase.neighborlist import NewPrimitiveNeighborList
import numpy as np
from numba.typed import Dict
from numba.core import types
import copy
from scipy.spatial import KDTree
#import os

# Version of the code finally adapted to LAMMPS and take into account carbon atoms
def born_charges(atoms=None):
        #0.05528761, 0.2507978 , 0.9990056
        # 0.3775918  -74.6910903    0.40010799
        #born_params={"B":{"B":{"q_infty":0.16724669},"N":{"C":-50.6228449,"beta": 0.33695933}},"N":{"B":{},"N":{}}}
        #born_params={"B":{"B":{"q_infty":0.3775918},"N":{"C":-74.6910903,"beta": 0.40010799}},"N":{"B":{},"N":{}}}
        born_params={"B":{"N":{"C":-8.11256518,"beta":0.56166857}},"N":{"B":{},"N":{}}}

        qs={"B":0.42,"N":-0.42, "C":0}  # Partial charges for LAMMPS, independent of the model

        # All parameters needed for the calculating the charges, it may be possible to reduce them
        # based on symmetries. Especially the q_N_0,i,p parameters
        # These parameters come from tight-binding calculations
        # Ideally one would like to get them from DFT calculations
        born_paramss=copy.deepcopy(born_params)
        #Symmetry constraints
        #born_paramss["N"]["N"]["q_infty"]=-copy.deepcopy(born_params["B"]["B"]["q_infty"])
        born_paramss["N"]["B"]["beta"]=copy.deepcopy(born_params["B"]["N"]["beta"])
        born_paramss["N"]["B"]["C"]=-copy.deepcopy(born_params["B"]["N"]["C"])
        pos=atoms.get_positions()
        cell=atoms.get_cell()
        na=len(pos)
        # dcharges is the difference between the calculated charge and the partial charges at infinite interlayer distance
        # it reflects the efect of local deformation on the polarization
        #cutoff=2.0
        #cutoff2=np.sqrt(3.225663**2+2*2.511**2/2)
        #nl=NeighborList(cutoffs=np.zeros(na)+cutoff/2,bothways=True,sorted=True)
        #nl2=NeighborList(cutoffs=np.zeros(na)+cutoff2/2,bothways=True)
        #nl.update(atoms)
        #nl2.update(atoms)
        #ns2=np.zeros((len(pos),3))
        # New version using NeighborList
        # Calculate the normal vectors for each atom
        '''
        for i in range(na):
            indices,offsets=nl.get_neighbors(i)
            gs=[]
            for j,offset in zip(indices,offsets): # Calculation of reciprocal lattice vectors
                if atoms.get_array("mol-id")[i] == atoms.get_array("mol-id")[j] and len(gs)<3 and atoms.symbols[i]!=atoms.symbols[j]:
                    r_ij=pos[j]-pos[i]+offset@cell
                    gs.append(r_ij)
            #print(gs)
            ns=[]
            for j in range(len(gs)): # Calculation of the normal
                n=np.cross(gs[j],gs[(j+1)%len(gs)])
                ns.append(n)
            ns=np.array(ns)
            #print(ns)
            n=ns[0]
            for j in range(len(ns)-1):
                if ns[0].dot(ns[j+1])<0:
                    ns[j+1]=-ns[j+1]
                n+=ns[j+1]
            n/=np.linalg.norm(n)
            ns2[i]=n
        '''
        rcut=8
        nlborn=NeighborList(cutoffs=np.zeros(na)+rcut/2,bothways=True,primitive=NewPrimitiveNeighborList)
        nlborn.update(atoms)
        dcharges=np.zeros(na)
        charges=np.zeros(na)
        charges_lammps=np.zeros(na)
        born = np.zeros((len(pos),3))
        
        for i in range(na):
            #charges[i]=born_paramss[atoms.symbols[i]][atoms.symbols[i]]["q_infty"]
            charges_lammps[i]=qs[atoms.symbols[i]]
            if atoms.get_array("mol-id")[i]==1:  
                indices,offsets=nlborn.get_neighbors(i)
                for j,offset in zip(indices,offsets):
                    if atoms.get_array("mol-id")[i] != atoms.get_array("mol-id")[j] and atoms.symbols[i]!=atoms.symbols[j] and atoms.symbols[i]!="C" and atoms.symbols[j]!="C":
                        '''
                        r=pos[j]-pos[i]+offset@cell
                        r_ij=np.linalg.norm(r)
                        Tap=20*(r_ij/rcut)**7 - 70*(r_ij/rcut)**6 + 84*(r_ij/rcut)**5 - 35*(r_ij/rcut)**4+1

                        # According to scipy analysis of the model, some of the parameters are redundant
                        rhosq_ij=r_ij**2-(r.dot(ns2[i]))**2
                        rhosq_ji=r_ij**2-(r.dot(ns2[j]))**2
                        C=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["C"]
                        d2=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["d2"]
                        epsilon=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["epsilon"]
                        beta=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["beta"]
                        
                        f_ij=C*np.exp(-rhosq_ij/d2)
                        f_ji=C*np.exp(-rhosq_ji/d2)
                        if beta==0:
                            print(born_params)
                        dcharges[i]+=Tap*(np.exp(-(r_ij/beta))*(epsilon+f_ij+f_ji))
                        dcharges[j]-=Tap*(np.exp(-(r_ij/beta))*(epsilon+f_ij+f_ji))

                        
                        # Derivative
                        dr_ij=r/r_ij
                        drhosq_ij=2*r-2*(np.diag(np.power(ns2[i],2)).dot(r))
                        drhosq_ji=2*r-2*(np.diag(np.power(ns2[j],2)).dot(r))
                        df_ij=-f_ij*drhosq_ij/d2
                        df_ji=-f_ji*drhosq_ji/d2

                        dTap=dr_ij*(7*20*(r_ij/rcut)**6-6*70*(r_ij/rcut)**5+5*84*(r_ij/rcut)**4-4*35*(r_ij/rcut)**3)
                        tmp=np.zeros(3)
                        tmp2=np.zeros(3)

                        tmp+=dTap*(np.exp(-(r_ij/beta))*(epsilon+f_ij+f_ji))
                        tmp+=Tap*(-np.exp(-(r_ij/beta))/beta*(epsilon+f_ij+f_ji))*dr_ij
                        tmp+=Tap*(np.exp(-(r_ij/beta))*(df_ij+df_ji))
                        tmp*=2 # This is because r_ij=-r_ji, as we don't calculate the second layer we add this here
                        # Reciprocity/Newton's third law
                        tmp2-=tmp

                        born[i]+=tmp
                        born[j]+=tmp2
                        '''
                        r=pos[j]-pos[i]+offset@cell
                        r_ij=np.linalg.norm(r)
                        Tap=20*(r_ij/rcut)**7 - 70*(r_ij/rcut)**6 + 84*(r_ij/rcut)**5 - 35*(r_ij/rcut)**4+1
                        #rhosq_ij=r_ij**2-(r.dot(ns2[i]))**2
                        #rhosq_ji=r_ij**2-(r.dot(ns2[j]))**2
                        C=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["C"]
                        #d2=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["d2"]
                        #epsilon=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["epsilon"]
                        beta=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["beta"]
                        
                        #f_ij=C*np.exp(-rhosq_ij/d2)
                        #f_ji=C*np.exp(-rhosq_ji/d2)
                        if beta==0:
                            print(born_params)
                        #dcharges[i]+=Tap*(np.exp(-(r_ij/beta))*(epsilon+f_ij+f_ji))
                        #dcharges[j]-=Tap*(np.exp(-(r_ij/beta))*(epsilon+f_ij+f_ji))
                        dcharges[i]+=Tap*(np.exp(-(r_ij/beta))*C)
                        dcharges[j]-=Tap*(np.exp(-(r_ij/beta))*C)
                        
                        # Derivative
                        dr_ij=r/r_ij
                        #drhosq_ij=2*r-2*(np.diag(np.power(ns2[i],2)).dot(r))
                        #drhosq_ji=2*r-2*(np.diag(np.power(ns2[j],2)).dot(r))
                        #df_ij=-f_ij*drhosq_ij/d2
                        #df_ji=-f_ji*drhosq_ji/d2

                        dTap=dr_ij*(7*20*(r_ij/rcut)**6-6*70*(r_ij/rcut)**5+5*84*(r_ij/rcut)**4-4*35*(r_ij/rcut)**3)
                        tmp=np.zeros(3)
                        tmp2=np.zeros(3)

                        tmp+=dTap*(np.exp(-(r_ij/beta))*C)
                        tmp+=Tap*(-np.exp(-(r_ij/beta))/beta*C)*dr_ij
                        #tmp+=Tap*(np.exp(-(r_ij/beta))*(df_ij+df_ji))
                        tmp*=2 # This is because r_ij=-r_ji, as we don't calculate the second layer we add this here
                        # Reciprocity/Newton's third law
                        tmp2=tmp

                        born[i]+=tmp
                        born[j]+=tmp2
                        


        charges+=dcharges
        
        return born, charges_lammps, charges

def dipole_moment(atoms=None):
        pos=atoms.get_positions()
       
        charges=atoms.get_array("charges_model")
        '''
        dipole_moment=np.zeros(3)
        
        for i in range(len(pos)):

            dipole_moment+=charges[i]*pos[i]
        '''
        return np.sum(pos.T * charges, axis=1)  # Return the dipole moment as a vector

def to_jit(atoms=None):
        # Convert the atoms object to a format that can be used in JIT compilation
        na=len(atoms)
        cutoff=2.0
        cutoff2=np.sqrt(3.225663**2+2*2.511**2/2)
        nl=NeighborList(cutoffs=np.zeros(na)+cutoff/2,bothways=True,sorted=True)
        nl2=NeighborList(cutoffs=np.zeros(na)+cutoff2/2,bothways=True,sorted=True)
        nl.update(atoms)
        nl2.update(atoms)
        # Dictionary of Born parameters
        born_params=Dict.empty(key_type=types.unicode_type, value_type=types.DictType(types.unicode_type,types.float64) )
        params=np.loadtxt("BORN",usecols=[1,2,3,4,5])
        born_names=["q_infty","q_0","q_p","q_i","d"]
        species=["B","N"]
        for i,s1 in enumerate(species):
            tmp_dict1=Dict.empty(key_type=types.unicode_type, value_type=types.float64)
            for k, param_name in enumerate(born_names):
                #tmp_dict1.update({param_name : params[i,k]})
                tmp_dict1[param_name] = params[i,k]
            #tmp_dict1.update({"dummy":0.42*(-1)**i})
            tmp_dict1["dummy"]=0.42*(-1)**i
            born_params[s1] = tmp_dict1
            #born_params.update({s1 : tmp_dict1})
        # One can calculate gs and ns here
        # I am not convinced this will be faster
        return atoms.get_positions(), atoms.get_cell(), atoms.get_array("mol-id"), atoms.symbols
'''
def born_charges_jit(positions, cell, mol-ids, symbols):
    
    return 0
'''

def KDneighborList(atoms=None, cutoff=8.0):
    # Create a neighbor list using KDTree
    pos = atoms.get_positions()
    v1= atoms.get_cell()[0]
    v2= atoms.get_cell()[1]
    pos_periodic=np.concatenate((pos-v1-v2,pos-v1,pos-v1+v2,pos-v2,pos,pos+v2,pos+v1-v2,pos+v1,pos+v1+v2),axis=0)
    tree_periodic = KDTree(pos_periodic)
    tree = KDTree(pos)
    neighbors = tree.query_ball_tree(tree_periodic, r=cutoff)
    return neighbors, pos_periodic

def KDborn(atoms=None):
        born_params={"B":{"N":{"C":-8.11256518,"beta":0.56166857}},"N":{"B":{},"N":{}}}

        qs={"B":0.42,"N":-0.42, "C":0}  # Partial charges for LAMMPS, independent of the model

        # All parameters needed for the calculating the charges, it may be possible to reduce them
        # based on symmetries. Especially the q_N_0,i,p parameters
        # These parameters come from tight-binding calculations
        # Ideally one would like to get them from DFT calculations
        born_paramss=copy.deepcopy(born_params)
        #Symmetry constraints
        #born_paramss["N"]["N"]["q_infty"]=-copy.deepcopy(born_params["B"]["B"]["q_infty"])
        born_paramss["N"]["B"]["beta"]=copy.deepcopy(born_params["B"]["N"]["beta"])
        born_paramss["N"]["B"]["C"]=-copy.deepcopy(born_params["B"]["N"]["C"])
        pos=atoms.get_positions()
        cell=atoms.get_cell()
        na=len(pos)
        # dcharges is the difference between the calculated charge and the partial charges at infinite interlayer distance
        # it reflects the efect of local deformation on the polarization

        rcut=8
        dcharges=np.zeros(na)
        charges=np.zeros(na)
        charges_lammps=np.zeros(na)
        born = np.zeros((len(pos),3))
        bornNeighs,pos_periodic=KDneighborList(atoms, cutoff=rcut)
        #print(len(bornNeighs))
        #print(len(bornNeighs[0]))
        #print(na)
        
        #offsets=[[-1,-1,0],[-1,0,0],[-1,1,0],[0,-1,0],[0,0,0],[0,1,0],[1,-1,0],[1,0,0],[1,1,0]]
        for i in range(na):
            charges_lammps[i]=qs[atoms.symbols[i]]
            if atoms.get_array("mol-id")[i]==1:  
                indices=bornNeighs[i]
                for l in indices:
                    j=l%na
                    #offset=offsets[round(j/na)]
                    #print(offset@cell)
                    if atoms.get_array("mol-id")[i] != atoms.get_array("mol-id")[j] and atoms.symbols[i]!=atoms.symbols[j] and atoms.symbols[i]!="C" and atoms.symbols[j]!="C":
                        r=pos_periodic[l]-pos[i]
                        #print(r)
                        assert np.linalg.norm(r) < rcut, "Distance between atoms %d and %d is larger than rcut: %f" % (i,j, np.linalg.norm(r))
                        r_ij=np.linalg.norm(r)
                        Tap=20*(r_ij/rcut)**7 - 70*(r_ij/rcut)**6 + 84*(r_ij/rcut)**5 - 35*(r_ij/rcut)**4+1
                        C=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["C"]
                        beta=born_paramss[atoms.symbols[i]][atoms.symbols[j]]["beta"]
                        
          
                        if beta==0:
                            print(born_params)
                        dcharges[i]+=Tap*(np.exp(-(r_ij/beta))*C)
                        dcharges[j]-=Tap*(np.exp(-(r_ij/beta))*C)
                        
                        # Derivative
                        dr_ij=r/r_ij
                    

                        dTap=dr_ij*(7*20*(r_ij/rcut)**6-6*70*(r_ij/rcut)**5+5*84*(r_ij/rcut)**4-4*35*(r_ij/rcut)**3)
                        tmp=np.zeros(3)
                        tmp2=np.zeros(3)

                        tmp+=dTap*(np.exp(-(r_ij/beta))*C)
                        tmp+=Tap*(-np.exp(-(r_ij/beta))/beta*C)*dr_ij
                        tmp*=2 # This is because r_ij=-r_ji, as we don't calculate the second layer we add this here
                        tmp2=tmp

                        born[i]-=tmp
                        born[j]-=tmp2
                        


        charges+=dcharges
        
        return born, charges_lammps, charges