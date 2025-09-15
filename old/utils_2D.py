from model.build_2D import build_2D
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
import os
import numpy as np
from ase.io import read
from model.calculator import LAMMPS
from model.born import born_charges, dipole_moment
from tqdm import trange
from ase.io.trajectory import Trajectory
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.optimize import LBFGS
import matplotlib.pyplot as plt
import nglview as nv
from ase.optimize import BFGSLineSearch
from ase.filters import FrechetCellFilter
from subprocess import run
from scipy.optimize import curve_fit
from ase import Atoms
from ase.io import write
from ase.io.xsf import write_xsf
from scipy.spatial import KDTree
from ase.visualize import view
from tqdm import tqdm
plt.rcParams.update({'font.size': 17})
plt.rcParams.update({'font.family': 'sans-serif'})

def minimize_lammps(m,defo):
    build_2D(m=m,defo=defo,filename="tmp2D.lammps",verify=False)
    
    os.system("mpirun -np 12 lmp -in input_2D.lammps > looog.lammps")

    os.system("lmp -restart2data lammps2D.restart tmp2D_2.lammps > log2.lammps")
    #print(len(atoms),len(layer))
    
    atoms=read('tmp2D_2.lammps',format="lammps-data")
    atoms.center()
    atoms.wrap()
    layer = np.zeros(len(atoms))
    mid = np.mean(atoms.get_positions()[:, 2])
    layer[atoms.get_positions()[:, 2] < mid] = 1
    layer[atoms.get_positions()[:, 2] > mid] = 2
    atoms.set_array("mol-id", layer, dtype=int)
    born, charges_lammps,charges=born_charges(atoms)
    atoms.set_initial_charges(charges_lammps)
    atoms.set_array("born", born)
    atoms.set_array("charges_2", charges)
    voltage=[0.0,0.0]
    voltages=np.zeros(len(atoms))
    voltages[layer==1] = voltage[0]
    voltages[layer==2] = voltage[1]
    atoms.set_array("voltage", voltages)
    #atoms.calc = LAMMPS()
    

    return atoms
def wall_fit(x,x0,a0,w,b):
    return a0*np.tanh((x-x0)/w/2)+b

a=2.511
c=6.6612/2
Ly=2.511
Lx=4.349179577805451

v1=a*np.array([np.sqrt(3)/2,1/2,0])
v2=a*np.array([np.sqrt(3)/2,-1/2,0])

'''
def fit_zdata(zdata,V1,V2,n):
    # Fit zdata to harmonics of v1 and v2
    A=np.array([V1[:2], V2[:2]]).T
    B=np.linalg.inv(A.T)*2*np.pi
    g1=B[0]
    g2=B[1]
    g3=-g1-g2
    coeffs=[]

    for i in range(n):
'''   



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

def find_V0():
    AB=write_atoms([1/3,1/3,0])
    SP=write_atoms([1/2,1/2,0])
    
    born, charges_lammps, charges=born_charges(AB)
    AB.set_initial_charges(charges_lammps)
    AB.set_array("born", born)
    AB.set_array("charges_2", charges)
    voltage=[0.0,0.0]
    voltages=np.zeros(len(AB))
    voltages[AB.get_array("mol-id")==1] = voltage[0]
    voltages[AB.get_array("mol-id")==2] = voltage[1]
    AB.set_array("voltage", voltages)

    AB.calc = LAMMPS()

    E_AB=AB.get_potential_energy()

    born, charges_lammps, charges=born_charges(SP)
    SP.set_initial_charges(charges_lammps)
    SP.set_array("born", born)
    SP.set_array("charges_2", charges)
    SP.set_array("voltage", voltages)
    SP.calc = LAMMPS()
    E_SP=SP.get_potential_energy()

    V0=(E_SP-E_AB)/v1.dot(v2)

    return V0

def analyse_wall(atoms):
    
    s=1#v1.dot(v2)
    n_centers=20#round(len(atoms)/4)
    polarizations=np.zeros((n_centers**2,3))
    deformations=np.zeros((n_centers**2,3))
    centers=np.zeros((n_centers**2,2))
    ns=np.linspace(0,1,n_centers)
    for i in range(n_centers):
        for j in range(n_centers):
            centers[i*n_centers+j]=ns[i]*atoms.get_cell()[0][:2]+ns[j]*atoms.get_cell()[1][:2]
    pos=atoms.get_positions()
    ch=atoms.get_array("charges_2")
    born=atoms.get_array("born")
    #print(atoms.symbols)
    # B1 B2 N1 N2 is the order of the atoms in the file
    B1=pos[::4]
    #print(B1)
    ch_B1=ch[::4]   
    B2=pos[1::4]
    ch_B2=ch[1::4]
    N1=pos[2::4]
    ch_N1=ch[2::4]
    N2=pos[3::4]
    ch_N2=ch[3::4]
    '''
    print(ch_N1)

    B1_periodic=np.concatenate((B1, B1+v1, B1+v2, B1+v1+v2,B1-v1,B1-v2,B1-v1-v2,B1+v1-v2,B1-v1+v2), axis=0)
    B2_periodic=np.concatenate((B2, B2+v1, B2+v2, B2+v1+v2,B2-v1,B2-v2,B2-v1-v2,B2+v1-v2,B2-v1+v2), axis=0)
    N1_periodic=np.concatenate((N1, N1+v1, N1+v2, N1+v1+v2,N1-v1,N1-v2,N1-v1-v2,N1+v1-v2,N1-v1+v2), axis=0)
    N2_periodic=np.concatenate((N2, N2+v1, N2+v2, N2+v1+v2,N2-v1,N2-v2,N2-v1-v2,N2+v1-v2,N2-v1+v2), axis=0)
    tree = KDTree(B1_periodic[:,:2])
    KDmindist1,KDminindices1=tree.query(centers,k=1)
    tree = KDTree(B2_periodic[:,:2])
    KDmindist2,KDminindices2=tree.query(centers,k=1)
    tree = KDTree(N1_periodic[:,:2])
    KDmindist3,KDminindices3=tree.query(centers,k=1)
    tree = KDTree(N2_periodic[:,:2])
    KDmindist4,KDminindices4=tree.query(centers,k=1)
    #print(KDmindist)
    #print(KDminindices)
    for i in range(len(centers)):
        polarizations[i]+=ch_B1[KDminindices1[i]%len(B1)]*B1_periodic[KDminindices1[i]]
        polarizations[i]+=ch_B2[KDminindices2[i]%len(B1)]*B2_periodic[KDminindices2[i]]
        polarizations[i]+=ch_N1[KDminindices3[i]%len(B1)]*N1_periodic[KDminindices3[i]]
        polarizations[i]+=ch_N2[KDminindices4[i]%len(B1)]*N2_periodic[KDminindices4[i]]
    print(np.sum(polarizations,axis=0))
    print(np.sum(ch))
    '''
    # The data actually looks quite well but it needs to be smoothed, we'll use a periodic fit with the superlattice vectors
    order=3

    V1=atoms.get_cell()[0]
    V2=atoms.get_cell()[1]
    A=np.array([V1[:2], V2[:2]])
    B=np.linalg.inv(A.T)*2*np.pi
    #print(A)
    #print(B)
    g1=B[0]
    g2=B[1]
    g3=g1+g2
    g4=2*g1
    g5=2*g2
    g6=2*g1+g2
    g7=g1+2*g2
    g8=2*g1+2*g2


    def fit_zdata(points,av,p1,p2,p3,p4,p5,p6,p7,p8,i1,i2,i3,i4,i5,i6,i7,i8):
        return av+p1*np.cos(points.dot(g1))+p2*np.cos(points.dot(g2))+p3*np.cos(points.dot(g3))+p4*np.cos(points.dot(g4))+p5*np.cos(points.dot(g5))+p6*np.cos(points.dot(g6))\
               +i1*np.sin(points.dot(g1))+i2*np.sin(points.dot(g2))+i3*np.sin(points.dot(g3))+i4*np.sin(points.dot(g4))+i5*np.sin(points.dot(g5))+i6*np.sin(points.dot(g6))\
               +i7*np.sin(points.dot(g7))+i8*np.sin(points.dot(g8))+p7*np.cos(points.dot(g7))+p8*np.cos(points.dot(g8))
        #return av+p1*np.cos(points.dot(g1))+p2*np.cos(points.dot(g2))+p3*np.cos(points.dot(g3))\
        #       +i1*np.sin(points.dot(g1))+i2*np.sin(points.dot(g2))+i3*np.sin(points.dot(g3))  
    
    fitz1,_=curve_fit(fit_zdata, B1[:,:2], B1[:,2], maxfev=100000)
    #curve_fit()
    fitz2,_=curve_fit(fit_zdata, B2[:,:2], B2[:,2], maxfev=100000)
    fitz3,_=curve_fit(fit_zdata, N1[:,:2], N1[:,2], maxfev=100000)
    fitz4,_=curve_fit(fit_zdata, N2[:,:2], N2[:,2], maxfev=100000)
    fitq1,_=curve_fit(fit_zdata, B1[:,:2], ch_B1, maxfev=100000)
    fitq2,_=curve_fit(fit_zdata, B2[:,:2], ch_B2, maxfev=100000)
    fitq3,_=curve_fit(fit_zdata, N1[:,:2], ch_N1, maxfev=100000)
    fitq4,_=curve_fit(fit_zdata, N2[:,:2], ch_N2, maxfev=100000)
    #fitq1,_=curve_fit(fit_zdata, centers, ch_B1, maxfev=100000)
    #fitq2,_=curve_fit(fit_zdata, centers, ch_B2, maxfev=100000)
    #fitq3,_=curve_fit(fit_zdata, centers, ch_N1, maxfev=100000)
    #fitq4,_=curve_fit(fit_zdata, centers, ch_N2, maxfev=100000)
    
    # Works like a charm
    for i in range(len(centers)):
        polarizations[i,2]+=fit_zdata(centers[i], *fitz1)*fit_zdata(centers[i], *fitq1)
        polarizations[i,2]+=fit_zdata(centers[i], *fitz2)*fit_zdata(centers[i], *fitq2)
        polarizations[i,2]+=fit_zdata(centers[i], *fitz3)*fit_zdata(centers[i], *fitq3)
        polarizations[i,2]+=fit_zdata(centers[i], *fitz4)*fit_zdata(centers[i], *fitq4)
    # Can we get intralayer polarization?
    '''
    # This is a stupid approach but it might work
    # No it doesn't
    fitx1,_=curve_fit(fit_zdata, B1[:,:2], B1[:,0], maxfev=100000)
    fity1,_=curve_fit(fit_zdata, B1[:,:2], B1[:,1], maxfev=100000)
    fitx2,_=curve_fit(fit_zdata, B2[:,:2], B2[:,0], maxfev=100000)
    fity2,_=curve_fit(fit_zdata, B2[:,:2], B2[:,1], maxfev=100000)
    fitx3,_=curve_fit(fit_zdata, N1[:,:2], N1[:,0], maxfev=100000)
    fity3,_=curve_fit(fit_zdata, N1[:,:2], N1[:,1], maxfev=100000)
    fitx4,_=curve_fit(fit_zdata, N2[:,:2], N2[:,0], maxfev=100000)
    fity4,_=curve_fit(fit_zdata, N2[:,:2], N2[:,1], maxfev=100000)
    for i in range(len(centers)):
        polarizations[i,0]+=fit_zdata(centers[i], *fitx1)*fit_zdata(centers[i], *fitq1)
        polarizations[i,0]+=fit_zdata(centers[i], *fitx2)*fit_zdata(centers[i], *fitq2)
        polarizations[i,0]+=fit_zdata(centers[i], *fitx3)*fit_zdata(centers[i], *fitq3)
        polarizations[i,0]+=fit_zdata(centers[i], *fitx4)*fit_zdata(centers[i], *fitq4)
        polarizations[i,1]+=fit_zdata(centers[i], *fity1)*fit_zdata(centers[i], *fitq1)
        polarizations[i,1]+=fit_zdata(centers[i], *fity2)*fit_zdata(centers[i], *fitq2)
        polarizations[i,1]+=fit_zdata(centers[i], *fity3)*fit_zdata(centers[i], *fitq3)
        polarizations[i,1]+=fit_zdata(centers[i], *fity4)*fit_zdata(centers[i], *fitq4)
    '''



    
    
    #phi=np.linalg.norm(deformations[:,:2]-SP_v,axis=1)
    
    
    polarizations/=s

    # Polarization quanta
    #polarizations[:,0]-=-2.19292327e-02 
    #polarizations[:,1]-=-1.51700056e-08

    #V0=find_V0()
    #a0=a*np.sqrt(3)/3

    #lamé=gamma**2/V0/a0**2/9

    return centers, polarizations #centers[:,truth_axis], polarizations, phi, w_fit, lamé, deformations
        
def plot_polarization(centers, polarizations):
    
    plt.figure()
    plt.tricontourf(centers[:,0], centers[:,1], polarizations[:,2], levels=100, cmap='RdBu')
    plt.colorbar(label=r"$P_z$ [e/Å]")
    plt.xlabel(r"$x$ [Å]")
    plt.ylabel(r"$y$ [Å]")
    plt.axis('equal')
    plt.tight_layout()

    plt.figure()
    C=np.linalg.norm(polarizations[:,:2],axis=1)
    plt.quiver(centers[:,0], centers[:,1], polarizations[:,0]/C, polarizations[:,1]/C,C,pivot='middle', cmap='RdBu')
    plt.colorbar(label=r"$P_{xy}$ [e/Å]")
    plt.tight_layout()
    plt.xlabel(r"$x$ [Å]")
    plt.ylabel(r"$y$ [Å]")
    plt.axis('equal')
    plt.tight_layout()

def plot_deformation(centers, phi, deformations, w_fit,la):
    if la=='x':
        Lt=Ly
        Ll=Lx
    elif la=='y':
        Lt=Lx
        Ll=Lx
    plt.figure()
    plt.plot(centers, wall_fit(centers, *w_fit), 'b-',label=r"$tanh$ Fit")
    plt.plot(centers[:len(phi)], phi,'rx', label="Deformation")
    plt.axvline(x=w_fit[0]-2*w_fit[2],color='k', linestyle='--',label=r"$x_0\pm2w$")
    plt.axvline(x=w_fit[0]+2*w_fit[2],color='k', linestyle='--')
    plt.xlabel(rf"${la}$ (A)")
    plt.ylabel(r"Deformation (A)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.figure()
    plt.grid(True)
    plt.plot(centers,deformations[:,2],'rx')
    plt.tight_layout()
    plt.figure()
    plt.plot(centers,deformations[:,0],'rx',label=r"$\phi_x$")
    plt.plot(centers,deformations[:,1],'bx',label=r"$\phi_y$")
    plt.xlabel(rf"${la}$ (A)")
    plt.ylabel(r"Deformation (A)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.figure()
    burgers=np.zeros(len(centers))
    if la=='x':
        burgers=np.arctan2(deformations[:,1],deformations[:,0])*180/np.pi
    elif la=='y':
        burgers=np.arctan2(deformations[:,0],deformations[:,1])*180/np.pi
    plt.plot(centers,burgers,'rx')
    plt.xlabel(rf"${la}$ (A)")
    plt.ylabel(r"Burgers vector angle [°]")
    plt.grid(True)
    plt.tight_layout()


def full_creation_analysis(m,plot=True):
    # m is the twist parameter

    system0=minimize_lammps(m,defo=False)
    system1=minimize_lammps(m,defo=True)

    L_theta=np.linalg.norm(system0.get_cell()[0])
    A_theta=np.abs(system0.get_cell()[0].dot(system0.get_cell()[1]))
    
    
    system0.calc = LAMMPS()
    system1.calc = LAMMPS()
    E0=system0.get_potential_energy()
    E1=system1.get_potential_energy()
    gamma=(E1-E0)/3/L_theta
    s_energy=(E1-E0)/A_theta

    #centers, polarizations = analyse_wall(system1)
    #print(centers)
    #print(polarizations)
    #print(phi)
    if plot:
        centers, polarizations = analyse_wall(system1)
        print(f"Energy of wall: {gamma} eV/A")
        print(f"Surface energy: {s_energy} eV/A^2")
        #print(f"Effective Lamé parameter: {lamé} eV/A^2")
        #print(f"Wall width: {w_fit[2]} A")
        #print(f"Wall fit: {w_fit}")
        #plot_deformation(centers, phi, deformations,w_fit)
        plot_polarization(centers, polarizations)
    #print(polarizations[0])
    return system1, gamma, s_energy

def creation_analysis_parallel(ms, plot=True):
    """
    Run full_creation_analysis for a list of twist parameters ms and only calculate energies in the end
    """
    gammas = np.zeros(len(ms))
    s_energies = np.zeros(len(ms))
    Ls= np.zeros(len(ms))
    MoireStructs = []
    system0 = minimize_lammps(2, defo=False)
    print("Creating structures for twist parameters:", ms)
    for i in trange(len(ms)):
        m = ms[i]
        MoireStructs.append(minimize_lammps(m, defo=True))

        #np.savetxt("moire_results_big.dat", np.array([ms, Ls, gammas, s_energies]).T, header="m L gamma s_energy")

    print("Calculating energies for twist parameters:", ms)
    for i in trange(len(ms)):
        m = ms[i]
        system1= MoireStructs[i]
        L_theta=np.linalg.norm(system1.get_cell()[0])
        A_theta=np.abs(system1.get_cell()[0].dot(system1.get_cell()[1]))
        Ls[i]=L_theta
        
        A0=system0.get_cell()[0].dot(system0.get_cell()[1])
        system1.calc = LAMMPS()
        system0.calc = LAMMPS()
        E0=system0.get_potential_energy()*A_theta/A0
        E1=system1.get_potential_energy()
        gammas[i]=(E1-E0)/3/L_theta
        s_energies[i]=(E1-E0)/A_theta
    return ms, Ls, gammas, s_energies
'''
#ms=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#build_2D(60,defo=True,filename="test_60",verify=True,verbose=True)
ms=[55]#, 75, 100, 125, 150, 175, 200]
results=creation_analysis_parallel(ms, plot=False)
print("Twist parameter (m):", results[0])
print("Lattice length (A):", results[1])
print("Gamma (eV/A):", results[2])
print("Surface energy (eV/A^2):", results[3])
np.savetxt("moire_results_big3.dat", np.array([results[0], results[1], results[2], results[3]]).T, header="m L gamma s_energy")
#build_2D(m=3,defo=True,filename="tmp2D.lammps",verify=True)
'''
'''
ms=[2,3,4,5]#,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
gammas=np.zeros(len(ms))
s_energies=np.zeros(len(ms))
from tqdm import tqdm
for i, m in tqdm(enumerate(ms)):
    _,gammas[i],s_energies[i]=full_creation_analysis(m,plot=False)
print("Twist parameter (m):", ms)
print("Gamma (eV/A):", gammas)
print("Surface energy (eV/A^2):", s_energies)
MoireStruct=full_creation_analysis(m=21,plot=False)
#view(MoireStruct)
#plt.show()
'''

def analyse_cavity(atoms):
    
    s=v1.dot(v2)
    n_centers=20#round(len(atoms)/4)
    polarizations=np.zeros((n_centers**2,3))
    deformations=np.zeros((n_centers**2,3))
    centers=np.zeros((n_centers**2,2))
    ns=np.linspace(0,1,n_centers)
    for i in range(n_centers):
        for j in range(n_centers):
            centers[i*n_centers+j]=ns[i]*atoms.get_cell()[0][:2]+ns[j]*atoms.get_cell()[1][:2]
    pos=atoms.get_positions()
    ch=atoms.get_array("charges_2")
    born=atoms.get_array("born")
    #print(atoms.symbols)
    # B1 B2 N1 N2 is the order of the atoms in the file
    na=len(pos)
    B1=pos[:round(2*na/3):4]
    #print(B1)
    ch_B1=ch[:round(2*na/3):4]   
    B2=pos[1:round(2*na/3):4]
    ch_B2=ch[1:round(2*na/3):4]
    N1=pos[2:round(2*na/3):4]
    ch_N1=ch[2:round(2*na/3):4]
    N2=pos[3:round(2*na/3):4]
    ch_N2=ch[3:round(2*na/3):4]

    V1=atoms.get_cell()[0]
    V2=atoms.get_cell()[1]
    A=np.array([V1[:2], V2[:2]])
    B=np.linalg.inv(A.T)*2*np.pi

    g1=B[0]
    g2=B[1]
    g3=g1+g2
    g4=2*g1
    g5=2*g2
    g6=2*g1+g2
    g7=g1+2*g2
    g8=2*g1+2*g2


    def fit_zdata(points,av,p1,p2,p3,p4,p5,p6,p7,p8,i1,i2,i3,i4,i5,i6,i7,i8):
        return av+p1*np.cos(points.dot(g1))+p2*np.cos(points.dot(g2))+p3*np.cos(points.dot(g3))+p4*np.cos(points.dot(g4))+p5*np.cos(points.dot(g5))+p6*np.cos(points.dot(g6))\
               +i1*np.sin(points.dot(g1))+i2*np.sin(points.dot(g2))+i3*np.sin(points.dot(g3))+i4*np.sin(points.dot(g4))+i5*np.sin(points.dot(g5))+i6*np.sin(points.dot(g6))\
               +i7*np.sin(points.dot(g7))+i8*np.sin(points.dot(g8))+p7*np.cos(points.dot(g7))+p8*np.cos(points.dot(g8))
    
    fitz1,_=curve_fit(fit_zdata, B1[:,:2], B1[:,2], maxfev=100000)
    fitz2,_=curve_fit(fit_zdata, B2[:,:2], B2[:,2], maxfev=100000)
    fitz3,_=curve_fit(fit_zdata, N1[:,:2], N1[:,2], maxfev=100000)
    fitz4,_=curve_fit(fit_zdata, N2[:,:2], N2[:,2], maxfev=100000)
    fitq1,_=curve_fit(fit_zdata, B1[:,:2], ch_B1, maxfev=100000)
    fitq2,_=curve_fit(fit_zdata, B2[:,:2], ch_B2, maxfev=100000)
    fitq3,_=curve_fit(fit_zdata, N1[:,:2], ch_N1, maxfev=100000)
    fitq4,_=curve_fit(fit_zdata, N2[:,:2], ch_N2, maxfev=100000)

    
    # Works like a charm
    for i in range(len(centers)):
        polarizations[i,2]+=fit_zdata(centers[i], *fitz1)*fit_zdata(centers[i], *fitq1)
        polarizations[i,2]+=fit_zdata(centers[i], *fitz2)*fit_zdata(centers[i], *fitq2)
        polarizations[i,2]+=fit_zdata(centers[i], *fitz3)*fit_zdata(centers[i], *fitq3)
        polarizations[i,2]+=fit_zdata(centers[i], *fitz4)*fit_zdata(centers[i], *fitq4)

    
    polarizations/=s

    return centers, polarizations