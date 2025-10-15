import numpy as np
import matplotlib.pyplot as plt
from ase.io import Trajectory
from tqdm import trange

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['figure.constrained_layout.use'] = True

Lx=4.349179577805451#2.511
Ly=2.511
def get_deformations(system,Nx,Ny,dir):
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
    #polarizations = np.zeros((Ns,3))
    deformations = np.zeros((Ns,3))
    for i in range(Nx):
        for j in range(Ny):
            if dir=='0' or dir=='60':
                l=j
            elif dir=='30' or dir=='90':
                l=i
            for k in range(4):
                #polarizations[l] += system.get_array("charges_model")[4*(i+j*Nx)+k]*system.positions[4*(i+j*Nt)+k]/Nt
                #polarizations[l]+=system.get_array("charges_model")[4*(i+j*Nx)+k+round(len(system)/2)]*system.positions[4*(i+j*Nt)+k+round(len(system)/2)]/Nt
                deformations[l] += system.positions[4*(i+j*Nx)+k+round(len(system)/2)] - system.positions[4*(i+j*Nx)+k]
    deformations/=4*Nt
    a=2.511
    if dir=='0' or dir=='90':
        SP_v=np.array([0,a*np.sqrt(3)/3])
    elif dir=='60' or dir =='30':
        SP_v=np.array([-a*np.sqrt(3)/3*np.cos(np.pi/3)/2,a*np.sqrt(3)/3*np.sin(np.pi/3)/2])
    phi=np.linalg.norm(deformations[:,:2]-SP_v,axis=1)

    return  phi
print("Reading trajectory")
#traj=Trajectory('data/DW2_0_-10_700_100_1_300.traj', 'r')
#phis=np.load('data/deformations2_0_-10_400_100_1_300.npy')
phis=np.load('data/deformations2_0_-10_15500_350_1_300.npy')
for i in trange(len(phis)):#trange(1):
    #system=traj[i]
    #print(system.get_velocities())
    #phi=get_deformations(system,1,300,'0')
    phi=phis[i]
    plt.figure()
    plt.grid(True)
    plt.plot(np.arange(len(phi))*Ly,phi,'r-')
    plt.xlabel("y [A]")
    plt.ylabel("Phi [A]")
    plt.title(f"t={i*10} fs")
    plt.ylim(1.3,3.8)
    plt.savefig(f"frames/3_{i}.jpg")
    plt.close()
from ase.io import write
#print(np.max(system.get_velocities()))
#write("DW2_dir_0_in.lmp", system, format='lammps-data',velocities=True,atom_style='full')


#Make film

print("Making film")
import imageio.v2 as imageio

images = []
for i in trange(len(phis)):
    images.append(imageio.imread(f'frames/3_{i}.jpg'))
imageio.mimsave('3_movie.gif', images)
