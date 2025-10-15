from DW_vacancy import vacancy_energy, rebuild_system
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from mpi4py import MPI

me = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()

Nx=1 
Ny=300
system=rebuild_system(Nx,Ny,'0')
print(vacancy_energy(system,'0',1))
energies=np.zeros(len(system)//2)
for i in trange(len(system)//2):
    energies[i]=vacancy_energy(system,'0',i)
    if i%20==0:
        np.savetxt("vacancy_energies_1.txt",energies)
np.savetxt("vacancy_energies_1.txt",energies)
plt.plot(energies)
plt.show()
