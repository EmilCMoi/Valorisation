import numpy as np
import matplotlib.pyplot as plt
from DW_vacancy import rebuild_system

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['figure.constrained_layout.use'] = True
Lx=4.349179577805451
vacancy_energy=np.loadtxt("vacancy_energies_0.txt")/Lx

system=rebuild_system(1,300,'0')
plt.figure()
print(system.positions[:len(vacancy_energy)//2:2])
plt.grid(True)
plt.plot(system.positions[:len(vacancy_energy)//2:2][:,1],vacancy_energy[:len(vacancy_energy)//2:2]-vacancy_energy[0],'r-',label='B')
#plt.figure()
#plt.grid(True)
plt.plot(system.positions[1:len(vacancy_energy)//2:2][:,1],vacancy_energy[1:len(vacancy_energy)//2:2]-vacancy_energy[1],'b-',label='N')
plt.axvline(system.positions[len(system)//8][1],ls='--',color='k',label='DW')
plt.axvline(system.positions[3*len(system)//8][1],ls='--',color='k')
plt.legend()
plt.xlabel(r"$x$ [Å]")
plt.ylabel("Vacancy formation energy [eV/Å]")
plt.show()