from ase.io.trajectory import Trajectory
from ase.visualize import view
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
#traj= Trajectory('A2B.traj')
#print(len(traj))
'''
-12053.046652808482
-12052.801725397996
-12052.21883445041
-12051.83768158364
-12051.879936052892
-12052.221283138324
-12052.383074285297
'''
'''
Energies=[-12053.046652808482,
-12052.801725397996,
-12052.21883445041,
-12051.83768158364,
-12051.879936052892,
-12052.221283138324,
-12052.383074285297]
'''
Energies, dipoles=np.loadtxt("neb.dat", unpack=True)
Energies1, dipoles1=np.loadtxt("neb_1.0.dat", unpack=True)
Energies2, dipoles2=np.loadtxt("neb_2.0.dat", unpack=True)
Energies3, dipoles3=np.loadtxt("neb_3.0.dat", unpack=True)
plt.figure()
plt.plot(Energies,'rx--',label=r'$E_z=0.0$',markersize=10)
plt.plot(Energies1,'bx--',label=r'$E_z\approx 0.3$',markersize=10)
plt.plot(Energies2,'gx--',label=r'$E_z\approx 0.6$',markersize=10)
plt.plot(Energies3,'kx--',label=r'$E_z\approx 0.9$',markersize=10)
plt.legend()
plt.xlabel('NEB step')
plt.ylabel('Energy [eV]')
plt.title('NEB Energy Convergence')
plt.grid(True)
plt.tight_layout()
plt.figure()
plt.plot(dipoles,Energies,'rx--',label=r'$E_z=0.0$',markersize=10)
plt.plot(dipoles1,Energies1,'bx--',label=r'$E_z\approx 0.3$',markersize=10)
plt.plot(dipoles2,Energies2,'gx--',label=r'$E_z\approx 0.6$',markersize=10)
plt.plot(dipoles3,Energies3,'kx--',label=r'$E_z\approx 0.9$',markersize=10)
plt.legend()
plt.xlabel('Dipole Moment [eÅ]')
plt.ylabel('Energy [eV]')
plt.title('NEB Energy vs Dipole Moment')
plt.grid(True)
plt.tight_layout()

plt.show()