import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('output_0.010000_15000.txt')
time=data[:,0]
energy=data[:,1]
polarization=data[:,4]

plt.figure()
plt.plot(time,energy)
plt.xlabel('Time (fs)')
plt.ylabel('Energy (eV)')
plt.grid()  
plt.figure()
plt.plot(time,polarization)
plt.xlabel('Time (fs)')
plt.ylabel('Polarization (a.u.)')
plt.grid()  
plt.show()