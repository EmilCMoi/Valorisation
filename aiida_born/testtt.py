import numpy as np

import matplotlib.pyplot as plt

angs=[0,np.pi/6,np.pi/3,np.pi/2]
ens=[0.0763,0.0880,0.1119,0.1244]

plt.figure()
plt.plot(angs,ens,'o-')
plt.xlabel("Angle (rad)")
plt.ylabel("Energy (eV)")
plt.grid(True)
plt.show()