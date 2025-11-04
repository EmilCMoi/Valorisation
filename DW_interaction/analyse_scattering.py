import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("scattering_0.txt")

vin=data[:,0]
vout=data[:,1]
plt.plot(vin,vout,'o')
plt.show()