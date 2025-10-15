import numpy as np
import matplotlib.pyplot as plt

fits=np.loadtxt("fit2_vals_0_-5.txt")
#polarizations=np.load("polarizations2_0_1.npy")

positions=fits[:,0]
widths=fits[:,1]
heights=fits[:,2]
centers=fits[:,3]

velocities=np.diff(positions)
accelerations=np.diff(velocities)

plt.figure()
plt.plot(positions)
plt.figure()
plt.plot(velocities)
plt.figure()
plt.plot(accelerations)
plt.show()