import numpy as np
import matplotlib.pyplot as plt
from utils_1D import full_creation_analysis_convergence

Ns=[300,400,500,600,700,800,900,1000,2000,3000,5000]

gammas=full_creation_analysis_convergence(Ns,'0')
Ns=np.array(Ns)
print(gammas)
np.savez("mass_convergence_0.npz",Ns=Ns,gammas=gammas)
plt.figure()
plt.plot(Ns*2.511,gammas*1000,'o-')
plt.xlabel("System length [Å]")
plt.ylabel("Domain wall energy [meV/Å]")
plt.grid(True)
plt.show()
