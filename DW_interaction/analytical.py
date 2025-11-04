import numpy as np
import matplotlib.pyplot as plt
from draw import cmaps
from tqdm import trange

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['figure.constrained_layout.use'] = True

Lflep, Lflep_r, Dflep, Dflep_r=cmaps()

def analytic_interaction(r,a0,V0,lambda0):
    return 1/(3*(-1+np.exp(-2*np.sqrt(2)*np.sqrt(V0/lambda0)*np.abs(r)/a0))**3)*8*np.sqrt(2*lambda0*V0)* \
        (a0*(-1-17*np.exp(-6*np.sqrt(2*V0/lambda0)*np.abs(r)/a0)+9*np.exp(-4*np.sqrt(2*V0/lambda0)*np.abs(r)/a0)\
             +9*np.exp(-2*np.sqrt(2*V0/lambda0)*np.abs(r)/a0))-12*np.sqrt(2*V0/lambda0)*np.abs(r)*np.exp(-4*np.sqrt(2*V0/lambda0)*np.abs(r)/a0)*(3+np.exp(-2*np.sqrt(2*V0/lambda0)*np.abs(r)/a0)))


rs=np.linspace(-20,20,101)
a0=1
V0=1
lambda0=1
#print(np.abs(0))
ints=analytic_interaction(rs,a0,V0,lambda0)

plt.figure()
plt.plot(rs,ints)
plt.xlabel("DW distance r")
plt.ylabel("Interaction energy E_int")
plt.title("Analytical DW interaction energy")
plt.grid()
plt.show()