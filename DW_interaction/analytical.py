import numpy as np
import matplotlib.pyplot as plt
from draw import cmaps
from tqdm import trange

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['figure.constrained_layout.use'] = True

Lflep, Lflep_r, Dflep, Dflep_r=cmaps()

def analytic_interaction(r,a0,V0,lambda0):
    return 1/(3*(-1+np.exp(2*np.sqrt(2)*np.sqrt(V0/lambda0)*r/a0))**3)*8*np.sqrt(2*lambda0*V0)* \
        (a0*(-1-17*np.exp(6*np.sqrt(2*V0/lambda0)*r/a0)+9*np.exp(4*np.sqrt(2*V0/lambda0)*r/a0)\
             +9*np.exp(2*np.sqrt(2*V0/lambda0)*r/a0))+12*np.sqrt(2*V0/lambda0)*r*np.exp(4*np.sqrt(2*V0/lambda0)*r/a0)*(3+np.exp(2*np.sqrt(2*V0/lambda0)*r/a0)))

def gamma(a0,lambd,V0):
    return 4/3*a0*np.sqrt(2*lambd*V0)#a0*(lambd+2/3*np.sqrt(lambd*V0*2))#a0*3*np.sqrt(lambd*V0/2)# a0*(2*np.sqrt(2*lambd*V0)/3)+0.04
def lambd(theta,lambda0,mu0):
    return ((lambda0)*np.cos(theta)**2 + mu0*np.sin(theta)**2*2/3)
a0=2.511*np.sqrt(3)/6
V0=1.439e-3
lambda0=1.779
mu0=7.939
ths=np.array([0,np.pi/6, np.pi/3, np.pi/2])
gs=[0.0763, 0.0880, 0.1119, 0.1244]

rs=np.linspace(-60,20,1001)
lambda0=lambd(0,lambda0,mu0)
#print(np.abs(0))
dints=np.diff(analytic_interaction(rs,a0,V0,lambda0))
ints=analytic_interaction(rs,a0,V0,lambda0)

plt.figure()
plt.plot(rs,ints/2/2.511/np.sqrt(3))
plt.xlabel("DW distance r")
plt.ylabel("Interaction energy E_int")
plt.title("Analytical DW interaction energy")
plt.grid()

plt.figure()
plt.plot(rs[:-1],dints)
plt.xlabel("DW distance r")
plt.ylabel("Interaction force F_int")
plt.title("Analytical DW interaction force")
plt.grid()
plt.show()