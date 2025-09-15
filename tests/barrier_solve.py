import numpy as np
import matplotlib.pyplot as plt
# -7.29333842e+02  6.69091301e-03
#  0.00121232
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})

E0=6.69091301e-03
dz0=0.00121232

def fitE(x,a=E0):
    return a*2*np.cos(x)+a*np.cos(2*x)
def fitdz(x,a=dz0):
    return 2*a*np.sin(x)-a*np.sin(2*x)

ps=np.linspace(2*np.pi/3,4*np.pi/3,100)
#Es=[0,0.3,0.6,0.9,0.12]
N=5
Es=np.linspace(0,1.2,N)
maxs=np.zeros(N)
plt.figure()

for i,E in enumerate(Es):
    tmp=fitE(ps)+E*fitdz(ps)
    tmp-=tmp[0]
    maxs[i]=np.max(tmp)
    plt.plot(ps,tmp)
plt.grid(True)
plt.tight_layout()

plt.figure()
plt.plot(Es,maxs)
plt.grid(True)
plt.tight_layout()

plt.show()