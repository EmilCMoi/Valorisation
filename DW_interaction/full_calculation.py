import numpy as np
import matplotlib.pyplot as plt
from draw import cmaps
from tqdm import trange
from numba import jit
Nx=2000

Lx=200

dx=Lx/Nx

Lflep, Lflep_r, Dflep, Dflep_r=cmaps()

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['figure.constrained_layout.use'] = True


@jit(nopython=True,cache=True)
def phi0(x,d):
    return np.tanh((x+d/2)/np.sqrt(2))-np.tanh((x-d/2)/np.sqrt(2))-1

@jit(nopython=True,cache=True)
def phixx(phi,dx,pbc=False):
    d2phi_dx2=np.zeros(len(phi))
    for i in range(1,len(phi)-1):
        d2phi_dx2[i]=(phi[i+1]-2*phi[i]+phi[i-1])/(dx**2)
    # Periodic boundary conditions
    if pbc:
        d2phi_dx2[0]=(phi[1]-2*phi[0]+phi[-1])/(dx**2)
        d2phi_dx2[-1]=(phi[0]-2*phi[-1]+phi[-2])/(dx**2)
    # Von Neumann BCs put this to 0
    return d2phi_dx2
@jit(nopython=True,cache=True)
def phitt(phi,phit,G,E,dx,pbc=False):
    return -G*phit + E + phi -phi**3 + phixx(phi,dx,pbc)
@jit(nopython=True)
def run_simulation(E,G,d0,Nt,dt,save_interval):
    # Velocity Verlet integration
    xs=np.linspace(-Lx/2,Lx/2,Nx)
    phi=phi0(xs,d0)
    phit=np.zeros(len(phi))
    phis=np.zeros((int(Nt//save_interval),len(phi)))
    count=0
    phis[0,:]=phi
    for n in range(Nt):
        a=phitt(phi,phit,G,E,dx,pbc=False)
        phit_half=phit + 0.5*a*dt
        phi=phi + phit_half*dt
        a_new=phitt(phi,phit_half,G,E,dx,pbc=False)
        phit=phit_half + 0.5*a_new*dt
        if (n)%save_interval==0:
            phis[count,:]=phi
            count+=1
        
    return xs,phis
T=2000000

dt=0.05
Nt=int(T/dt)

xs,phis=run_simulation(E=-0.0005,G=30,d0=100,Nt=Nt,dt=dt,save_interval=Nt//10)

#plot_interval=Nt//10
colors=Dflep_r(np.linspace(0, 1, len(phis)))
for n in range(len(phis)):
    plt.plot(xs,phis[n,:],label=f"t={n*dt:.1f} fs", color=colors[n])
plt.xlabel("Position (A)")
plt.ylabel("Order parameter $\phi$")
plt.show()