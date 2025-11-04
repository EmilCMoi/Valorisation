import numpy as np
import matplotlib.pyplot as plt
from draw import cmaps
from tqdm import trange
from numba import jit
from scipy.fft import fftshift
#Nx=1000

Lx=100
dx=0.1
Nx=int(Lx/dx)
dx=Lx/Nx

T=2000

dt=0.05
Nt=int(T/dt)
Nimages=1000
Nsteps_Efield=60
d0=50
E=-0.1
Lflep, Lflep_r, Dflep, Dflep_r=cmaps()
ti=800#int(Nsteps_Efield*dt)
tf=950
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['figure.constrained_layout.use'] = True


@jit(nopython=True,cache=True)
def phi0(x,d):
    return np.tanh((x+d/2)/np.sqrt(2))-np.tanh((x-d/2)/np.sqrt(2))-1
@jit(nopython=True,cache=True)
def phixx(phi,dx,pbc):
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
def phix(phi,dx,pbc):
    dphi_dx=np.zeros(len(phi))
    for i in range(1,len(phi)-1):
        dphi_dx[i]=(phi[i+1]-phi[i-1])/(2*dx)
    # Periodic boundary conditions
    if pbc:
        dphi_dx[0]=(phi[1]-phi[-1])/(2*dx)
        dphi_dx[-1]=(phi[0]-phi[-2])/(2*dx)
    # Von Neumann BCs ?
    return dphi_dx
@jit(nopython=True,cache=True)
def phitt(phi,phit,G,E,dx,pbc=False):
    return -G*phit + E + phi -phi**3 + phixx(phi,dx,pbc)
@jit(nopython=True,cache=True)
def energy(phi,phit,dx,pbc):
    phx=phix(phi,dx,pbc)
    return np.sum(0.5*phit**2 + 0.5*phx**2 + 0.25*(1 - phi**2)**2)*dx, np.sum(0.5*phx**2 + 0.25*(1 - phi**2)**2)*dx, np.sum(0.5*phit**2)*dx
@jit(nopython=True)
def run_simulation(E,G,d0,Nt,dt,save_interval,Nsteps_Efield,pbc):
    # Velocity Verlet integration
    xs=np.linspace(-Lx/2,Lx/2,Nx)
    phi=phi0(xs,d0)
    phit=np.zeros(len(phi))
    phis=np.zeros((int(Nt//save_interval),len(phi)))
    count=0
    phis[0,:]=phi
    dz=np.zeros(Nt)
    energies=np.zeros((Nt,3))
    for n in range(Nt):
        if n>Nsteps_Efield:
            E=0
        a=phitt(phi,phit,G,E,dx,pbc)
        phit_half=phit + 0.5*a*dt
        phi=phi + phit_half*dt
        a_new=phitt(phi,phit_half,G,E,dx,pbc)
        phit=phit_half + 0.5*a_new*dt
        energies[n]=energy(phi,phit,dx,pbc)
        dz[n]=np.sum(phi)*dx/Lx
        if (n)%save_interval==0:
            phis[count,:]=phi
            count+=1
        

    return xs,phis,energies,dz

xs,phis,energies,dz=run_simulation(E=E,G=0.0,d0=d0,Nt=Nt,dt=dt,save_interval=Nt//Nimages,Nsteps_Efield=Nsteps_Efield,pbc=True)

#plot_interval=Nt//10
#colors=Dflep_r(np.linspace(0, 1, len(phis)))
plt.figure()
#plt.grid(True)
#for n in range(len(phis)):
#    plt.plot(xs,phis[n,:],label=f"t={n*dt:.1f} fs", color=colors[n])
plt.pcolormesh(np.arange(Nimages)*Nt//Nimages*dt,np.arange(len(xs))*dx,phis.T,cmap=Dflep_r)
plt.colorbar()
plt.xlabel("Position (A)")
plt.ylabel("Order parameter $\phi$")

plt.figure()
plt.grid(True)
plt.plot(np.arange(Nt)*dt, energies[:,0],'r-', label='Total Energy')
#plt.plot(np.arange(Nt)*dt, energies[:,2],'g-', label='Kinetic Energy')
#plt.plot(np.arange(Nt)*dt, energies[:,1],'b-', label='Potential Energy')
plt.axvline(x=Nsteps_Efield*dt, color='k', linestyle='--', label='E-field turned off')
plt.xlabel("Time (fs)")
plt.ylabel("Energy")
plt.figure()
plt.grid(True)
plt.plot(np.arange(Nt)*dt, dz,'b-')
fit_range=range(int(ti/dt),int(tf/dt))
coeffs=np.polyfit(np.array(fit_range)*dt,dz[fit_range],1)
plt.plot(np.array(fit_range)*dt, np.polyval(coeffs, np.array(fit_range)*dt), 'r--', label=f'Fit slope={coeffs[0]:.4f} A/fs')
plt.axvline(x=Nsteps_Efield*dt, color='k', linestyle='--', label='E-field turned off')
plt.xlabel("Time (fs)")
plt.ylabel("Polarization ")
plt.legend()
plt.show()