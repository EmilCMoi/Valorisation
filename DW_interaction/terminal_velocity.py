import numpy as np
import matplotlib.pyplot as plt
from draw import cmaps
from tqdm import trange
from numba import jit
from scipy.optimize import curve_fit
Lflep, Lflep_r, Dflep, Dflep_r=cmaps()
# Numerical integration of the equations of motion to get terminal velocity of a DW under an applied voltage

# 2 parts to the acceleration: relativistic E-field driving force and friction force proportional to velocity
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['figure.constrained_layout.use'] = True

@jit(nopython=True,cache=True)
def gamma(v,c):
    return 1/np.sqrt(1-(v/c)**2)

@jit(nopython=True,cache=True)
def F_field(E,alpha,v,c):
    return alpha*E/gamma(v,c)**3
@jit(nopython=True,cache=True)
def F_friction(beta,v):
    return -beta*v
@jit(nopython=True,cache=True)
def dv_dt(E,alpha,beta,v,c):
    return (F_field(E,alpha,v,c)+F_friction(beta,v))/gamma(v,c)**3
@jit(nopython=True,cache=True)
def run_terminal_velocity(E,alpha,beta,c,v0=0,tmax=1e7,dt=1e3):
    times=np.linspace(0,tmax,int(tmax/dt))
    vs=np.zeros(len(times))
    vs[0]=v0
    for i in range(1,len(times)):
        a=dv_dt(E,alpha,beta,vs[i-1],c)
        vs[i]=vs[i-1]+a*dt
    return times,vs
@jit(nopython=True,cache=True)
def update(x,v,dt,E,alpha,beta,c):
    a=dv_dt(E,alpha,beta,v,c)
    vnew=v+a*dt
    xnew=x+vnew*dt
    return xnew,vnew

def tanh_fit(x,w):
    return np.tanh(w*x) 
Es=np.linspace(-1,1,200)
vmaxs=np.zeros(len(Es))
colors=Dflep(np.linspace(0, 1, len(Es)))
plt.figure()
plt.grid(True)
for i in trange(len(Es)):
    ts,vs=run_terminal_velocity(E=Es[i],alpha=1,beta=0.2,c=1,v0=0,tmax=100,dt=1e-3)
    plt.plot(ts,vs,label=f"E={Es[i]:.2f}",color=colors[i])
    vmaxs[i]=vs[-1]
plt.xlabel("Time (fs)")
plt.ylabel("DW Velocity ")
#plt.legend()
fit,_=curve_fit(tanh_fit,Es,vmaxs,p0=[1])
plt.figure()
plt.plot(Es,tanh_fit(Es,*fit),label=f"Fit: 2={fit[0]:.2f}",color='k')
plt.plot(Es,vmaxs,'x')
plt.grid(True)
plt.xlabel("Applied E-field ")
plt.ylabel("Terminal DW Velocity ")
plt.legend()
plt.show()