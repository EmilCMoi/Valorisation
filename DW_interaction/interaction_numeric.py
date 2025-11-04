import numpy as np
import matplotlib.pyplot as plt
from draw import cmaps
from tqdm import trange

from numba import jit
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['figure.constrained_layout.use'] = True

ff=[189.36674159, 561.5515542, 59.30191732, 0.58906484, 2.05600221]
Ly=2.511
Ny=1000

xs=np.arange(Ny)*Ly

Lflep, Lflep_r, Dflep, Dflep_r=cmaps()
# Without time

def walls(xs,x1,x2,w,a0):
    x=xs

    x3=Ny*Ly+x1
    x4=-Ny*Ly+x1
    x5=Ny*Ly+x2
    x6=-Ny*Ly+x2
    
    tmp=a0*np.tanh((x - x1)/abs(w))#*(np.abs(x-x1)<=eta*w)+a0*np.sign(x-x1)*(np.abs(x-x1)>eta*w)
    tmp-=a0*np.tanh((x - x2)/abs(w))#*(np.abs(x-x2)<=eta*w) -a0*np.sign(x-x2)*(np.abs(x-x2)>eta*w)
    tmp+=a0*np.tanh((x-x3)/abs(w))#*(np.abs(x-x3)>eta*w)#+a0*np.sign(x-x3)*(np.abs(x-x3)>eta*w)
    tmp+=a0*np.tanh((x-x4)/abs(w))#*(np.abs(x-x4)>eta*w)#+a0*np.sign(x-x4)*(np.abs(x-x4)>eta*w)
    tmp-=a0*np.tanh((x-x5)/abs(w))#*(np.abs(x-x5)>eta*w)#+a0*np.sign(x-x5)*(np.abs(x-x5)>eta*w)
    tmp-=a0*np.tanh((x-x6)/abs(w))#*(np.abs(x-x6)>eta*w)#+a0*np.sign(x-x6)*(np.abs(x-x6)>eta*w)
    #tmp-=a0
    tmp*=np.sign(x2-x1)
    tmp-=a0
    return tmp
def V(phi,V0,a0):
    return V0/a0**4*(a0**2-phi**2)**2/4
def deriv_walls(xs,x1,x2,w,a0):
    x=xs

    x3=Ny*Ly+x1
    x4=-Ny*Ly+x1
    x5=Ny*Ly+x2
    x6=-Ny*Ly+x2
    
    tmp= (a0/abs(w))/np.cosh((x - x1)/abs(w))**2
    tmp-= (a0/abs(w))/np.cosh((x - x2)/abs(w))**2
    tmp+= (a0/abs(w))/np.cosh((x - x3)/abs(w))**2
    tmp+= (a0/abs(w))/np.cosh((x - x4)/abs(w))**2
    tmp-= (a0/abs(w))/np.cosh((x - x5)/abs(w))**2
    tmp-= (a0/abs(w))/np.cosh((x - x6)/abs(w))**2
    tmp*=np.sign(x2-x1)
    return tmp

x10=ff[0]
x20=ff[1]

a0=2#ff[3]
#w0=ff[2]
#c0=-1#ff[4]
lambda0=7#2*a0/(w0)
V0=0.001#lambda0**2*a0**4/4
w0=a0*np.sqrt(lambda0/V0/2)
w0*=0.5
print(ff)
x1s=np.arange(21)*Ny*Ly/21
x2s=Ny*Ly-x1s
xs=np.arange(Ny)*Ly

colors=Dflep(np.linspace(0, 1, 21))
plt.figure()
for i in range(10):
   plt.plot(walls(xs,x1s[i],x2s[i],w0,a0), color=colors[i], label=f"Dw distance={(x2s[i]-x1s[i]):.1f} A")
plt.legend()
#plt.show()


d0=Ny/2*Ly
ds=np.linspace(-2*d0,2*d0,1001)

def denergy(xs,x1,x2,w,a0):
    return lambda0/2*deriv_walls(xs,x1,x2,w,a0)**2+V(walls(xs,x1,x2,w,a0),V0,a0)
plt.figure()

for i in range(10):
    plt.plot(denergy(xs,x1s[i],x2s[i],w0,a0), color=colors[i], label=f"Dw distance={(x2s[i]-x1s[i]):.1f} A")
xs=np.linspace(0,Ny*Ly,10000)
energies=[]
for i in trange(len(ds)):
    d=ds[i]
    x1=Ny/2*Ly-d/2#x10 - d/2
    x2=Ny/2*Ly+d/2

    energy=np.trapz(denergy(xs,x1,x2,w0,a0),xs)
    energies.append(energy)
    #print(f"d={d:.1f} Energy={energy:.6f}")

plt.figure()
plt.plot(ds,energies)
plt.xlabel("DW Separation (A)")
plt.ylabel("Energy (eV)")
plt.show()
'''

# With time
# For simplicity, one of the DWs is kept fixed, while the other is moving relativistically
@jit(nopython=True,cache=True)
def walls(xs,x1,x2,w,a0,v):
    x=xs

    x3=Ny*Ly+x1
    x4=-Ny*Ly+x1
    x5=Ny*Ly+x2
    x6=-Ny*Ly+x2
    gamma=1/np.sqrt(1-(v/c)**2)
    tmp=a0*np.tanh(gamma*(x - x1)/abs(w))#*(np.abs(x-x1)<=eta*w)+a0*np.sign(x-x1)*(np.abs(x-x1)>eta*w)
    tmp-=a0*np.tanh((x - x2)/abs(w))#*(np.abs(x-x2)<=eta*w) -a0*np.sign(x-x2)*(np.abs(x-x2)>eta*w)
    tmp+=a0*np.tanh(gamma*(x-x3)/abs(w))#*(np.abs(x-x3)>eta*w)#+a0*np.sign(x-x3)*(np.abs(x-x3)>eta*w)
    tmp+=a0*np.tanh(gamma*(x-x4)/abs(w))#*(np.abs(x-x4)>eta*w)#+a0*np.sign(x-x4)*(np.abs(x-x4)>eta*w)
    tmp-=a0*np.tanh((x-x5)/abs(w))#*(np.abs(x-x5)>eta*w)#+a0*np.sign(x-x5)*(np.abs(x-x5)>eta*w)
    tmp-=a0*np.tanh((x-x6)/abs(w))#*(np.abs(x-x6)>eta*w)#+a0*np.sign(x-x6)*(np.abs(x-x6)>eta*w)
    tmp*=np.sign(x2-x1)
    #tmp+=c
    tmp-=a0
    return tmp
@jit(nopython=True,cache=True)
def deriv_walls(xs,x1,x2,w,a0,v):
    x=xs

    x3=Ny*Ly+x1
    x4=-Ny*Ly+x1
    x5=Ny*Ly+x2
    x6=-Ny*Ly+x2
    gamma=1/np.sqrt(1-(v/c)**2)
    tmp= gamma*(a0/abs(w))/np.cosh(gamma*(x - x1)/abs(w))**2
    tmp-= (a0/abs(w))/np.cosh((x - x2)/abs(w))**2
    tmp+= gamma*(a0/abs(w))/np.cosh(gamma*(x - x3)/abs(w))**2
    tmp+= gamma*(a0/abs(w))/np.cosh(gamma*(x - x4)/abs(w))**2
    tmp-= (a0/abs(w))/np.cosh((x - x5)/abs(w))**2
    tmp-= (a0/abs(w))/np.cosh((x - x6)/abs(w))**2
    tmp*=np.sign(x2-x1)
    #print(w)
    return tmp
@jit(nopython=True,cache=True)
def deriv_t_walls(xs,x1,x2,w,a0,v):
    x=xs

    x3=Ny*Ly+x1
    x4=-Ny*Ly+x1
    x5=Ny*Ly+x2
    x6=-Ny*Ly+x2
    gamma=1/np.sqrt(1-(v/c)**2)
    #print(np.min(gamma*(x - x1)/abs(w)))
    tmp= -v*gamma*(a0/abs(w))/np.cosh(gamma*(x - x1)/abs(w))**2
    #tmp-= (a0/abs(w))/np.cosh((x - x2)/abs(w))**2
    tmp+= -v*gamma*(a0/abs(w))/np.cosh(gamma*(x - x3)/abs(w))**2
    tmp+= -v*gamma*(a0/abs(w))/np.cosh(gamma*(x - x4)/abs(w))**2
    #tmp-= (a0/abs(w))/np.cosh((x - x5)/abs(w))**2
    #tmp-= (a0/abs(w))/np.cosh((x - x6)/abs(w))**2
    tmp*=np.sign(x2-x1)
    return tmp
@jit(nopython=True,cache=True)
def V(phi,V0,a0):
    return V0/a0**4*(a0**2-phi**2)**2/4

x10=ff[0]
x20=ff[1]
w0=ff[2]
a0=1#ff[3]
#c0=ff[4]
lambda0=1
V0=0.001
w0=a0*np.sqrt(lambda0/V0/2)
rho=1# Arbitrary mass density
print(ff)
x1s=np.arange(10)*Ny*Ly/10
x2s=Ny*Ly-x1s
xs=np.arange(Ny)*Ly
c=1
d0=Ny/2*Ly
ds=np.linspace(-d0/2,d0/2,301)
vs=np.linspace(-0.9*c,0.9*c,301)

colors=Dflep(np.linspace(0, 1, 10))
@jit(nopython=True,cache=True)
def denergy(xs,x1,x2,w,a0,v):
    return lambda0/2*deriv_walls(xs,x1,x2,w,a0,v)**2+V(walls(xs,x1,x2,w,a0,v),V0,a0)+rho/2*deriv_t_walls(xs,x1,x2,w,a0,v)**2

energies=np.zeros((len(ds),len(vs)))

for i in trange(len(ds)):
    for j in range(len(vs)):
        d=ds[i]
        v=vs[j]
        x1=Ny/2*Ly-d/2#x10 - d/2
        x2=Ny/2*Ly+d/2

        energies[i,j]=np.trapz(denergy(xs,x1,x2,w0,a0,v),xs)
        #print(f"d={d:.1f} v={v:.3f} Energy={energy:.6f}")
from matplotlib import cm, ticker
plt.figure()
plt.plot(denergy(xs,x10,x20,w0,a0,0.1), label="v=0")
plt.xlabel("DW Separation (A)")
plt.ylabel("Energy (eV)")

plt.figure()
plt.contourf(ds,vs,energies.T,levels=100,cmap=Dflep)
plt.contour(ds,vs,energies.T,levels=30,colors='w',linewidths=0.5)#,locator=ticker.LogLocator(subs='auto'))
plt.xlabel("DW Separation (A)")
plt.ylabel("DW Velocity (c)")
plt.colorbar(label="Energy (eV)")
plt.show()
'''