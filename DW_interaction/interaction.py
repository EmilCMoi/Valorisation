import numpy as np
import matplotlib.pyplot as plt

from DW_dynamics import get_polarizations
from DW_dynamics import wall_fit
from scipy.optimize import curve_fit
from model.build_1D import build_1D_deformation
from utils_1D import rebuild_system_def, rebuild_system
from model.calculator import LAMMPS
from tqdm import trange
from draw import cmaps
from scipy.fft import fftshift
from scipy.spatial import KDTree
Lflep, Lflep_r, Dflep, Dflep_r=cmaps()

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['figure.constrained_layout.use'] = True

Ly=2.511
Nx=1
Ny=300
dir='0'



#a0*np.tanh(2*(x-x1)/abs(w))-a0*np.tanh(2*(x-x2)/abs(w))+c
eta=2
def def_f(i,j,x1,x2,w,a0,c):
    x= j*Ly

    x3=300*Ly+x1
    x4=-300*Ly+x1
    x5=300*Ly+x2
    x6=-300*Ly+x2
    
    tmp=a0*np.tanh(2*(x - x1)/abs(w))#*(np.abs(x-x1)<=eta*w)+a0*np.sign(x-x1)*(np.abs(x-x1)>eta*w)
    tmp-=a0*np.tanh(2*(x - x2)/abs(w))#*(np.abs(x-x2)<=eta*w) -a0*np.sign(x-x2)*(np.abs(x-x2)>eta*w)
    tmp+=a0*np.tanh(2*(x-x3)/abs(w))#*(np.abs(x-x3)>eta*w)#+a0*np.sign(x-x3)*(np.abs(x-x3)>eta*w)
    tmp+=a0*np.tanh(2*(x-x4)/abs(w))#*(np.abs(x-x4)>eta*w)#+a0*np.sign(x-x4)*(np.abs(x-x4)>eta*w)
    tmp-=a0*np.tanh(2*(x-x5)/abs(w))#*(np.abs(x-x5)>eta*w)#+a0*np.sign(x-x5)*(np.abs(x-x5)>eta*w)
    tmp-=a0*np.tanh(2*(x-x6)/abs(w))#*(np.abs(x-x6)>eta*w)#+a0*np.sign(x-x6)*(np.abs(x-x6)>eta*w)
    tmp*=np.sign(x2-x1)
    tmp+=c
    return np.array([tmp,0])
    #return np.array([(a0*np.tanh(2*(x - x1)/abs(w))*(np.abs(x-x1)<=eta*w)+a0*np.sign(x-x1)*(np.abs(x-x1)>eta*w) - a0*np.tanh(2*(x - x2)/abs(w))*(np.abs(x-x2)<=eta*w) -a0*np.sign(x-x2)*(np.abs(x-x2)>eta*w))*np.sign(x2-x1) + c,0])
def tmp_def(xs,x1,x2,w,a0,c):
    x=xs
    # Artificially turning the function periodic (only 1 period though)
    #if x1<xs[len(xs)//4] or x1>xs[3*len(xs)//4]:
    #    #x=fftshift(xs)
    #    tt=x1
    #    x1=x2
    #    x2=-tt
    #    sgn=-1
    #    x=fftshift(x)
    #else:
    #    sgn=1
    #tmp=sgn*(a0*np.tanh(2*(x - x1)/abs(w))*(np.abs(x-x1)<=eta*w)+a0*np.sign(x-x1)*(np.abs(x-x1)>eta*w) - a0*np.tanh(2*(x - x2)/abs(w))*(np.abs(x-x2)<=eta*w) -a0*np.sign(x-x2)*(np.abs(x-x2)>eta*w))*np.sign(x2-x1) + c
    #if x1<xs[len(xs)//4] or x1>xs[3*len(xs)//4]:
        #tmp=#fftshift(tmp)+2*a0

    x3=300*Ly+x1
    x4=-300*Ly+x1
    x5=300*Ly+x2
    x6=-300*Ly+x2
    
    tmp=a0*np.tanh(2*(x - x1)/abs(w))#*(np.abs(x-x1)<=eta*w)+a0*np.sign(x-x1)*(np.abs(x-x1)>eta*w)
    tmp-=a0*np.tanh(2*(x - x2)/abs(w))#*(np.abs(x-x2)<=eta*w) -a0*np.sign(x-x2)*(np.abs(x-x2)>eta*w)
    tmp+=a0*np.tanh(2*(x-x3)/abs(w))#*(np.abs(x-x3)>eta*w)#+a0*np.sign(x-x3)*(np.abs(x-x3)>eta*w)
    tmp+=a0*np.tanh(2*(x-x4)/abs(w))#*(np.abs(x-x4)>eta*w)#+a0*np.sign(x-x4)*(np.abs(x-x4)>eta*w)
    tmp-=a0*np.tanh(2*(x-x5)/abs(w))#*(np.abs(x-x5)>eta*w)#+a0*np.sign(x-x5)*(np.abs(x-x5)>eta*w)
    tmp-=a0*np.tanh(2*(x-x6)/abs(w))#*(np.abs(x-x6)>eta*w)#+a0*np.sign(x-x6)*(np.abs(x-x6)>eta*w)
    tmp*=np.sign(x2-x1)
    tmp+=c
    return tmp
def initial_fit(system):
    xs=np.arange(Ny)*Ly
    _,phi=get_polarizations(system,Nx,Ny,dir)
    fit,_=curve_fit(wall_fit, xs,phi,p0=[xs[len(xs)//4],3*xs[len(xs)//4],60,2,2])
    return fit
def displace_DWs(system, displacement,dwidth):
    #xs=300*Ly
    xs=np.arange(Ny)*Ly
    _,phi=get_polarizations(system,Nx,Ny,dir)
    fit,_=curve_fit(wall_fit, xs,phi,p0=[xs[len(xs)//4],3*xs[len(xs)//4],60,2,2])
    #print(fit)
    

    fit[0]+=displacement
    fit[1]-=displacement
    fit[2]=dwidth
    #plt.plot(xs,phi)
    #plt.plot(xs,tmp_def(xs,*fit))
    #plt.show()
   

    build_1D_deformation(Nx,Ny,dir,def_f,fit,f"displaced_DW_{displacement:.2f}.lmp")
    #new_defo=wall_fit(xs,fit[0]+displacement,fit[1]-displacement,fit[2],fit[3])
def kd_freeze_indices(system,dir,x1,x2):
    
    if dir=='0' or dir=='60':
        tree=KDTree(system.positions[:,0:1])
    else:
        tree=KDTree(system.positions[:,1:2])
    return np.concatenate(tree.query(x1,4)[1],tree.query(x2,4)[1])

ff=initial_fit(rebuild_system(Nx,Ny,dir))
x10=ff[0]
x20=ff[1]
w0=ff[2]
a0=ff[3]
c0=ff[4]
print(ff)
x1s=np.arange(10)*300*Ly/10
x2s=300*Ly-x1s
xs=np.arange(Ny)*Ly

colors=Dflep(np.linspace(0, 1, 10))
for i in range(10):
   plt.plot(tmp_def(xs,x1s[i],x2s[i],w0,a0,c0), color=colors[i], label=f"Dw distance={(x2s[i]-x1s[i]):.1f} A")
plt.legend()
#plt.show()

Nd=201
defos=np.linspace(-(x20-x10)/2,2*(x20-x10),Nd)
#defos=[(x20-x10)/2+(x20-x10)/6]
system=rebuild_system(Nx,Ny,dir)
Nw=1
dws=np.linspace(w0/Nw,w0,Nw)
energies=np.zeros((Nd,Nw))
#plt.figure()

systems=[]
for i in trange(Nd):
    for j in range(Nw):
        displace_DWs(system,defos[i],dws[j])
        freeze_indices=kd_freeze_indices(system,dir,x10+defos[i],x20-defos[i],relax=True)
        new_system=rebuild_system_def(Nx,Ny,dir,defos[i],freeze_indices=freeze_indices)
        new_system.pbc=[True,True,False]
        #new_system.calc=LAMMPS()
        #energies[i,j]=new_system.get_potential_energy()
        systems.append(new_system)

for i in trange(Nd):
    for j in range(Nw):
        new_system=systems[i*Nw+j]
        new_system.pbc=[True,True,False]
        new_system.calc=LAMMPS()
        energies[i,j]=new_system.get_potential_energy()

#plt.show()

np.savez("DW_displacement_energy_6.npz",defos=defos,dws=dws,energies=energies)

plt.figure()
plt.plot(energies)
plt.show()
'''
defos,dws,energies=np.load("DW_displacement_energy_6.npz").values()
tdefos=np.zeros(2*len(defos)-1)
tenergies=np.zeros((2*len(defos)-1,len(dws)))
plt.figure()
plt.plot(energies)
plt.show()

plt.figure()
plt.contourf(dws,x20-x10-2*defos,energies-np.min(energies),levels=50,cmap=Dflep)
plt.xlabel("DW width (A)")
plt.ylabel("DW distance (A)")
plt.colorbar(label="Total Energy (eV)")

tdefos[:len(defos)]=defos
tenergies[:len(defos)]=energies
for i in range(len(defos)-1):
    tdefos[len(defos)+i]=defos[-1]+defos[len(defos)-1-i]
    tenergies[len(defos)+i]=energies[i]
tenergies=tenergies[:,1:]
dws=dws[1:]
#plt.figure()
#plt.plot(x20-x10-2*defos,energies-energies[0],'o-')
#plt.xlabel("DW distance (A)")
#plt.ylabel("Energy difference (eV)")

plt.figure()
plt.pcolor(dws,x20-x10-2*tdefos,tenergies-np.min(tenergies),cmap=Dflep)
plt.xlabel("DW width (A)")
plt.ylabel("DW distance (A)")
plt.colorbar(label="Total Energy (eV)")

bim=np.argsort(x20-x10-2*tdefos)
colors=Dflep(np.linspace(0, 1, len(dws)))
plt.figure()
plt.grid(True)
for i in range(len(dws)):
    plt.plot((x20-x10-2*tdefos)[bim],tenergies[bim,i]-np.min(tenergies[:,i]),'rx-',label=f"Width={dws[i]:.1f} A",color=colors[i])
plt.xlabel("DW distance (A)")
plt.ylabel("Energy difference (eV)")
plt.legend()

plt.figure()
plt.grid(True)
for i in range(len(dws)):
    xs=Ly*np.arange(Ny)
    plt.plot(xs,wall_fit(xs,x10,x20,dws[i],0.58906484, 2.05600221),label=f"Width={dws[i]:.1f} A",color=colors[i])
plt.xlabel("Position (A)")
plt.ylabel("Polarization (a.u.)")
plt.legend()

plt.figure()
#for i in range(len(dws)):
plt.plot(dws,np.max(tenergies,axis=0)-np.min(tenergies,axis=0),'rx-')
plt.xlabel("DW width (A)")
plt.ylabel("Energy barrier (eV)")
plt.legend()
plt.figure()
plt.plot(dws,np.max(tenergies,axis=0)-tenergies[0,:],'rx-')
plt.xlabel("DW width (A)")
plt.ylabel("Energy barrier (eV)")
'''
plt.show()
