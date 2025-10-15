import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
from draw import cmaps

Lflep, Lflep_r, Dflep, Dflep_r=cmaps()

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['figure.constrained_layout.use'] = True
c=0.9155
def fit_x(t,a,prop):
    vtilde=0
    #v=c*(a*t+vtilde)/np.sqrt(c**2+(a*t)**2+2*a*t*vtilde+vtilde**2)
    xtilde=-c*np.sqrt(vtilde**2+c**2)/a # Initial position is 0
    x=xtilde+c/a*np.sqrt(c**2+(a*t)**2+2*a*t*vtilde+vtilde**2)
    return x*prop+0.000471


#dVs=[-1,-2,-3,-4]
dVs=[-1]
Nx=1
Ny=300
dir='0'
#Nsteps=1000
#NstepsEfield=1000
Nsteps=5000
NstepsEfield=5000
sampledColors=Dflep(np.linspace(0, 1, len(dVs)))
for i,dV in enumerate(dVs):
    data=np.load(f"data/polarizations2_{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.npy")
    if i==0:
        dip=np.sum(data,axis=1)
        t=np.arange(len(dip))
        dVss=np.repeat(dV,len(data))
    else:
        dip=np.concatenate((dip,np.sum(data,axis=1)))
        t=np.concatenate((t,np.arange(len(data))))
        dVss=np.concatenate((dVss,np.repeat(dV,len(data))))

def to_min(params):
    a,prop=params
    sum=0
    for i in range(len(dip)):
        sum+=(fit_x(t[i],a*dVss[i],prop)-dip[i])**2
    #print(sum)
    return sum

#fit=minimize(to_min,[1,1],method='Nelder-Mead',options={'xatol':1e-12,'fatol':1e-12,'maxfev':100000})
#print(fit)
#print(fit.x)

#xx=[0.0005,0.9,-1e-4]
#0.0005,0.9,-1e-4
#1.633e-03  2.766e+00 -3.200e-05
#xx=fit.x
xx=[5.40690551e-04, -9.66655716e-05]
plt.figure()
plt.grid(True)
for i in range(len(dVs)):
    data=np.load(f"data/polarizations2_{dir}_{dVs[i]}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.npy")
    dip=np.sum(data,axis=1)
    t=np.arange(len(dip))
    plt.plot(t,fit_x(t,xx[0]*dVs[i],xx[1]),'b--')#,color=sampledColors[i])
    plt.plot(t[::200],dip[::200],'x',label=rf"$E_z\approx{dVs[i]/(6.602/2):.2f}$ V/Å",color=sampledColors[i],markersize=10,linewidth=4)
plt.legend()
plt.xlabel("Time [fs]")
plt.ylabel("Total Dipole moment [a.u.]")
plt.show()
'''
    data=np.load(f"data/polarizations2_{dir}_{dV}_{Nsteps}_{NstepsEfield}_{Nx}_{Ny}.npy")
    dip=np.sum(data,axis=1)
    t=np.arange(len(dip))
    fit,_=curve_fit(fit_x,t,dip,ftol=1e-12,maxfev=100000)

    print(f"dV={dV}, a={fit[0]}, prop={fit[1]}")
    #plt.plot(t,fit_x(t,*fit),linestyle='dashed')
    plt.plot(t,dip**2,label=f"dV={dV}")
    avals[i]=fit[0]
    #cvals[i]=fit[1]
    propvals[i]=fit[1]
plt.legend()
plt.xlabel("Time step")
plt.ylabel("Dipole moment")

plt.figure()
plt.grid(True)
plt.plot(dVs,avals,marker='o')
plt.xlabel("dV")
plt.ylabel("a")

plt.figure()
plt.grid(True)
plt.plot(dVs,cvals,marker='o')
plt.xlabel("dV")
plt.ylabel("c")

plt.figure()
plt.grid(True)
plt.plot(dVs,propvals,marker='o')
plt.xlabel("dV")
plt.ylabel("prop")

plt.show()

'''