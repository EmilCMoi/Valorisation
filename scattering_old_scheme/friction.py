import numpy as np
from DW_dynamics import run_dw_dynamics_2, continue_run, plot_dynamics_new, plot_dynamics_continued, continue_lammps, analyze_lammps
from draw import cmaps
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rcParams['figure.constrained_layout.use'] = True
Lflep, Lflep_r, Dflep, Dflep_r=cmaps()
def fit_v(t,dV,c=0.9155):
    a=dV*5.407e-04
    vtilde=0
    v=c*(a*t+vtilde)/np.sqrt(c**2+(a*t)**2+2*a*t*vtilde+vtilde**2)
    return v

EfieldStepss=[2,5,10,15]
#EfieldSteps=5
#5,10,15
dV=-1
#velocities=fit_v(EfieldStepss,dV)

#Nstepsi=50
Nstepsi=200000
Nsteps=300000
Nx=1
Ny=300
dir='0'
Nevery=500
#run_dw_dynamics_2(Nx,Ny,dir,EfieldSteps,Nstepsi,dV,NVT=False,Temperature=0)
#for EfieldSteps in EfieldStepss:
#    continue_run(Nsteps-Nstepsi,Nevery,Nx,Ny,dir,EfieldSteps,Nstepsi,dV,NVT=False,Temperature=0)

EfSteps=[2,5,10,15]
colors=Dflep(np.linspace(0, 1, len(EfSteps)))
#plot_dynamics_continued(Nx,Ny,dir,EfieldSteps,Nsteps,dV)
def tanhfit(x, a, b, c, d):
    return a * np.tanh(b * x ) 
def tanhfit_deriv(x, a, b, c,d):
    return a * b / np.cosh(b * x + c)**2
def tanhfit_deriv2(x, a, b, c,d):
    return -2 * a * b**2 * np.tanh(b * x + c) / np.cosh(b * x + c)**2
def expfit(x, a, b, c):
    b=7.21426204e-06
    return -a/b*np.exp(-b * x) + c
def freakyahhfit(x,a,b,d):
    #print(a,b,c,d)
    #a=2.05388266e-05
    c=-9.66655716e-05*0.9155
    return b+(np.pi/2-np.arctan(np.sinh(a*x+d)))*c/a#np.cosh(a*x+d)*c/np.cosh(a*x+d)/a

for EfieldSteps in EfSteps:
    dips=np.load(f"data/polarizations2_{dir}_{dV}_{Nsteps}_{EfieldSteps}_{Nx}_{Ny}.npy")
    dip=np.sum(dips,axis=1)
    t=np.loadtxt(f"data/times_{dir}_{dV}_{Nsteps}_{EfieldSteps}_{Nx}_{Ny}.txt")
    p=np.polyfit(t,dip,3)
    ps=np.poly1d(p)
    dps=ps.deriv()/-9.66655716e-05
    ddps=dps.deriv()

    #fit,_=curve_fit(tanhfit,t,dip,maxfev=10000)
    #fit=[3.35405538e-02, 1.89837719e-05,1,1]
    #plt.figure()
    #fit=curve_fit(expfit,t,dip,maxfev=10000)[0]
    fit=curve_fit(freakyahhfit,t,dip,p0=[1e-5,1,1],maxfev=10000)[0]
    #fit[0]=1.01495004e-05
    #sfit[2]=-1.04395705e-06
    plt.grid(True)
    plt.plot(t/1000,dip,'r-',color=colors[EfSteps.index(EfieldSteps)],label=fr"$v_i$={-fit_v(EfieldSteps,dV)/0.9155:.3f} c")
    plt.xlabel("Time [ps]")
    plt.ylabel("Total Dipole moment [a.u.]")
    #plt.plot(t,freakyahhfit(t,*fit),'b--')
    plt.legend()

    #plt.plot(t,tanhfit(t,*fit),'b--')
    
    print(fit)
for EfieldSteps in EfSteps:
    plot_dynamics_continued(Nx,Ny,dir,EfieldSteps,Nsteps,dV)
    plt.title(fr"$v_i$={-fit_v(EfieldSteps,dV)/0.9155:.3f} c")
plt.show()
