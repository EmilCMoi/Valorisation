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
def fit_v(t,dV,c=0.9155):
    a=dV*5.407e-04
    vtilde=0
    v=c*(a*t+vtilde)/np.sqrt(c**2+(a*t)**2+2*a*t*vtilde+vtilde**2)
    return v

ts=np.arange(1000)
plt.figure()
plt.grid(True)
plt.plot(ts,-fit_v(ts,-1),'r-',label='-10 V')
for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    plt.axhline(i,ls='--',color='gray')
plt.show()

# Times needed to reach certain velocities (dV=-10V) (0.1 to 0.9 c)
# 17, 35, 53, 74, 98, 127, 166, 226, 350