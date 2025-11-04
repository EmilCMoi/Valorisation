import numpy as np
import matplotlib.pyplot as plt

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

rho=3.286e-27 # mass density in kg/A^2

#for th in ths:
#    print(np.cos(th)**2,np.sin(th)**2)
thetas=np.linspace(0,np.pi/2,100)
plt.figure()
plt.plot(thetas*180/np.pi,gamma(a0,lambd(thetas,lambda0,mu0),V0),'ro')
plt.plot([th*180/np.pi for th in ths],gs,'bx')
plt.xlabel("Theta (deg)")
plt.ylabel("Gamma")

plt.figure()
plt.plot(thetas*180/np.pi,np.sqrt(lambd(thetas,lambda0,mu0)/rho*1.602e-19),'ro')

plt.figure()
plt.plot(thetas*180/np.pi,(lambd(thetas,lambda0,mu0)),'ro')
plt.plot(ths*180/np.pi,lambd(ths,lambda0,mu0),'bx')
plt.xlabel("Theta (deg)")
plt.ylabel("Lambda (eV/A^2)")
plt.show()