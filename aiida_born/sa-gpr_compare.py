import numpy as np
import matplotlib.pyplot as plt

dataA=np.loadtxt('predictionA.txt')
dataB=np.loadtxt('predictionB.txt')
dataC=np.loadtxt('predictionC.txt')
dataD=np.loadtxt('predictionD.txt')

dataA=dataA.reshape(12,2,9)
dataB=dataB.reshape(12,2,9)
dataC=dataC.reshape(12,2,9)
dataD=dataD.reshape(12,2,9)

print(dataA)
plt.figure()
plt.plot(dataA[:,0,:].flatten(),dataA[:,1,:].flatten(),'o')
plt.plot(dataB[:,0,:].flatten(),dataB[:,1,:].flatten(),'o')
plt.plot(dataC[:,0,:].flatten(),dataC[:,1,:].flatten(),'o')
plt.plot(dataD[:,0,:].flatten(),dataD[:,1,:].flatten(),'o')
#plt.plot([-10,10],[-10,10],'k--')
plt.show()