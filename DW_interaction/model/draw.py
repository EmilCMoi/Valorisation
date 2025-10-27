import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
'''
This script generates all the necessary plots for the report
'''

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'serif'})

# Colormap shenanigans

rouge = (1, 0, 0)
groseille = (181/255, 31/255, 31/255)
leman = (0, 167/255, 159/255)
canard = (0, 116/255, 128/255)
taupe = (65/255, 61/255, 58/255)
perle = (202/255, 199/255, 199/255)
cdict = {'red':   [[0.0,  181/255, 181/255],
                   [0.1,  1.0, 1.0],
                   [0.5,  1, 1],
                   [0.9, 0, 0],
                   [1.0,0,0]],
         'green': [[0.0,  31/255, 31/255],
                   [0.1, 0.0, 0.0],
                   [0.5, 1, 1],
                   [0.9, 167/255, 167/255],
                   [1.0,  116/255, 116/255]],
         'blue':  [[0.0,  31/255, 31/255],
                   [0.1,0,0],
                   [0.5,  1, 1],
                   [0.9,159/255,159/255],
                   [1.0,  128/255, 128/155]]}
cdict2={'red':    [[0.0,  1.0, 1.0],
                   [0.1,  181/255, 181/255],
                   [0.5,  65/255, 65/255],
                   [0.9,0,0],
                   [1.0, 0, 0]
                   ],
         'green': [[0.0,  0.0,0.0],
                   [0.1, 31/255, 31/255],
                   [0.5, 61/255, 61/255],
                   [0.9, 116/255, 116/255],
                   [1.0,  167/255, 167/255]],
         'blue':  [[0.0,  0.0,0.0],
                   [0.1,31/255,31/255],
                   [0.5,  58/255, 58/255],
                   [0.9,128/255,128/255],
                   [1.0,  159/255, 159/155]]}

newcmp = LinearSegmentedColormap('Lflep', segmentdata=cdict, N=256)
newcmp2 = LinearSegmentedColormap('testCmap', segmentdata=cdict2, N=256)

Lflep=newcmp
Lflep_r=newcmp.reversed()
Dflep=newcmp2
Dflep_r=newcmp2.reversed()
def cmaps():
    return Lflep, Lflep_r, Dflep, Dflep_r
'''
def plot_dipole(dipole,time):
    plt.figure()
    plt.plot(time,dipole, color=groseille, linewidth=2)
    plt.xlabel(r"$t$ [fs]")
    plt.ylabel(r"$d_z$ [eÅ]")

def movie(charges,positions, times, filename="movie.mp4"):
    # First create polarization data from positions and charges
    polarization = np.zeros((len(times), 3))
    for i in range(len(times)):
        for j in range(len(charges[i])):
            polarization[i] += charges[i][j] * positions[i][j]
'''