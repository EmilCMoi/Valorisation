from model.double_build_1D import double_build_1D
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.visualize import view

double_build_1D(Nx=2, Ny=20, dir='zigzag', defo=True, fileName='test.lammps', plot=True)
#double_build_1D(Nx=100, Ny=2, dir='armchair', defo=True, fileName='test.lammps', plot=True)
atoms=read('test.lammps',format="lammps-data")
atoms.wrap()
atoms.center()
view(atoms)

plt.show()