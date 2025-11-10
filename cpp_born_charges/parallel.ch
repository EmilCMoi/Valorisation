mpicxx -I${HOME}/mylammps/src -c born_charges.cpp
mpicxx -L${HOME}/mylammps/src born_charges.o -llammps -o born_charges
mpirun -np 16 ./born_charges 16 input_continue.lammps 