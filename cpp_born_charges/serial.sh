mpicxx -I${HOME}/mylammps/src -c born_charges.cpp
mpicxx -L${HOME}/mylammps/src born_charges.o -llammps -o born_charges
./born_charges 1 input_continue.lammps 0.001 100