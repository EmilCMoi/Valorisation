/*
//#include <iostream>
//#include "knncpp.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

// these are LAMMPS include files
#include "lammps.h"
#include "input.h"
#include "atom.h"
#include "library.h"
#include "molecule.h"

using namespace std;
using namespace LAMMPS_NS;

// mpicxx -I${HOME}/mylammps/src -c simple.cpp
// mpicxx -L${HOME}/mylammps/src simple.o -llammps -o simpleCC

int main(int argc, char** argv) {
    cout << "This is a placeholder for the born_charges.cpp file." << endl;
    return 0;
}

int main(int narg, char **arg)
{
  // setup MPI and various communicators
  // driver runs on all procs in MPI_COMM_WORLD
  // comm_lammps only has 1st P procs (could be all or any subset)

  MPI_Init(&narg,&arg);

  if (narg != 3) {
    printf("Syntax: %s P in.lammps\n", arg[0]);
    exit(1);
  }

  int me,nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD,&me);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

  int nprocs_lammps = atoi(arg[1]);
  if (nprocs_lammps > nprocs) {
    if (me == 0)
      printf("ERROR: LAMMPS cannot use more procs than available\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  int lammps;
  if (me < nprocs_lammps) lammps = 1;
  else lammps = MPI_UNDEFINED;
  MPI_Comm comm_lammps;
  MPI_Comm_split(MPI_COMM_WORLD,lammps,0,&comm_lammps);
  
  // open LAMMPS input script

  FILE *fp;
  if (me == 0) {
    fp = fopen(arg[2],"r");
    if (fp == nullptr) {
      printf("ERROR: Could not open LAMMPS input script\n");
      MPI_Abort(MPI_COMM_WORLD,1);
    }
  }

  // run the input script thru LAMMPS one line at a time until end-of-file
  // driver proc 0 reads a line, Bcasts it to all procs
  // (could just send it to proc 0 of comm_lammps and let it Bcast)
  // all LAMMPS procs call input->one() on the line
  
  LAMMPS *lmp = nullptr;
  if (lammps == 1) lmp = new LAMMPS(0,nullptr,comm_lammps);

  int n;
  char line[1024];
  while (true) {
    if (me == 0) {
      if (fgets(line,1024,fp) == nullptr) n = 0;
      else n = strlen(line) + 1;
      if (n == 0) fclose(fp);
    }
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    if (n == 0) break;
    MPI_Bcast(line,n,MPI_CHAR,0,MPI_COMM_WORLD);
    if (lammps == 1) lammps_command(lmp,line);
  }

  delete lmp;
  if (lammps == 1) MPI_Comm_free(&comm_lammps);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
*/
#include "lammps.h"

#include <mpi.h>
#include <iostream>

int main(int argc, char **argv)
{
    LAMMPS_NS::LAMMPS *lmp;
    // custom argument vector for LAMMPS library
    const char *lmpargv[] {"liblammps", "-log", "none"};
    int lmpargc = sizeof(lmpargv)/sizeof(const char *);

    // explicitly initialize MPI
    MPI_Init(&argc, &argv);

    // create LAMMPS instance
    lmp = new LAMMPS_NS::LAMMPS(lmpargc, (char **)lmpargv, MPI_COMM_WORLD);
    // output numerical version string
    std::cout << "LAMMPS version ID: " << lmp->num_ver << std::endl;
    // delete LAMMPS instance
    delete lmp;

    // stop MPI environment
    MPI_Finalize();
    return 0;
}