
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <cmath>
#include "string.h"
// these are LAMMPS include files
#include "lammps.h"
#include "input.h"
#include "atom.h"
#include "library.h"
#include "molecule.h"

#include "modify.h"
#include "fix.h"
#include "fix_external.h"

// necessary for drivinng LAMMPS from external code
#include "many2one.h"
#include "one2many.h"
#include "files.h"
#include "memory.h"
#include "error.h"


using namespace std;
using namespace LAMMPS_NS;

// callback function that will calculate born charges and the effect of an efield
void born_callback(void *, bigint, int, int *, double **, double **);

// struct to pass multiple info to the callback function
struct Info {
  int me;
  LAMMPS *lmp;
  FixExternal *fix;
  double *efield;
};

/*
FORMAT: mpirun -np P ./born_charges P in.lammps in.efield R RR
         P = # of procs to run LAMMPS on
             must be <= # of procs the driver code itself runs on
         in.lammps = LAMMPS input script
         efield = eletric field value
         nsteps = number of steps to run
*/
int main(int narg, char **arg)
{
  // setup MPI and various communicators
  // driver runs on all procs in MPI_COMM_WORLD
  // comm_lammps only has 1st P procs (could be all or any subset)
  char str[128];
  MPI_Init(&narg,&arg);

  if (narg != 5) {
    printf("Syntax: ./born_charges P in.lammps efield nsteps \n");
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
  
  // Memory *memory = new Memory(comm_lammps);
  // Error *error = new Error(comm_lammps);
  // open LAMMPS input script

  FILE *fp;
  if (me == 0) {
    fp = fopen(arg[2],"r");
    if (fp == nullptr) {
      printf("ERROR: Could not open LAMMPS input script\n");
      MPI_Abort(MPI_COMM_WORLD,1);
    }
  }

  // run the input script through LAMMPS one line at a time until end-of-file
  // driver proc 0 reads a line, Bcasts it to all procs
  // (could just send it to proc 0 of comm_lammps and let it Bcast)
  // all LAMMPS procs call input->one() on the line
  
  LAMMPS *lmp = nullptr;
  if (lammps == 1) lmp = new LAMMPS(0,nullptr,comm_lammps);

  // Initialization of the simulation conditions
  if (lammps == 1) {
    lmp->input->file(arg[2]);
  }
  
  // set up callback for born charges and efield effect
  double efield_value = atof(arg[3]);
  int nsteps = atoi(arg[4]);
  
  Info info;
  info.me = me;
  info.efield = &efield_value;
  info.lmp = lmp;
  //int niter=1000;
  
  //if (lammps==1)
  //{
  //int ifix = lmp->modify->get_fix_by_id("born");
  if (lammps == 1) {
    FixExternal *fix = (FixExternal *) lmp->modify->get_fix_by_id("born");//lmp->modify->fix[ifix];
    info.fix = fix;
    fix->set_callback(born_callback, &info);
  }
  //info.quest_input = quest_input;

  // run the simulation
  cout << "Starting simulation..." << endl;
  string command = "run " + to_string(nsteps);
  if (lammps == 1) lammps_commands_string(lmp, command.c_str());

  delete lmp;
  //delete memory;
  //delete error;
  if (lammps == 1) MPI_Comm_free(&comm_lammps);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}

// callback function that will calculate born charges and the effect of an efield
void born_callback(void *ptr, bigint ntimestep, int nlocal, int *id, double **x, double **f)
{
  // cast the void pointer to a LAMMPS pointer
  //cout << "Entering born_callback..." << endl;
  Info *info = (Info *) ptr;
  LAMMPS *lmp = info->lmp;
  double *ef = info->efield;
  int *iatom = new int[nlocal];
  int *numneigh = new int[nlocal];
  int **neighbors = new int *[nlocal];
  double polarization[3]={0.0,0.0,0.0};
  // get access to atom properties from LAMMPS
  Atom *atom = lmp->atom;

  int *atom_type = atom->type;
  int *atom_molecule = atom->molecule;

  int ntypes = lmp->atom->ntypes;
  FixExternal *fix = info->fix;
  double *energy_per_atom = new double[nlocal];
  double *charges = atom->q;
  double energy = 0.0;
  // for each local atom, compute born charge and apply efield force

  // allocate a contiguous 3D array [nlocal][3][3] flattened to 1D
  double *born_charges_dummy = new double[nlocal * 3 * 3];

  for (int idx = 0; idx < nlocal * 3 * 3; ++idx) {
    born_charges_dummy[idx] = 0.0; // initialization
  }
  // Create a 3D view onto the flat buffer: [nlocal][3][3]
  double (*born_charges)[3][3] =
    reinterpret_cast<double (*)[3][3]>(born_charges_dummy);

  // BEC calculation, performance is almost the same as without driving script wuth  one nlocal loop
  for (size_t i=0; i<nlocal; i++){
    //cout << i << " " << numneigh[i] ;
    lammps_neighlist_element_neighbors(lmp,1,nlocal,iatom,numneigh,neighbors);
    for (int neigh=0; neigh<*numneigh; neigh++){
      int j = *neighbors[neigh];
      //cout << j << endl;
      int jtype = atom_type[j];
      
      // simple pairwise interaction to modify born charges
      for (int m=0; m<3; m++){
        for (int n=0; n<3; n++){
          born_charges[i][m][n] += 0.0001 * jtype; // dummy interaction effect
          //cout << "all good so far..." << endl;
        }
      }
    }
  }
  
  // born_charges_3d[i][j][k] is now valid and refers to born_charges[i*9 + j*3 + k]for (int i = 0; i < nlocal; i++) {
  for (int i = 0; i < nlocal; i++) {
    int itype = atom_type[i];
    //double born_charge = 0.001 * itype; // dummy born charge calculation
    double efield[3] = {0.0, 0.0, *ef}; // example electric field in z-direction

    // apply force due to electric field: F = qE
    f[i][0] = born_charges[i][0][0] * efield[0]+ born_charges[i][0][1] * efield[1]+ born_charges[i][0][2] * efield[2];
    f[i][1] = born_charges[i][1][0] * efield[1]+ born_charges[i][1][1] * efield[1]+ born_charges[i][1][2] * efield[2];
    f[i][2] = born_charges[i][2][0] * efield[2]+ born_charges[i][2][1] * efield[2]+ born_charges[i][2][2] * efield[2];
    energy_per_atom[i] =  charges[i]*(efield[0]*x[i][0] + efield[1]*x[i][1] + efield[2]*x[i][2]); // dummy energy per atom calculation
    energy +=  energy_per_atom[i];
    polarization[0] += charges[i]*x[i][0];
    polarization[1] += charges[i]*x[i][1];
    polarization[2] += charges[i]*x[i][2];
  }
  //cout << energy << endl;
  // cout << energy_per_atom[0] << endl;
  //cout << "Exiting born_callback..." << endl;
  //cout << born_charges[0][0] << endl;
  fix->set_energy_peratom(energy_per_atom);

  // Gather total energy across all procs
  double total_energy = 0.0;
  double total_polarization[3]={0.0,0.0,0.0};
  MPI_Allreduce(polarization, total_polarization, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&energy, &total_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  //cout << "Total energy from born charges: " << total_energy << endl;
  fix->set_energy_global(total_energy);
  /*
  if (info->me == 0){
    ofstream outfile ("output.txt");
    outfile << ntimestep << " "
            << total_energy << " "
            << total_polarization[0] << " "
            << total_polarization[1] << " "
            << total_polarization[2] << endl;
    //outfile.close();
  }
  */
  /*
   wo energy from born charges
         0   0.20346667    -16121.503      0             -16121.44      -662.80764    
       500   625.44607     -16121.718      0             -15927.77      -349.72812    
      1000   2498.435      -16121.512      0             -15346.759      590.38818   
   w energy from born charges, parallel
         0   0.20346667    -16121.503      0             -15253.463     -662.80764    
       500   625.44607     -16121.718      0             -15015.705     -349.72812    
      1000   2498.435      -16121.512      0             -14302.432      590.38818  
  */
  // We created new arrays with new, so we need to delete them here

  // Cleanup at end of function:
  //   delete [] numneigh;
  //   delete [] neighbors;
  // Do NOT delete neighbors[i]; you did not allocate those.
  //delete energy;
  delete [] energy_per_atom;
  delete [] iatom;
  delete [] numneigh;
  delete [] neighbors;
  /*
  for (int i = 0; i < nlocal; i++) {
    for (int j = 0; j < 3; j++) {
      delete [] born_charges[i][j];
    }
    delete [] born_charges[i];
  }
    */
  delete [] born_charges;
  //delete [] charges;
  //delete [] born_charges_3d;
}