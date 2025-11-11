
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
#include "neighbor.h"
#include "neigh_list.h"

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
  int nsteps;
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
  info.nsteps = nsteps;
  // Setting up output files
  // output velocities in 3 files for x,y,z components, 2 lines per timestep, one with indices, one with values
  // output dipole moment in one file, one line per timestep with 3 values
  // output "local dipole moment" in 3 files for x,y,z components, one line per timestep with N values (N = # of atoms)
  // output file name will be unique by adding efield value and # of steps to output file name

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
  int nghost= lammps_extract_setting(lmp,"nghost");
  double *efield = info->efield; // In current implementation, this is the interlayer voltage difference.
  double rcut=8.0; // Angstrom
  
  
  double polarization[3]={0.0,0.0,0.0};
  double charges_born[nlocal]={0.0}; // Could be a point of failure
  double pcharges_born[nlocal][3]={{0.0,0.0,0.0}}; // derivative of partial charges
  // get access to atom properties from LAMMPS
  Atom *atom = lmp->atom;
 
  //cout << lammps_extract_setting(lmp,"nghost") << endl;
  double beta=0.56166857; // Angstrom^-1
  double q=0.0;
  int *atom_type = atom->type;
  int *atom_molecule = atom->molecule;
  int ntypes = lmp->atom->ntypes;
  FixExternal *fix = info->fix;
  double *energy_per_atom=new double[nlocal];
  //double *charges = atom->q;
  double energy = 0.0;
  
  // Find the neighbor list index for the pair style
  int nlindex = lammps_find_pair_neighlist(lmp, "zero", 0, 0, 0);
  
  // Get the number of atoms in this neighbor list (inum = number of atoms with neighbors)
  int num_atoms_with_neighbors = lammps_neighlist_num_elements(lmp, nlindex);
  
  // Single loop over all atoms in the neighbor list
  for (int element = 0; element < num_atoms_with_neighbors; element++) {
    // Declare variables to receive the data for this ONE atom
    int iatom;           // Will hold the local index of the central atom
    int numneigh;        // Will hold the number of neighbors
    int *neighbors;      // Will hold pointer to the neighbor list
    
    // Get neighbor info for this one atom
    lammps_neighlist_element_neighbors(lmp, nlindex, element, &iatom, &numneigh, &neighbors);
    
    // Check if we got valid data
    if (iatom < 0 || neighbors == nullptr) continue;
    
    // Now iatom is the local index of the central atom
    int atom_i_local_index = iatom;
    int atom_i_type = atom->type[atom_i_local_index];
    int atom_i_molecule = atom->molecule[atom_i_local_index];
    
    // Determine q value based on atom type
    if (atom_i_type == 1) {
      q = -8.11256518;
    } else if (atom_i_type == 2) {
      q = 8.11256518;
    } else {
      printf("ERROR: Unknown atom type for born charge calculation\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Determine voltage difference based on which layer the atom belongs to
    double dV = (atom_i_molecule == 1) ? (*efield) / 2.0 : -(*efield) / 2.0;
    
    // Loop over all neighbors of this atom to calculate Born charges
    for (int j = 0; j < numneigh; j++) {
      int neighbor_j_local_index = neighbors[j];
      /*
      if (neighbor_j_local_index > nlocal){
        cout << "Ghost atom neighbor index: " << neighbor_j_local_index << endl;
      }
      */
      // neighbor_j_local_index can be either:
      // - A local atom: 0 <= neighbor_j_local_index < nlocal
      // - A ghost atom: nlocal <= neighbor_j_local_index < nlocal + nghost
      // Both cases are handled correctly since atom->type, atom->molecule, and x 
      // are all sized [nlocal + nghost]
      
      // Only calculate for interlayer interactions
      if ((atom_i_molecule != atom->molecule[neighbor_j_local_index]) and (atom_i_type != atom->type[neighbor_j_local_index])) {
        // Calculate distance vector and magnitude
        // dx points from j to i (important for sign convention)
        double dx = x[atom_i_local_index][0] - x[neighbor_j_local_index][0];
        double dy = x[atom_i_local_index][1] - x[neighbor_j_local_index][1];
        double dz = x[atom_i_local_index][2] - x[neighbor_j_local_index][2];
        double r = sqrt(dx*dx + dy*dy + dz*dz);
        
        // Calculate tapering function and its derivative
        double r_rc = r / rcut;
        double Tap = 20*pow(r_rc, 7) - 70*pow(r_rc, 6) + 84*pow(r_rc, 5) - 35*pow(r_rc, 4) + 1;
        double dTap = (140*pow(r_rc, 6) - 420*pow(r_rc, 5) + 420*pow(r_rc, 4) - 140*pow(r_rc, 3)) / rcut;
        
        // Calculate exponential term
        double exp_term = exp(-beta * r);
        
        // Determine q for neighbor atom
        /*
        double q_neighbor;
        int neighbor_j_type = atom->type[neighbor_j_local_index];
        if (neighbor_j_type == 1) {
          q_neighbor = -8.11256518;
        } else if (neighbor_j_type == 2) {
          q_neighbor = 8.11256518;
        } else {
          q_neighbor = 0.0; // Should not happen, but safe fallback
        }
        */
        // Accumulate Born effective charge for atom i
        // This represents the charge induced on atom i due to neighbor j
        double charge_contribution = q * exp_term * Tap;
        charges_born[atom_i_local_index] += charge_contribution;
        
        // Accumulate derivative of partial charges (for forces on atom i)
        // These are derivatives with respect to atom i's position
        double common_factor = q * exp_term;
        double dq_dx = common_factor * (-beta * Tap * dx/r + dTap * dx/r);
        double dq_dy = common_factor * (-beta * Tap * dy/r + dTap * dy/r);
        double dq_dz = common_factor * (-beta * Tap * dz/r + dTap * dz/r);
        
        pcharges_born[atom_i_local_index][0] += dq_dx;
        pcharges_born[atom_i_local_index][1] += dq_dy;
        pcharges_born[atom_i_local_index][2] += dq_dz;
        
        // Newton's third law: if neighbor j is also a local atom, accumulate its contributions
        // This ensures proper force balance and avoids double-counting in parallel simulations
        /*
        if (neighbor_j_local_index < nlocal) {
          // Neighbor j is local, so we also update its Born charges
          // The contribution to j is opposite in sign for the charge derivative
          charges_born[neighbor_j_local_index] -= charge_contribution;
          
          // Forces on j are opposite to forces on i (Newton's 3rd law)
          pcharges_born[neighbor_j_local_index][0] -= dq_dx;
          pcharges_born[neighbor_j_local_index][1] -= dq_dy;
          pcharges_born[neighbor_j_local_index][2] -= dq_dz;
        }
        */  
        // If neighbor_j_local_index >= nlocal, it's a ghost atom, and its owning processor
        // will compute its contributions, so we don't accumulate here
      }
    }
    
    // Now that we've accumulated all neighbor contributions, apply forces and calculate energies
    // Apply force due to electric field: F = -dq/dr * V
    f[atom_i_local_index][0] = pcharges_born[atom_i_local_index][0] * dV;
    f[atom_i_local_index][1] = pcharges_born[atom_i_local_index][1] * dV;
    f[atom_i_local_index][2] = 0.0;  // pcharges_born[atom_i_local_index][2] * dV;
    
    // Calculate energy per atom: E = q * V
    energy_per_atom[atom_i_local_index] = charges_born[atom_i_local_index] * dV;
    energy += energy_per_atom[atom_i_local_index];
    
    // Calculate polarization contributions
    polarization[0] += charges_born[atom_i_local_index] * x[atom_i_local_index][0];
    polarization[1] += charges_born[atom_i_local_index] * x[atom_i_local_index][1];
    polarization[2] += charges_born[atom_i_local_index] * x[atom_i_local_index][2];
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
  if (info->me == 0) {
    string fName="output_"+to_string(*efield)+"_"+to_string(info->nsteps)+".txt";
    if (ntimestep == 0) {
      ofstream outfile (fName, ios::trunc);
      outfile << "#Timestep Total_Energy Polarization_X Polarization_Y Polarization_Z" << endl;
      outfile << ntimestep << " "
            << total_energy << " "
            << total_polarization[0] << " "
            << total_polarization[1] << " "
            << total_polarization[2] << endl;
      outfile.close();

    }else{
    ofstream outfile (fName, ios::app);
    outfile << ntimestep << " "
            << total_energy << " "
            << total_polarization[0] << " "
            << total_polarization[1] << " "
            << total_polarization[2] << endl;
    outfile.close();
    }
    
  }
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
  //delete [] iatom;
  //delete [] numneigh;
  //delete [] neighbors;
  /*
  for (int i = 0; i < nlocal; i++) {
    for (int j = 0; j < 3; j++) {
      delete [] born_charges[i][j];
    }
    delete [] born_charges[i];
  }
    */
  //delete [] born_charges;
  //delete [] charges_born;
  //delete [] pcharges_born;
  //delete [] charges;
  //delete [] born_charges_3d;
}