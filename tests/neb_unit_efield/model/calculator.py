from lammps import lammps as lammps_engine
from ase.calculators.lammps import Prism, convert
from ase.calculators.calculator import Calculator
import numpy as np
import ctypes

from model.born import KDborn, born_charges


class LAMMPS(Calculator):

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, log_file="lammps.log", **kwargs):

        super().__init__(**kwargs)
        cmd_args = ["-echo", "log", "-log", log_file, "-screen", "none", "-nocite"]
        self.lmp = lammps_engine(cmdargs=cmd_args)
        self.units = "metal"
        self.init = False

    def init_atoms(self, atoms):

        self.atoms = atoms
        self.prism = Prism(atoms.cell, atoms.pbc)
        atoms.calc = self

        atoms.write("input.tmp", format="lammps-data", atom_style="full", masses=True)

        self.lmp.command("units metal")
        self.lmp.command("atom_style full")

        pbc = ["p" if pb else "f" for pb in atoms.pbc]
        self.lmp.command(f"boundary {' '.join(pbc)}")
        self.lmp.command("read_data input.tmp")

        self.lmp.command("group top molecule 1")
        self.lmp.command("group bottom molecule 2")
        self.lmp.command("group graphene molecule 3")
         #Only BN
        
        self.lmp.command(
            "pair_style hybrid/overlay tersoff ilp/graphene/hbn 16.0 1 coul/shield 16.0"
        )
        self.lmp.command("pair_coeff * * tersoff model/BNC.tersoff B N")
        self.lmp.command("pair_coeff * * ilp/graphene/hbn model/BNCH.ILP B N")

        self.lmp.command("pair_coeff 1 1 coul/shield 0.70")
        self.lmp.command("pair_coeff 1 2 coul/shield 0.69498201415576216335")
        self.lmp.command("pair_coeff 2 2 coul/shield 0.69")
        '''
        # BNC
        self.lmp.command("pair_style  hybrid/overlay tersoff ilp/graphene/hbn/opt 16.0 1 coul/shield 16.0")
        self.lmp.command("pair_coeff * * tersoff model/BNC.tersoff B C N")
        self.lmp.command("pair_coeff * * ilp/graphene/hbn/opt model/BNCH.ILP B C N")
        self.lmp.command("pair_coeff 1 1 coul/shield 0.70")
        self.lmp.command("pair_coeff 1 3 coul/shield 0.69498201415576216335")
        self.lmp.command("pair_coeff 3 3 coul/shield 0.69")
        '''

        self.lmp.command("variable pxx equal pxx")
        self.lmp.command("variable pyy equal pyy")
        self.lmp.command("variable pzz equal pzz")
        self.lmp.command("variable pxy equal pxy")
        self.lmp.command("variable pxz equal pxz")
        self.lmp.command("variable pyz equal pyz")

        self.lmp.command("variable pe equal pe")

        self.lmp.command("variable fx atom fx")
        self.lmp.command("variable fy atom fy")
        self.lmp.command("variable fz atom fz")

        self.lmp.command("neigh_modify one 10000")

        self.lmp.command(f"run 0")

        self.init = True

    def calculate(self, atoms, properties, system_changes):

        if not self.init:
            self.init_atoms(atoms)
        '''
        if "cell" in system_changes:
            self.prism = Prism(atoms.cell, atoms.pbc)
            lammps_prism = self.prism.get_lammps_prism()
            xhi, yhi, zhi, xy, xz, yz = convert(
                lammps_prism, "distance", "ASE", self.units
            )
            cell_cmd = (
                f"change_box all     "
                f"x final 0 {xhi} y final 0 {yhi} z final 0 {zhi}      "
                f"xy final {xy} xz final {xz} yz final {yz} units box"
            )
            self.lmp.command(cell_cmd)
        '''
        if "positions" in system_changes:
            # Update the positions in LAMMPS
            self.set_lammps_pos(atoms)

        if len(system_changes) != 0:
            self.lmp.command(f"run 0")

        if "energy" in properties:
            # Extract the forces and energy
            field_energy=np.sum(atoms.get_array("voltage")*atoms.get_array("charges_2"))
            #for i in range(len(atoms)):
            #    field_energy+=atoms.get_array("charges_2")[i]*atoms.get_array("voltage")[i]
            self.results["energy"] = self.lmp.extract_variable("pe", None, 0)+field_energy

        if "stress" in properties:
            stress = np.empty(6)
            stress_vars = ["pxx", "pyy", "pzz", "pyz", "pxz", "pxy"]

            for i, var in enumerate(stress_vars):
                stress[i] = self.lmp.extract_variable(var, None, 0)

            stress_mat = np.zeros((3, 3))
            stress_mat[0, 0] = stress[0]
            stress_mat[1, 1] = stress[1]
            stress_mat[2, 2] = stress[2]
            stress_mat[1, 2] = stress[3]
            stress_mat[2, 1] = stress[3]
            stress_mat[0, 2] = stress[4]
            stress_mat[2, 0] = stress[4]
            stress_mat[0, 1] = stress[5]
            stress_mat[1, 0] = stress[5]

            stress_mat = self.prism.tensor2_to_ase(stress_mat)

            stress[0] = stress_mat[0, 0]
            stress[1] = stress_mat[1, 1]
            stress[2] = stress_mat[2, 2]
            stress[3] = stress_mat[1, 2]
            stress[4] = stress_mat[0, 2]
            stress[5] = stress_mat[0, 1]

            self.results["stress"] = convert(-stress, "pressure", self.units, "ASE")

        if "forces" in properties:
            f = np.array(self.lmp.gather_atoms("f", 1, 3)).reshape(-1, 3)
            #f-=atoms.get_array("born")*atoms.get_array("voltage")
            for i in range(len(atoms)):
                f[i,:2]-= atoms.get_array("born")[i,:2]*atoms.get_array("voltage")[i]
                #f[i]-= atoms.get_array("born")[i]*atoms.get_array("voltage")[i]

            #print(f)
            self.results["forces"] = self.prism.vector_to_ase(f)

        #born, charges, dcharges=KDborn(atoms)
        born, charges, dcharges = born_charges(atoms)
        atoms.set_initial_charges(charges)
        atoms.set_array("born", born)
        atoms.set_array("charges_2", dcharges)
        atoms.set_array("charges_model", dcharges)

        self.atoms = atoms.copy()

    def set_lammps_pos(self, atoms):
        pos = self.prism.vector_to_lammps(atoms.positions, wrap=True)
        lmp_positions = list(pos.ravel())
        c_double_array = ctypes.c_double * len(lmp_positions)
        lmp_c_positions = c_double_array(*lmp_positions)
        self.lmp.scatter_atoms("x", 1, 3, lmp_c_positions)
