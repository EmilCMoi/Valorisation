from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida.engine import submit
import aiida.orm as orm
from aiida_quantumespresso.common.types import ElectronicType
import numpy as np
from aiida.orm import StructureData
from ase import Atoms
from ase.visualize import view
# First step is to verify ibrav
# Ph part seems to work but it needs a parent folder, meaning a scf calculation first
def launch_pw_workchain(structure):
    builder = PwBaseWorkChain.get_builder_from_protocol(
        code=orm.load_node(4873),  # Replace with your actual code node
        structure=structure,
        electronic_type=ElectronicType.INSULATOR,
        overrides={
                'pseudo_family': 'PseudoDojo/0.4/LDA/SR/standard/upf'  }
    ) 
    print(builder)
    #pseudo_family = "PseudoDojo/0.4/LDA/SR/standard/upf"
    #builder.pw.pseudo_family = pseudo_family
    builder.pw.parameters['SYSTEM']['ibrav'] =4
    #builder.pw.parameters['SYSTEM']['vdw_corr'] = 'DFT-D'
    builder.pw.parameters['SYSTEM']['assume_isolated'] = '2D'
    
    builder.pw.metadata.options.resources = {
        'num_machines': 1,
        'num_mpiprocs_per_machine': 4
    }
    builder.pw.metadata.options.max_wallclock_seconds = 6 * 60  # 6 minutes
    builder.pw.metadata.options.parser_name = 'quantumespresso.pw'
    kpoints_mesh = [12, 12, 1]
    kpoints=orm.KpointsData()
    kpoints.set_kpoints_mesh(kpoints_mesh)
    builder.kpoints = kpoints
    workchain = submit(builder)
    print(f"Submitted {workchain.process_label}<{workchain.pk}>")

def launch_ph_workchain(pwWorkchain):
    builder = PhBaseWorkChain.get_builder_from_protocol(
        #pw_code=orm.load_node(4873),
        code=orm.load_node(60429),
        #structure=structure,
        electronic_type=ElectronicType.INSULATOR,
        parent_folder=orm.load_node(pwWorkchain).outputs.remote_folder,
        #pseudo_family="PseudoDojo/0.4/LDA/SR/standard/upf"
    )
    print(builder)
    
    qpoints_mesh = [1,1,1]  # Gamma point
    qpoints=orm.KpointsData()
    qpoints.set_kpoints_mesh(qpoints_mesh)
    builder.qpoints = qpoints

    builder.ph.parameters['INPUTPH']['epsil']= True
    builder.ph.parameters['INPUTPH']['tr2_ph']= 1.0e-12

    builder.ph.metadata.options.resources = {
        'num_machines': 1,
        'num_mpiprocs_per_machine': 4
    }
    builder.ph.metadata.options.max_wallclock_seconds = 20 * 60  # 20 minutes
    builder.ph.metadata.options.parser_name = 'quantumespresso.ph'
    builder.ph.metadata.label = 'PH for Born charge training'
    
 

    workchain = submit(builder)
    print(f"Submitted {workchain.process_label}<{workchain.pk}>")

def create_structure(r):
    a = 2.511
    c = 6.6612 / 2
    v1 = a*np.array([1,0,0])
    v2 = a*np.array([-1/2,np.sqrt(3)/2,0])
    
    # Adjusting for the convention of ibrav=4:
    tmp=r[0]
    r[0]=r[1]
    r[1]=-tmp
    # Create positions based on the input r
    positions = np.zeros((4, 3))
    positions[0] = np.array([0, 0, 0])
    positions[1] = np.array([0, 0, 0]) + v1 / 3 - v2 / 3
    positions[2] = np.array([0, 0, c]) + r[0] * v1 + r[1] * v2 + r[2]
    positions[3] = np.array([0, 0, c]) + v1 / 3 - v2 / 3 + r[0] * v1 + r[1] * v2 + r[2]
    
    labels = ["B", "N", "B", "N"]
    atoms = StructureData()
    struct=Atoms(symbols=labels, positions=positions,cell=[v1, v2, [0, 0, 30]])
    
    struct.set_pbc([True, True, False])
    struct.wrap()
    #struct.center()
    #struct.wrap()
    
    #view(struct)
    atoms.set_ase(struct)
    
    return atoms
#struct=create_structure([1/2,1/2, 0])
'''
group = orm.load_group(34)
N=12
structs=np.zeros(N*N,dtype=int)
rs=np.zeros((N*N,2))
for i in range(N):
    for j in range(N):
            r = [i/N, j/N, 0]
            print(f"Creating structure for r={r}")
            struct = create_structure(r)
            print(r)
            rs[i*N+j]=r[:2]
            print(r)
            #struct.store()
            #structs[i*N+j]=struct.store().pk
            #group.add_nodes(struct)
            
            #launch_pw_workchain(struct)
#view(struct.get_ase())
#print(struct.get_ase())
#np.savez("structures_rs.npz",structs=structs,rs=rs)
'''
qb=orm.QueryBuilder()
qb.append(orm.Group, filters={'id':34}, tag='group')
qb.append(StructureData, with_group='group', tag='structure')
qb.append(PwBaseWorkChain, with_incoming='structure', tag='pw_workchain',filters={'attributes.exit_status':{"==":0}})
group= orm.load_group(34)
print(len(qb.all()))
for i in range(len(qb.all())//3):
    pwWorkchain=qb.all()[i][0].pk
    #launch_ph_workchain(pwWorkchain)

#struct=create_structure([1/3, 1/3, 0])
#print(struct.cell)
#launch_pw_workchain(struct)
#launch_ph_workchain(60704)