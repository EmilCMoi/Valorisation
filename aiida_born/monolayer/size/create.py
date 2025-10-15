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
    builder.pw.metadata.options.max_wallclock_seconds = 10 * 60  # 10 minutes
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
    builder.ph.metadata.options.max_wallclock_seconds = 5 * 60  # 15 minutes
    builder.ph.metadata.options.parser_name = 'quantumespresso.ph'
    builder.ph.metadata.label = 'PH for Born charge training'
    
 

    workchain = submit(builder)
    print(f"Submitted {workchain.process_label}<{workchain.pk}>")

def create_structure(r): # r is lattice constant
    a = r
    v1 = a*np.array([1,0,0])
    v2 = a*np.array([-1/2,np.sqrt(3)/2,0])
    
    # Adjusting for the convention of ibrav=4:
    #tmp=r[0]
    #r[0]=r[1]
    #r[1]=-tmp
    # Create positions based on the input r
    positions = np.zeros((2, 3))
    positions[0] = np.array([0, 0, 0])
    positions[1] = np.array([0, 0, 0]) + v1 / 3 - v2 / 3
    
    
    labels = ["B", "N"]
    atoms = StructureData()
    struct=Atoms(symbols=labels, positions=positions,cell=[v1, v2, [0, 0, 30]])
    
    struct.wrap()
    struct.center()
    struct.wrap()
    #view(struct)
    atoms.set_ase(struct)
    
    return atoms
#struct=create_structure([1/2,1/2, 0])
'''
np.random.seed(12345)
group = orm.load_group(32)
N=15
# 2.1, 2.9, N=21
# 1.5, 2.1, N=15
rs=np.linspace(4.1,3.5,N,endpoint=False)
print(rs)

for i in range(N):
            r = rs[i]
            print(f"Creating structure for r={r}")
            struct = create_structure(r)
            #struct.store()
            #group.add_nodes(struct)
            
            #launch_pw_workchain(struct)
#view(struct.get_ase())

'''
qb=orm.QueryBuilder()
qb.append(orm.Group, filters={'id': 32}, tag='group')
qb.append(StructureData, with_group='group', tag='structure',project='*')
qb.append(PwBaseWorkChain, with_incoming='structure', tag='pw_workchain',filters={'attributes.exit_status':{"==":0}},project='*')
group= orm.load_group(32)
print(len(qb.all()))
for i in range(len(qb.all())):
    if np.linalg.norm(qb.all()[i][0].get_ase().cell[0])>3.5:
        pwWorkchain=qb.all()[i][1].pk
        print(i,np.linalg.norm(qb.all()[i][0].get_ase().cell[0]))
        launch_ph_workchain(pwWorkchain)
    #pwWorkchain=qb.all()[i][1].pk
    #launch_ph_workchain(pwWorkchain)

#struct=create_structure([1/3, 1/3, 0])
#print(struct.cell)
#launch_pw_workchain(struct)
#launch_ph_workchain(60704)