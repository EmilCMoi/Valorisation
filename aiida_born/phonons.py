import numpy as np
from aiida import orm
from aiida_quantumespresso.calculations.ph import PhCalculation
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida.orm import StructureData
from aiida.orm import RemoteData
import matplotlib.pyplot as plt
from aiida.common.exceptions import NotExistentAttributeError
from draw import cmaps#..old.model.draw import cmaps

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'sans-serif'})
#plt.rcParams['figure.constrained_layout.use'] = True

Lflep, Lflep_r, Dflep, Dflep_r = cmaps()

qb_ph=orm.QueryBuilder()
qb_ph.append(orm.Group, filters={'id': 29}, tag='group')
qb_ph.append(StructureData, with_group='group', tag='structure',project='*')
qb_ph.append(PwBaseWorkChain, with_incoming='structure', tag='pw_workchain', filters={'attributes.exit_status':{"==":0}})
qb_ph.append(RemoteData, with_incoming='pw_workchain', tag='pw_remote')
qb_ph.append(PhCalculation, with_incoming='pw_remote', tag='ph_workchain',project='*')
#qb_ph.append(StructureData, with_outgoing='group', tag='structure2',project='*')

'''
qb_pw=orm.QueryBuilder()
qb_pw.append(orm.Group, filters={'id': 29}, tag='group')
qb_pw.append(StructureData, with_group='group', tag='structure')
qb_pw.append(PwBaseWorkChain, with_incoming='structure', tag='pw_workchain', filters={'attributes.exit_status':{"==":0}})
'''
group= orm.load_group(29)
print(len(qb_ph.all()))
#print(qb_ph.all())
rs=np.zeros((len(qb_ph.all()),2))
omegas=np.zeros((len(qb_ph.all()),12))
count=0
bl=[]
nerrors=0
errorpks=[]
which_to_read=[]
for structure, ph in qb_ph.all():
    #print(count)
    print(ph.pk)
    dict_ph=ph.outputs.output_parameters.get_dict()
    #print(ph.pk)
    #print(dict_ph['done_effective_charge_eu'])
    struct=structure.get_ase()
    r=(struct.positions[2]-struct.positions[0])[:2]
    #print(r)
    #assert dict_ph["done_effective_charge_eu"]
    try:
        
        oms=dict_ph["dynamical_matrix_1"]['frequencies']
        which_to_read.append(True)
    except (NotExistentAttributeError, KeyError):
        oms=np.zeros(12)
        
        which_to_read.append(False)
        nerrors+=1
        errorpks.append(ph.pk)
    count+=1
    rs[count-1]=r
    omegas[count-1]=oms#.reshape((4,3,3))
    bl.append(struct)

plt.figure()
plt.scatter(rs[:,0], rs[:,1], c=omegas[:,0], cmap=Lflep_r, s=100)
plt.colorbar()
plt.xlabel('r1')
plt.ylabel('r2')
plt.title('Dynamical Frequencies')
plt.show()
print(errorpks)