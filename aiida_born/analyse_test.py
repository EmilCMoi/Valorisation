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
borns=np.zeros((len(qb_ph.all()),4,3,3))
count=0
bl=[]
nerrors=0
errorpks=[]
which_to_read=[]
for structure, ph in qb_ph.all():
    #print(count)
    dict_ph=ph.outputs.output_parameters.get_dict()
    #print(ph.pk)
    #print(dict_ph['done_effective_charge_eu'])
    struct=structure.get_ase()
    r=(struct.positions[2]-struct.positions[0])[:2]
    #print(r)
    #assert dict_ph["done_effective_charge_eu"]
    try:
        
        born=dict_ph["effective_charges_eu"]
        which_to_read.append(True)
    except (NotExistentAttributeError, KeyError):
        born=np.zeros((4,3,3))
        #print("No effective charges found, using zeros")
        #print(ph.pk)
        which_to_read.append(False)
        nerrors+=1
        errorpks.append(ph.pk)
    count+=1
    rs[count-1]=r
    borns[count-1]=born#.reshape((4,3,3))
    bl.append(struct)
from ase.visualize import view
print(which_to_read)
#print(bl)
#view(bl[0])
#view(bl[-1])
#print(borns[0])
#print(qb_ph.all()[0][1].pk)
print("Number of errors: ", nerrors, " corresponding to: ",nerrors/len(qb_ph.all())*100, "% of the total")
print("Error handled by replacing with zeros")
print("Error calculations: ", errorpks,"\n")
print(rs)
print("Calculating periodic repetitions 1/2")
for i,r in enumerate(rs):
    v1=struct.cell[0][:2]
    v2=struct.cell[1][:2]
    if abs(r[1])<1e-1:
        rs=np.concatenate((rs,[r-v2]),axis=0)
        #print(np.shape(borns))
        #print(np.shape(borns[i]))
        borns=np.concatenate((borns,[borns[i]]),axis=0)
        which_to_read.append(which_to_read[i])
    #print(borns[i])
    asr=np.sum(borns[i],axis=0)
    #print("ASR for structure ", i, ": ", asr)
    borns[i]-=asr/4
print("Calculating periodic repetitions 2/2\n")
for i,r in enumerate(rs):
    v1=struct.cell[0][:2]
    v2=struct.cell[1][:2]
    print(r[0]+r[1]*1/2)
    if abs(r[0]+r[1]*1/2)<2e-1:
        rs=np.concatenate((rs,[r+v1]),axis=0)
        borns=np.concatenate((borns,[borns[i]]),axis=0)
        which_to_read.append(which_to_read[i])
rs=rs[which_to_read]
borns=borns[which_to_read]
for i in range(4):
    plt.figure()
    b=borns[:,i,2,:2] # 3,xy components
    #print(len(b))
    C=np.linalg.norm(b,axis=1)
    plt.quiver(rs[:,0],rs[:,1],b[:,0]/C,b[:,1]/C,C,cmap=Dflep,pivot='middle')
    plt.colorbar()
    #print("before")
    plt.axis('equal')

    plt.figure()
    plt.scatter(rs[:,0],rs[:,1],c=borns[:,i,2,2],cmap=Dflep,s=200)
    plt.colorbar()
    plt.axis('equal')
for i in range(2):
    plt.figure()
    b=borns[:,2*i,2,:2] + borns[:,2*i+1,2,:2]
    C=np.linalg.norm(b,axis=1)
    plt.quiver(rs[:,0],rs[:,1],b[:,0]/C,b[:,1]/C,C,cmap=Dflep,pivot='middle')
    plt.colorbar()
    plt.axis('equal')

    plt.figure()
    plt.scatter(rs[:,0],rs[:,1],c=borns[:,2*i,2,2] + borns[:,2*i+1,2,2],cmap=Dflep,s=200)
    plt.colorbar()
    plt.axis('equal')

plt.show()

np.savez("borns.npz", rs=rs, borns=borns, which_to_read=which_to_read)
#print("after")

#print(rs)
#print(np.unique(rs))
#print(count)