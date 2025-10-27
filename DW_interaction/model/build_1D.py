import numpy as np
import matplotlib.pyplot as plt

def build_1D(Nx,Ny,dir,defo,fileName,plot=False):

    a=2.511
    c=6.6612/2

    pos_b=np.zeros((4*Nx*Ny,2))
    pos_t=np.zeros((4*Nx*Ny,2))

    v1=a*np.array([np.sqrt(3)/2,1/2])
    v2=a*np.array([np.sqrt(3)/2,-1/2])

    Lx=np.linalg.norm(v1+v2)
    Ly=np.linalg.norm(v1-v2)
    #print(Lx)
    #print(Ly)
    #print(Lx,Ly)
    for i in range(Nx):
        for j in range(Ny):
            pos_b[4*(i+j*Nx)]=np.array([i*Lx,j*Ly])
            pos_b[4*(i+j*Nx)+2]=np.array([i*Lx,j*Ly])+v1
            pos_b[4*(i+j*Nx)+1]=np.array([(i-1/3)*Lx,j*Ly])+v1
            pos_b[4*(i+j*Nx)+3]=np.array([(i-1/3)*Lx,j*Ly])+v1+v2

    # Verification of the positions
    if plot:
        plt.figure()
        plt.plot(pos_b[::4,0],pos_b[::4,1],'ro')
        plt.plot(pos_b[1::4,0],pos_b[1::4,1],'bo')
        plt.plot(pos_b[2::4,0],pos_b[2::4,1],'r.')
        plt.plot(pos_b[3::4,0],pos_b[3::4,1],'b.')
        plt.axis('equal')
    

    # Initial deformation of top layer (with respect to AA stacking)
    if dir=='0' or dir=='60': # zigzag
        N1=Ny
        N2=Nx
    elif dir=='30' or dir=='90': # armchair
        N1=Nx
        N2=Ny
    deformation=np.zeros((4*Nx*Ny,2))
    
    #def_x=defo*dir*Lx/3*(1-np.abs(np.linspace(-1,1,N1)))
    #def_y=defo*(1-dir)*Ly/2*(1-np.abs(np.linspace(-1,1,N1)))
    if dir=='0' or dir=='90':
        def_y=np.zeros(N1)
        def_x=defo*(1-np.abs(np.linspace(-1,1,N1)))*a*np.sqrt(3)/3
    elif dir=='60' or dir =='30':
        def_x=-defo*(1-np.abs(np.linspace(-1,1,N1)))*a*np.sqrt(3)/3*np.cos(np.pi/3)
        def_y=defo*(1-np.abs(np.linspace(-1,1,N1)))*a*np.sqrt(3)/3*np.sin(np.pi/3)
    if dir=='30':
        def_x=-defo*(1-np.abs(np.linspace(-1,1,N1)))*a*np.sqrt(3)/3*np.cos(np.pi/3)
        def_y=defo*(1-np.abs(np.linspace(-1,1,N1)))*a*np.sqrt(3)/3*np.sin(np.pi/3)
        

    #def_y=np.concatenate((np.linspace(0,Ly,round(Nx/2)),np.linspace(Ly,0,round(Nx/2))))*(1-dir)
    for i in range(Nx):
        for j in range(Ny):
            if dir=='0' or dir=='60':
                k=j
            elif dir=='30' or dir=='90':
                k=i
            # The Lx/3 factor is for pure AB stacking
            deformation[4*(i+j*Nx)]=np.array([Lx/3+def_x[k],def_y[k]])
            deformation[4*(i+j*Nx)+1]=np.array([Lx/3+def_x[k],def_y[k]])
            deformation[4*(i+j*Nx)+2]=np.array([Lx/3+def_x[k],def_y[k]])
            deformation[4*(i+j*Nx)+3]=np.array([Lx/3+def_x[k],def_y[k]])

    pos_b=pos_b-deformation/2
    pos_t=pos_b+deformation
    # Verification of the positions
    if plot:
        plt.figure()
        plt.plot(pos_b[:,0],pos_b[:,1],'ro')
        plt.plot(pos_t[:,0],pos_t[:,1],'bo')
        plt.axis('equal')
    
    # Create lammps data file
    datdat=[""]
    datdat.append(f"\n{2*4*Nx*Ny} atoms \n2 atom types\n")
    datdat.append(f"0 {Nx*Lx} xlo xhi\n0 {Ny*Ly} ylo yhi\n-100 100 zlo zhi\n\n")
    #datdat.append(f"0 0 0 xy xz yz\n\n")
    datdat.append("Masses\n\n1 10.811\n2 14.007\n\n")
    datdat.append("Atoms # full\n\n")
    for i in range(len(pos_b)):
        datdat.append(f"{i+1} 1 {i%2+1} 0 {pos_b[i,0]} {pos_b[i,1]} 50.0\n")
        datdat.append(f"{len(pos_b)+i+1} 2 {i%2+1} 0 {pos_t[i,0]} {pos_t[i,1]} {50+c}\n")
    dat = open(fileName,'w')
    dat.writelines(datdat)
    dat.close()
    #plt.show()
    return fileName

def build_1D_deformation(Nx,Ny,dir,def_f,params,fileName):
    a=2.511
    c=6.6612/2

    pos_b=np.zeros((4*Nx*Ny,2))
    pos_t=np.zeros((4*Nx*Ny,2))

    v1=a*np.array([np.sqrt(3)/2,1/2])
    v2=a*np.array([np.sqrt(3)/2,-1/2])

    Lx=np.linalg.norm(v1+v2)
    Ly=np.linalg.norm(v1-v2)
    #print(Lx)
    #print(Ly)
    #print(Lx,Ly)
    for i in range(Nx):
        for j in range(Ny):
            pos_b[4*(i+j*Nx)]=np.array([i*Lx,j*Ly])
            pos_b[4*(i+j*Nx)+2]=np.array([i*Lx,j*Ly])+v1
            pos_b[4*(i+j*Nx)+1]=np.array([(i-1/3)*Lx,j*Ly])+v1
            pos_b[4*(i+j*Nx)+3]=np.array([(i-1/3)*Lx,j*Ly])+v1+v2

    # Verification of the positions
    '''
    if plot:
        plt.figure()
        plt.plot(pos_b[::4,0],pos_b[::4,1],'ro')
        plt.plot(pos_b[1::4,0],pos_b[1::4,1],'bo')
        plt.plot(pos_b[2::4,0],pos_b[2::4,1],'r.')
        plt.plot(pos_b[3::4,0],pos_b[3::4,1],'b.')
        plt.axis('equal')
    '''

    # Initial deformation of top layer (with respect to AA stacking)
    if dir=='0' or dir=='60': # zigzag
        N1=Ny
        N2=Nx
    elif dir=='30' or dir=='90': # armchair
        N1=Nx
        N2=Ny
    deformation=np.zeros((4*Nx*Ny,2))
    '''
    #def_x=defo*dir*Lx/3*(1-np.abs(np.linspace(-1,1,N1)))
    #def_y=defo*(1-dir)*Ly/2*(1-np.abs(np.linspace(-1,1,N1)))
    if dir=='0' or dir=='90':
        def_y=np.zeros(N1)
        def_x=defo*(1-np.abs(np.linspace(-1,1,N1)))*a*np.sqrt(3)/3
    elif dir=='60' or dir =='30':
        def_x=-defo*(1-np.abs(np.linspace(-1,1,N1)))*a*np.sqrt(3)/3*np.cos(np.pi/3)
        def_y=defo*(1-np.abs(np.linspace(-1,1,N1)))*a*np.sqrt(3)/3*np.sin(np.pi/3)
    if dir=='30':
        def_x=-defo*(1-np.abs(np.linspace(-1,1,N1)))*a*np.sqrt(3)/3*np.cos(np.pi/3)
        def_y=defo*(1-np.abs(np.linspace(-1,1,N1)))*a*np.sqrt(3)/3*np.sin(np.pi/3)
    '''   

    #def_y=np.concatenate((np.linspace(0,Ly,round(Nx/2)),np.linspace(Ly,0,round(Nx/2))))*(1-dir)
    for i in range(Nx):
        for j in range(Ny):
            if dir=='0' or dir=='60':
                k=j
            elif dir=='30' or dir=='90':
                k=i
            # The Lx/3 factor is for pure AB stacking
            deformation[4*(i+j*Nx)]=def_f(i,j,*params)#np.array([Lx/3+def_x[k],def_y[k]])
            deformation[4*(i+j*Nx)+1]=def_f(i,j,*params)#np.array([Lx/3+def_x[k],def_y[k]])
            deformation[4*(i+j*Nx)+2]=def_f(i,j,*params)#np.array([Lx/3+def_x[k],def_y[k]])
            deformation[4*(i+j*Nx)+3]=def_f(i,j,*params)#np.array([Lx/3+def_x[k],def_y[k]])

    pos_b=pos_b-deformation/2
    pos_t=pos_b+deformation
    # Verification of the positions
    '''
    if plot:
        plt.figure()
        plt.plot(pos_b[:,0],pos_b[:,1],'ro')
        plt.plot(pos_t[:,0],pos_t[:,1],'bo')
        plt.axis('equal')
    '''
    # Create lammps data file
    datdat=[""]
    datdat.append(f"\n{2*4*Nx*Ny} atoms \n2 atom types\n")
    datdat.append(f"0 {Nx*Lx} xlo xhi\n0 {Ny*Ly} ylo yhi\n-100 100 zlo zhi\n\n")
    #datdat.append(f"0 0 0 xy xz yz\n\n")
    datdat.append("Masses\n\n1 10.811\n2 14.007\n\n")
    datdat.append("Atoms # full\n\n")
    for i in range(len(pos_b)):
        datdat.append(f"{i+1} 1 {i%2+1} 0 {pos_b[i,0]} {pos_b[i,1]} 50.0\n")
        datdat.append(f"{len(pos_b)+i+1} 2 {i%2+1} 0 {pos_t[i,0]} {pos_t[i,1]} {50+c}\n")
    dat = open(fileName,'w')
    dat.writelines(datdat)
    dat.close()
    #plt.show()
    return fileName


def build_1D_charges(atoms,charges,fileName):
    # Create lammps data file
    datdat=[""]
    datdat.append(f"\n{len(atoms)} atoms \n2 atom types\n")
    datdat.append(f"0 {atoms.cell[0][0]} xlo xhi\n0 {atoms.cell[1][1]} ylo yhi\n-100 100 zlo zhi\n\n")
    #datdat.append(f"0 0 0 xy xz yz\n\n")
    datdat.append("Masses\n\n1 10.811\n2 14.007\n\n")
    datdat.append("Atoms # full\n\n")
    for i in range(len(atoms)):
        datdat.append(f"{i+1} {atoms.get_array('mol-id')[i]} {i%2+1} {charges[i]} {atoms.get_positions()[i,0]} {atoms.get_positions()[i,1]} {atoms.get_positions()[i,2]}\n")

    dat = open(fileName,'w')
    dat.writelines(datdat)
    dat.close()
    #plt.show()
    return fileName