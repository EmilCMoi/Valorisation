import numpy as np
import matplotlib.pyplot as plt

# Collision of different kinds of solitons in bilayer hBN

def double_build_1D(Nx,Ny,dir,defo,fileName,plot=False):

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
    '''
    if dir=='0' or dir=='60': # zigzag
        N1=Ny
        N2=Nx
    elif dir=='30' or dir=='90': # armchair
        N1=Nx
        N2=Ny
    '''
    if dir=='zigzag': # zigzag
        N1=Ny
        N2=Nx
    elif dir=='armchair':
        N1=Nx
        N2=Ny
    
    deformation=np.zeros((4*Nx*Ny,2))
    
    #def_x=defo*dir*Lx/3*(1-np.abs(np.linspace(-1,1,N1)))
    #def_y=defo*(1-dir)*Ly/2*(1-np.abs(np.linspace(-1,1,N1)))
    '''
    if dir=='0' or dir=='90':
        def_y=np.zeros(N1)
        def_x=defo*(1-np.abs(np.linspace(-1,1,N1)))*a*np.sqrt(3)/3
    elif dir=='60' or dir =='30':
        def_x=-defo*(1-np.abs(np.linspace(-1,1,N1)))*a*np.sqrt(3)/3*np.cos(np.pi/3)
        def_y=defo*(1-np.abs(np.linspace(-1,1,N1)))*a*np.sqrt(3)/3*np.sin(np.pi/3)
    '''

    # Correct direction only depends on N1, no need to check dir
    # This creates a 0° or 90° soliton and an 30° or 60° antisoliton
    # What I want to see is if they can annihilate or are topologically protected
    # I expect the latter

    # We actually need two pairs of solitons, as the number of atoms is conserved otherwise, but the system is not periodic

    ''' # Could be a solution, but it seems to be unstable
    def_x1=-defo*(1-np.abs(np.linspace(-1,1,N1//2)))*a*np.sqrt(3)/3*np.cos(np.pi/3)
    def_y1=defo*(1-np.abs(np.linspace(-1,1,N1//2)))*a*np.sqrt(3)/3*np.sin(np.pi/3)

    def_x2=-defo*(1-np.abs(np.linspace(-1,1,N1//2)))*a*np.sqrt(3)/3
    def_y2=np.zeros(N1//2)

    def_x=np.zeros(N1)
    def_y=np.zeros(N1)
    def_x[:N1//4]=def_x1[:N1//4]
    def_y[:N1//4]=def_y1[:N1//4]
    def_x[N1//4:N1//2+N1//4]=def_x2+def_x1[N1//4]-def_x2[0]
    def_y[N1//4:N1//2+N1//4]=def_y2+def_y1[N1//4]-def_y2[0]
    def_x[N1//2+N1//4:]=def_x1[N1//4:]# + def_x2[-1] - def_x2[0]
    def_y[N1//2+N1//4:]=def_y1[N1//4:]# + def_y2[-1] - def_y2[0]

    '''
    shift_x=-defo*a*np.sqrt(3)/3*np.cos(np.pi/3)-defo*a*np.sqrt(3)/3
    shift_y=defo*a*np.sqrt(3)/3*np.sin(np.pi/3)

    def_x=np.zeros(N1)
    def_y=np.zeros(N1)
    
    def_x[:N1//2]=defo*(1-np.abs(np.linspace(-1,0,N1//2)))*a*np.sqrt(3)/3

    def_x[N1//2:]=-defo*(1-np.abs(np.linspace(0,1,N1-N1//2)))*a*np.sqrt(3)/3*np.cos(np.pi/3)-shift_x
    def_y[N1//2:]=defo*(1-np.abs(np.linspace(0,1,N1-N1//2)))*a*np.sqrt(3)/3*np.sin(np.pi/3)- shift_y
    
    for i in range(Nx):
        for j in range(Ny):
            if dir=='zigzag': # zigzag
                k=j
            elif dir=='armchair':
                k=i
            # The Lx/3 factor is for pure AB stacking
            deformation[4*(i+j*Nx)]=np.array([Lx/3+def_x[k],def_y[k]])
            deformation[4*(i+j*Nx)+1]=np.array([Lx/3+def_x[k],def_y[k]])
            deformation[4*(i+j*Nx)+2]=np.array([Lx/3+def_x[k],def_y[k]])
            deformation[4*(i+j*Nx)+3]=np.array([Lx/3+def_x[k],def_y[k]])


    pos_b=pos_b-deformation/2
    pos_t=pos_b+deformation # Confusing, but sure

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