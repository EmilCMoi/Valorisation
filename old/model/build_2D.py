import numpy as np
import matplotlib.pyplot as plt
import os
def build_2D(m,defo,filename,verify=False,verbose=False):
    """
    Build a 2D twisted bilayer structure with m as the twist parameter, 
    defo is just to indicated wether the twist is applied or not, 
    and filename is the name of the output file.
    """

    n=m-1
    a=2.511
    a_g=2.46
    c=6.451326
    theta=np.arctan2(np.sqrt(3,dtype=np.float128)*(m**2-n**2),m**2+n**2+4*m*n)
    Ng=round(np.floor(a/a_g/np.sin(theta/2)/2))
    #(theta/np.pi*180)
    M=np.array([[n, m],[-m,m+n]])
    v1=np.array([ a/2, a*np.sqrt(3/4)])
    v2=np.array([ -a/2,a*np.sqrt(3/4)])
    # Rotate vectors
    R=np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    vv1=R.dot(v1)
    vv2=R.dot(v2)

    R2=np.array([[np.cos(theta/2), -np.sin(theta/2)],
                 [np.sin(theta/2), np.cos(theta/2)]])
    vvv1=R2.dot(v1)
    vvv2=R2.dot(v2)
    R_inv=R.T
    vinv1=R_inv.dot(vvv1)
    vinv2=R_inv.dot(vvv2)
    R_inv2=R2.T
    vinvv1=R_inv2.dot(vvv1)
    vinvv2=R_inv2.dot(vvv2)
    #vg1=np.array([ a_g/2, a_g*np.sqrt(3/4)])
    #vg2=np.array([ -a_g/2,a_g*np.sqrt(3/4)])
    A1=M[0,0]*v2+M[0,1]*v1
    A2=M[1,0]*v2+M[1,1]*v1
    vg1=A1/np.linalg.norm(A1)*a_g
    vg2=A2/np.linalg.norm(A2)*a_g
    ratio=round(A1.dot(A2)/(v1.dot(v2)))
    #print(ratio)
    #print(np.linalg.norm(A1)*3)

    # Generate input file
    
    inp=open('/home/zanko/PDM/FINAL/model/twist.template','r')
    inp=inp.readlines()
    inp[21]=f"{theta}\n"
    inp[27]=f"{round(m)} {round(n)}\n"
    inp[31]=f"{round(-n)} {round(m+n)}\n"
    inpd=open('twist.inp','w')
    inpd.writelines(inp)
    inpd.close()

    # Run twister
    if verbose:
        os.system("python3 /home/zanko/PDM/Twister/SRC/twister.py")
    else:
        os.system("python3 /home/zanko/PDM/Twister/SRC/twister.py > twister.log")

    #Load data

    os.system("sed -i 's/B/1/' superlattice.dat")
    os.system("sed -i 's/N/2/' superlattice.dat")

    datLat=np.loadtxt('superlattice.dat',skiprows=1,max_rows=2)

    dat1=np.loadtxt('superlattice.dat',skiprows=4,max_rows=ratio*2)
    dat2=np.loadtxt('superlattice.dat',skiprows=6+2*ratio,max_rows=2*ratio)

    datLat*=a
    A1=datLat[1,:2]
    A2=datLat[0,:2]


    B1=np.array([dat1[i] for i in range(len(dat1)) if dat1[i,0]==1])
    N1=np.array([dat1[i] for i in range(len(dat1)) if dat1[i,0]==2])
    #print(np.shape(B1))
    if defo:
        B2=np.array([dat2[i] for i in range(len(dat2)) if dat2[i,0]==1])
        N2=np.array([dat2[i] for i in range(len(dat2)) if dat2[i,0]==2])
    else:
        B2=np.copy(B1)
        N2=np.copy(N1)
        for i in range(len(B1)):
            B2[i,1:3]-=vvv2/np.sqrt(3)
            N2[i,1:3]-=vvv2/np.sqrt(3)
            B2[i,3]=c/2
            N2[i,3]=c/2

    
    # Define transformation for LAMMPS

    A=A2
    A2=A1
    A1=A
    ax=np.linalg.norm(A1)
    bx=A1.dot(A2)/ax
    B=A2-bx/ax*A1
    by=np.linalg.norm(B)
    eta=0.995#0.99 # elastic deformation term, may yield better results
    M=np.linalg.inv([A/np.linalg.norm(A), B/np.linalg.norm(B)])*eta
    #print(ratio)
    datdat=[""]
    datdat.append(f"\n{round(ratio*4)} atoms \n2 atom types\n")
    datdat.append(f"0 {ax*eta} xlo xhi\n0 {by*eta} ylo yhi\n-100 100 zlo zhi\n\n")
    datdat.append(f"{bx} 0 0 xy xz yz\n\n")
    datdat.append("Masses\n\n1 10.811\n2 14.007\n\n")
    datdat.append("Atoms # full\n\n")


    #print(len(B2))
    #print(len(N2))
    #print(len(N1))
    #print(len(B1))
    for i in range(len(N1)):
        B1[i,1:3]=M.dot(B1[i,1:3])
        B2[i,1:3]=M.dot(B2[i,1:3])
        N1[i,1:3]=M.dot(N1[i,1:3])
        N2[i,1:3]=M.dot(N2[i,1:3])
        
        datdat.append(f"{4*i+1} 1 1 0.42 {B1[i,1]} {B1[i,2]} 0.0\n")
        datdat.append(f"{4*i+2} 2 1 0.42 {B2[i,1]} {B2[i,2]} {c/2}\n")
        datdat.append(f"{4*i+3} 1 2 -0.42 {N1[i,1]} {N1[i,2]} 0.0\n")
        datdat.append(f"{4*i+4} 2 2 -0.42 {N2[i,1]} {N2[i,2]} {c/2}\n")

    if verify:
        A1=M.dot(A1)
        A2=M.dot(A2)
        v1=M.dot(vvv1)
        v2=M.dot(vvv2)
        plt.figure()
        plt.arrow(0,0,A1[0],A1[1],width=0.05)
        plt.arrow(0,0,A2[0],A2[1],width=0.05)

        plt.arrow(0,0,v1[0],v1[1],width=0.05)
        plt.arrow(0,0,v2[0],v2[1],width=0.05)
        plt.plot(B1[:,1],B1[:,2],'r.',markersize=10)
        plt.plot(N1[:,1],N1[:,2],'b.',markersize=10)
        plt.plot(B2[:,1],B2[:,2],'m.',markersize=5)
        plt.plot(N2[:,1],N2[:,2],'c.',markersize=5)
        plt.axis('equal')
        plt.show()


    dat = open(filename,'w')
    dat.writelines(datdat)
    dat.close()

def build_2D_cavity(m,r,defo,filename,verify=False,verbose=False):
    """
    Build a 2D twisted bilayer structure with m as the twist parameter,
    r is the radius of the cavity, 
    defo is just to indicated wether the twist is applied or not, 
    and filename is the name of the output file.
    """

    n=m-1
    a=2.511
    a_g=2.46
    c=6.451326
    theta=np.arctan2(np.sqrt(3,dtype=np.float128)*(m**2-n**2),m**2+n**2+4*m*n)
    Ng=round(np.floor(a/a_g/np.sin(theta/2)/2))
    #(theta/np.pi*180)
    M=np.array([[n, m],[-m,m+n]])
    v1=np.array([ a/2, a*np.sqrt(3/4)])
    v2=np.array([ -a/2,a*np.sqrt(3/4)])
    # Rotate vectors
    R=np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    vv1=R.dot(v1)
    vv2=R.dot(v2)

    R2=np.array([[np.cos(theta/2), -np.sin(theta/2)],
                 [np.sin(theta/2), np.cos(theta/2)]])
    vvv1=R2.dot(v1)
    vvv2=R2.dot(v2)
    R_inv=R.T
    vinv1=R_inv.dot(vvv1)
    vinv2=R_inv.dot(vvv2)
    R_inv2=R2.T
    vinvv1=R_inv2.dot(vvv1)
    vinvv2=R_inv2.dot(vvv2)
    #vg1=np.array([ a_g/2, a_g*np.sqrt(3/4)])
    #vg2=np.array([ -a_g/2,a_g*np.sqrt(3/4)])
    A1=M[0,0]*v2+M[0,1]*v1
    A2=M[1,0]*v2+M[1,1]*v1
    vg1=A1/np.linalg.norm(A1)*a_g
    vg2=A2/np.linalg.norm(A2)*a_g
    ratio=round(A1.dot(A2)/(v1.dot(v2)))
    #print(ratio)
    #print(np.linalg.norm(A1)*3)

    # Generate input file
    
    inp=open('/home/zanko/PDM/FINAL/model/twist.template','r')
    inp=inp.readlines()
    inp[21]=f"{theta}\n"
    inp[27]=f"{round(m)} {round(n)}\n"
    inp[31]=f"{round(-n)} {round(m+n)}\n"
    inpd=open('twist.inp','w')
    inpd.writelines(inp)
    inpd.close()

    # Run twister
    if verbose:
        os.system("python3 /home/zanko/PDM/Twister/SRC/twister.py")
    else:
        os.system("python3 /home/zanko/PDM/Twister/SRC/twister.py > twister.log")

    #Load data

    os.system("sed -i 's/B/1/' superlattice.dat")
    os.system("sed -i 's/N/2/' superlattice.dat")

    datLat=np.loadtxt('superlattice.dat',skiprows=1,max_rows=2)

    dat1=np.loadtxt('superlattice.dat',skiprows=4,max_rows=ratio*2)
    dat2=np.loadtxt('superlattice.dat',skiprows=6+2*ratio,max_rows=2*ratio)

    datLat*=a
    A1=datLat[1,:2]
    A2=datLat[0,:2]


    B1=np.array([dat1[i] for i in range(len(dat1)) if dat1[i,0]==1])
    N1=np.array([dat1[i] for i in range(len(dat1)) if dat1[i,0]==2])
    #print(np.shape(B1))
    if defo:
        B2=np.array([dat2[i] for i in range(len(dat2)) if dat2[i,0]==1])
        N2=np.array([dat2[i] for i in range(len(dat2)) if dat2[i,0]==2])
    else:
        B2=np.copy(B1)
        N2=np.copy(N1)
        for i in range(len(B1)):
            B2[i,1:3]-=vvv2/np.sqrt(3)
            N2[i,1:3]-=vvv2/np.sqrt(3)
            B2[i,3]=c/2
            N2[i,3]=c/2


    # Graphene 

    C1=np.zeros((Ng*Ng,2))
    C2=np.zeros((Ng*Ng,2))
    for i in range(Ng):
        for j in range(Ng):
            C1[i*Ng+j]=i*vg1+j*vg2
            C2[i*Ng+j]=i*vg1+j*vg2+(vg1+vg2)/3
    # Remove the atoms in the holes
    # Centers of holes have to be defined
    h1=A1/2
    h2=A2/2
    h3=A1/2+A2/2
    # Periodic repetitions of the holes, easier to implement this way
    h4=A1+A2/2
    h5=A1/2+A2

    C1=C1[np.linalg.norm(C1-h1,axis=1)>r]
    C2=C2[np.linalg.norm(C2-h1,axis=1)>r]
    C1=C1[np.linalg.norm(C1-h2,axis=1)>r]
    C2=C2[np.linalg.norm(C2-h2,axis=1)>r]
    C1=C1[np.linalg.norm(C1-h3,axis=1)>r]
    C2=C2[np.linalg.norm(C2-h3,axis=1)>r]
    C1=C1[np.linalg.norm(C1-h4,axis=1)>r]
    C2=C2[np.linalg.norm(C2-h4,axis=1)>r]
    C1=C1[np.linalg.norm(C1-h5,axis=1)>r]
    C2=C2[np.linalg.norm(C2-h5,axis=1)>r]

    # Define transformation for LAMMPS

    A=A2
    A2=A1
    A1=A
    ax=np.linalg.norm(A1)
    bx=A1.dot(A2)/ax
    B=A2-bx/ax*A1
    by=np.linalg.norm(B)
    eta=0.995#0.99 # elastic deformation term, may yield better results
    M=np.linalg.inv([A/np.linalg.norm(A), B/np.linalg.norm(B)])*eta
    #print(ratio)
    datdat=[""]
    datdat.append(f"\n{round(ratio*4)} atoms \n2 atom types\n")
    datdat.append(f"0 {ax*eta} xlo xhi\n0 {by*eta} ylo yhi\n-100 100 zlo zhi\n\n")
    datdat.append(f"{bx} 0 0 xy xz yz\n\n")
    datdat.append("Masses\n\n1 10.811\n2 14.007\n\n")
    datdat.append("Atoms # full\n\n")


    #print(len(B2))
    #print(len(N2))
    #print(len(N1))
    #print(len(B1))
    for i in range(len(N1)):
        B1[i,1:3]=M.dot(B1[i,1:3])
        B2[i,1:3]=M.dot(B2[i,1:3])
        N1[i,1:3]=M.dot(N1[i,1:3])
        N2[i,1:3]=M.dot(N2[i,1:3])
        
        datdat.append(f"{4*i+1} 1 1 0.42 {B1[i,1]} {B1[i,2]} 0.0\n")
        datdat.append(f"{4*i+2} 2 1 0.42 {B2[i,1]} {B2[i,2]} {c}\n")
        datdat.append(f"{4*i+3} 1 2 -0.42 {N1[i,1]} {N1[i,2]} 0.0\n")
        datdat.append(f"{4*i+4} 2 2 -0.42 {N2[i,1]} {N2[i,2]} {c}\n")

    # Add graphene atoms
    for i in range(len(C1)):
        C1[i,:]=M.dot(C1[i,:])
        datdat.append(f"{4*len(N1)+i+1} 3 3 0.0 {C1[i,0]} {C1[i,1]} {c/2}\n")
    for i in range(len(C2)):
        C2[i,:]=M.dot(C2[i,:])
        datdat.append(f"{4*len(N1)+len(C1)+i+1} 3 3 0.0 {C2[i,0]} {C2[i,1]} {c/2}\n")

    if verify:
        A1=M.dot(A1)
        A2=M.dot(A2)
        v1=M.dot(vvv1)
        v2=M.dot(vvv2)
        plt.figure()
        plt.arrow(0,0,A1[0],A1[1],width=0.05)
        plt.arrow(0,0,A2[0],A2[1],width=0.05)

        plt.arrow(0,0,v1[0],v1[1],width=0.05)
        plt.arrow(0,0,v2[0],v2[1],width=0.05)
        plt.plot(B1[:,1],B1[:,2],'r.',markersize=15)
        plt.plot(N1[:,1],N1[:,2],'b.',markersize=15)
        plt.plot(B2[:,1],B2[:,2],'m.',markersize=10)
        plt.plot(N2[:,1],N2[:,2],'c.',markersize=10)
        plt.plot(C1[:,0],C1[:,1],'g.',markersize=5)
        plt.plot(C2[:,0],C2[:,1],'g.',markersize=5)

        plt.axis('equal')
        plt.show()


    dat = open(filename,'w')
    dat.writelines(datdat)
    dat.close()