#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#  HARTREE-FOCK SELF CONSISTENT FIELD CODE FOR H2O. (BOND-LENGTH = 1.1 Angstroms, BOND-ANGLE = 104 , BASIS SET = STO-3G)
# 
#  The files containing integrals s.dat, v.dat, t.dat, eri.dat, enuc.dat have to be stored in the execution folder.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import numpy as np
import scipy.linalg as la
import math

H = np.zeros((7,7))     # The H-core
T = np.zeros((7,7))     # The KE matrix
V = np.zeros((7,7))     # The PE matrix
S = np.zeros((7,7))     # Overlap matrix
F = np.zeros((7,7))     # Fock matrix
S_inv = np.zeros((7,7)) 
Co = np.zeros((7,7))    # intial Coefficient matrix
Do = np.zeros((7,7))    # initial Density matrix
eigvecs = np.zeros((7,7))
eigvals = np.zeros(7)
eigval_mat = np.zeros((7,7))
val = np.zeros(228)     # array to read 2 electron integral values from eri.dat
tei = np.zeros(406)     # 1-d array to store 2401 two-electron integrals.

#******************** FUNCTION DEFINITIONS ***********************************************


def indices(i,j):            # function to create compound indices for storing 2e integrals in a 1-d array

  if i>j:
      ij = i*(i+1)/2 + j
  else:
      ij = j*(j+1)/2 + i

  n = int(ij)
  return n


#*****************************************************************************************

def mat(data):                 # Function to store S,V,T matrix
  M = np.zeros((7,7))
  x = list()
  for c in range(28):
      x = data.readline()
      y = x.split()
      i = int(y[0])-1
      j = int(y[1])-1
      M[i][j] = M[j][i] = y[2]

  return M

#******************************************************************************************

def calc_F(D,H,tei):         # function to calculate Fock matrix
  for i in range(7):
    for j in range(7):
        F[i][j] = H[i][j]
        for k in range(7):
            for l in range(7):
                  ij = indices(i,j)
                  kl = indices(k,l)
                  ik = indices(i,k)
                  jl = indices(j,l)
                  ikjl = indices(ik,jl)
                  ijkl= indices(ij,kl)
                  F[i][j] = F[i][j] + ( D[k][l] * (( 2 * tei[ijkl] ) - tei[ikjl] ))

  return F

#*******************************************************************************************

def calc_D(C,H,F):           # Function to calcuate Density matrix
  E = 0.0
  D = np.zeros((7,7))
  for i in range(7):
    for j in range(7):
        s=0.0
        for k in range(5):   # occupied orbitals are first five columns of C
             s = s + (C[i][k]*C[j][k])
        D[i][j] = s
        E = E + D[i][j]*(H[i][j]+F[i][j])   # calculating energy
  return [E,D];

#*******************************************************************************************

def rmsd(D_p,D_n):           # Function to calculate root mean square deviation of density matrix.
    r=0.0
    d=0.0
    for i in range(7):
        for j in range(7):
            d = d + ((D_n[i][j] - D_p[i][j])**2)
    r = math.sqrt(d)
    return r

#*******************************************************************************************

def interchange(C):
   for i in range(7):        # The Co matrix produced had (3rd/6th) and (5th/7th) column exchanged from the reference Co in github
      c = C[i][2]            # So i have exchanged them to match with the reference, else initial density matrix will be different.
      C[i][2] = C[i][5]      # It is effectively setting first five columns as occupied eigenvectors.
      C[i][5] = c
      d = C[i][4]
      C[i][4] = C[i][6]
      C[i][6] = d
   return C 

#**************** FUNCTION DEFINITIONS OVER ************************************************




overlap = open("s.dat","r")       # opening all files
kinetic = open("t.dat","r")
potential = open("v.dat","r")
eri = open("eri.dat","r")
enuc = open("enuc.dat","r")


#****************  STEP 1 - NUCLEAR REPULSION ENERGY ***************************************************

e_nuc = float(enuc.readline())
print("The Nuclear repulsion energy is ",e_nuc)


#****************  STEP 2- ONE ELCTRON INTEGRALS ***************************************************

S = mat(overlap)                  # passing the files to function mat.
print('---- OVERLAP MATRIX ----')
print(S)

T = mat(kinetic)
V = mat(potential)
print('---- KINETIC ENERGY MATRIX ----')
print(T)
print('---- NUCLEAR ATTRACTION MATRIX ----')
print(V)
print('---- H CORE ----')
H = T + V      
print(H)

#****************  STEP 3 -TWO ELCTRON INTEGRALS ***************************************************

for ii in range(228):        # Storing the 2 electron integrals as compound indices. We dont have to construct 4d matrix with 2401 values.
  line = eri.readline()      # we are reading line by line
  toks = line.split()        # We have split it into i,j,k,l and value of the TEI.
  i = int(toks[0])-1
  j = int(toks[1])-1
  k = int(toks[2])-1
  l = int(toks[3])-1
  ij = int(indices(i,j))
  kl = int(indices(k,l))
  ijkl = int(indices(ij,kl))    # forming the compound indices.
  #print(ij,kl,ijkl)
  tei[ijkl] = float(toks[4])    # storing the integral values in 1-d array having compound indices.


#****************  STEP 4 - BUILD THE ORTHOGONALIZATION MATRIX ***************************************************


#print('---- EIGENVALUES AND EIGENVECTORS OF S ----')
eigvals, eigvecs = la.eig(S)      # la.eig function diagonalizes the S matrix and returns eigenvalues and eigenvectors    
eigvals = eigvals.real
#print(eigvals)
#print(eigvecs)

for i in range(7):                # to form diagonal matrix from the eigenvalue array
    for j in range(7):
      if i==j :
         eigval_mat[i][j] = ((eigvals[i])**(-0.5))      # multiply -0.5 to each element to calculate the power -1/2 of matrix
      else:
         eigval_mat[i][j] = 0.0


#print(eigvecs)
#print(eigval_mat)

print("---- S_INVERSE MATRIX ----")
S_inv = np.matmul(eigvecs,np.matmul(eigval_mat,eigvecs.transpose()))    # multiply (eigvecs * eigval_mat * eigvecs_t) to get S power -1/2
print(S_inv)


#****************  STEP 5 - BUILD THE INITIAL GUESS DENSITY MATRIX ***************************************************


print("---- INITIAL FOCK MATRIX ----")
F = np.matmul(S_inv.transpose(),np.matmul(H,S_inv))         # form the initial guess Fock matrix
print(F)

eigvals, eigvecs = la.eig(F)               # Diagonalize intial Fock matrix
print("---- EIGENVALUES AND EIGENVECTORS OF F ----")
eigvals = eigvals.real
print(eigvals)                             # we can check the eigenvalues to see that 3rd and 5th value of array have highest energy
print(eigvecs)                             # Hence the eigenvectors in 3rd and 5th column are not occupied

Co = np.matmul(S_inv,eigvecs)
Co = interchange(Co)                       # interchange function was called to set the first 5 columns as occupied eigenvectors.

print("---- INTIAL COEFFICIENT MATRIX ----")       
print(Co)


E_elec=0.0                               
for i in range(7):                # calculating initial density matrix
    for j in range(7):
        s=0.0
        for k in range(5):         
             s = s + (Co[i][k]*Co[j][k])
        Do[i][j] = s
        E_elec = E_elec + Do[i][j]*(H[i][j]+H[i][j])    # Calculating the initial energy too

        
#****************  STEP 6 - COMPUTE THE INITIAL SCF ENERGY ***************************************************
        
E_total = (E_elec + e_nuc)         
print("---- INITIAL DENSITY MATRIX ----")
print(Do)
print("The intial E_elec is =",E_elec)
print("E_total = E_elec + Enuc = ",E_total)

#****************  STEP 7 - COMPUTE THE NEW FOCK MATRIX ***************************************************

F = calc_F(Do,H,tei)             # calling the calc_F function to calculate Fock matrix                      
print("---- First iteration for Fock Matrix ----")
print(F)         

#****************  STEP 8 - THE SCF LOOP AND TEST FOR CONVERGENCE ***************************************************

D_p = Do           # declaring variables for the loop 
F_p = F
E_p = E_elec       
delta = 0.1       
r = 0.1
n = 0
print("---- ITERATIONS FOR THE ENERGY ----")
print("iter    E_elec        E_total              delta           rmsd")

while abs(delta) > 10**(-12) and r > 10**(-12):    # The SCF loop 
    n=n+1                                                  # counter
    F_n = calc_F(D_p,H,tei)
    F_o = np.matmul(S_inv.transpose(),np.matmul(F_n,S_inv))
    eigvals, eigvecs = la.eig(F_o)
    eigvals = eigvals.real                 # we can print eigvals to check energies and hence select column vectors for density.
    C = np.matmul(S_inv,eigvecs)
    C = interchange(C)             # updates the Coefficient matrix so that first five column vectors are the occupied ones.
    E_n, D_n = calc_D(C,H,F_o)
    r = rmsd(D_p,D_n)
    delta = E_n - E_p                      # the energy difference calculation.
    print(n,E_n,E_n + e_nuc,delta,r)
    E_p = E_n                              # updates the energy and density matrix, where p means previous, n means next.
    D_p = D_n



#**************** END *************************************************************************************
