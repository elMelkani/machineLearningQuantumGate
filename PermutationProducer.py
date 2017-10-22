"""
Program calculates the permutation matrix required to transform the 64-Hamiltonian into block diagonal form
"""

import numpy as np;

#To get proper block-diagonal form the ordering of the basis has to be changed in the order of excitations
#The Permutation matrix is the basis change matrix for that function.
A = np.zeros([64,1]);
N = 0;
for a in range(4):
    for b in range(4):
        for c in range(4):
            A[N] = (a+1)*100 + (b+1)*10 + (c+1);
            N = N + 1;
#A now contains entries corresponding to the tensor product of the atoms' basis(ie the original basis)
#It's Nth entry corresponds to a number ijk where i is level in atom1, j is level in atom2, etc.
#Thus its total energy is sum of i, j, and k.
perm = np.zeros([64,64]);
#Will permute A(original basis) into a new basis arranged in increasing order of total energy.
energy = 3;
completed = 0;
for i in range(9):
    energy = 3 + i;
    #Will try to get all leels having this much energy in this run of i-loop
    for j in range(64):
        if(int(A[j]/100) + int(A[j]/10)%10 + A[j]%10) == energy:
            perm[completed][j] = 1;
            completed = completed + 1;
perm[63][63] = 1;
a = np.asarray(perm);
np.savetxt("permuter.csv", a, delimiter=",");