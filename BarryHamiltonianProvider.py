"""
Program takes in the epsilon vectors and returns the total 20D Hamiltonian integrated over time N*dt
"""

import numpy as np;
#from PermutationProducer import getPermutationMat;

eeta = 200; #in MHz
eetaDash = 3*eeta;

#constant part has already been calculated by BarryHamiltonianProducer
constHam = np.genfromtxt ('constantHamiltonian.csv', delimiter=",");

P = np.genfromtxt ('permuter.csv', delimiter=",");
Id4 = np.eye(4);

def getHamiltonian(E1, E2, E3):
    dt = 1; #1 nanosecond - the time interval for steps in the epsilons
    N = E1.shape[0];
    intE1 = intE2 = intE3 = 0;
    
    for i in range(N):
        intE1 += E1[i]*dt;
        intE2 += E2[i]*dt;
        intE3 += E3[i]*dt;
        
    Neeta = dt*N*eeta;
    NDeeta = dt*N*eetaDash;
    H = constHam*N*dt;
    
    H1 = np.zeros([4,4]);
    H1[1][1] = intE1;
    H1[2][2] = 2*intE1 - Neeta;
    H1[3][3] = 3*intE1 - NDeeta; 
    
    H2 = np.zeros([4,4]);
    H2[1][1] = intE2;
    H2[2][2] = 2*intE2 - Neeta;
    H2[3][3] = 3*intE2 - NDeeta; 
    
    H3 = np.zeros([4,4]);
    H3[1][1] = intE3;
    H3[2][2] = 2*intE3 - Neeta;
    H3[3][3] = 3*intE3 - NDeeta; 
        
    part1 = np.kron(np.kron(H1, Id4), Id4);
    part1 = part1 + np.kron(np.kron(Id4, H2), Id4);
    part1 = part1 + np.kron(np.kron(Id4, Id4), H3);
    
    
    part1 = np.dot(P, part1);
    part1 = np.dot(part1, P.transpose());
    part1 = part1[0:20,0:20];
    
    H = H + part1;
    
    return H;