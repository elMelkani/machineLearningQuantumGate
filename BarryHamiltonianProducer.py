"""
Program calculates the constant part of the 20 dimensional Hamiltonian and writes it into a csv file. 
"""

import numpy as np;
#from PermutationProducer import getPermutationMat;

g = 30; #in MHz

X = np.zeros([4,4]);
X[0][1] = X[1][0] = 1;
X[2][1] = X[1][2] = np.sqrt(2);
X[2][3] = X[3][2] = np.sqrt(3);
XProduct = np.kron(X,X); #Xk*Xk+1

Y = np.zeros([4,4]);
Y[0][1] = -1; Y[1][0] = 1;
Y[1][2] = -np.sqrt(2); Y[2][1] = np.sqrt(2); 
Y[2][3] = -np.sqrt(3); Y[3][2] = np.sqrt(3);
YProduct = -np.kron(Y,Y); #Negative sign to take care of iota.

Id4 = np.zeros([4,4]);
for i in range(4):
    Id4[i][i] = 1;

M1 = np.kron(XProduct, Id4) + np.kron(Id4, XProduct);
M2 = np.kron(YProduct, Id4) + np.kron(Id4, YProduct);

total = (g/2)*(M1 + M2);

P = np.genfromtxt ('permuter.csv', delimiter=",");
Ptranspose = P.transpose();
Ham = np.dot(P, total);
Ham = np.dot(Ham, Ptranspose);
Ham = Ham[0:20,0:20];

a = np.asarray(Ham);
np.savetxt("constantHamiltonian.csv", a, delimiter=",");