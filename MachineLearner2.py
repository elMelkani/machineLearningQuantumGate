import numpy as np;
from BarryHamiltonianProvider import getHamiltonian;
import scipy;
#import matplotlib.pyplot as plt;

N = 30; #size of epsilons
P = 100; #population

"""
eps1 = np.random.rand(N);
eps2 = np.random.rand(N);
eps3 = np.random.rand(N);

a = np.asarray(H);
np.savetxt("bo.csv", a, delimiter=",");
"""

def getUnitary(E1, E2, E3):
    IH = getHamiltonian(E1, E2, E3);
    U = scipy.linalg.expm(-(1j)*IH);
    #M = np.dot(np.conjugate(U), U);
    UT = np.zeros([8,8], dtype=np.complex);
    
    for i in range(8):
        for j in range(8):
            if i>=0 and i<=3:
                m = i;
            elif i==4:
                m = 5;
            elif i==5 or i==6:
                m = i+2;
            elif i==7:
                m = 15;
                
            if j>=0 and j<=3:
                n = j;
            elif j==4:
                n = 5;
            elif j==5 or j==6:
                n = j+2;
            elif j==7:
                n = 15;
            UT[i][j] = U[m][n];
            
    return UT;

def getRotator(b1, b2, b3):
    rot1 = np.zeros([8,8], dtype=np.complex);
    rot1[0][0] = 1;
    rot1[1][1] = np.exp(-(1j)*b3);
    rot1[2][2] = np.exp(-(1j)*b2);
    rot1[3][3] = np.exp(-(1j)*b1);      #EXCHANGED TO 
    rot1[4][4] = np.exp(-(1j)*(b2+b3)); #EXCHANGED TO
    rot1[5][5] = np.exp(-(1j)*(b1+b3));
    rot1[6][6] = np.exp(-(1j)*(b2+b1));
    rot1[7][7] = np.exp(-(1j)*(b2+b1+b3));
    return rot1;

Utarget = np.eye(8);
#Fredkin:
Utarget[6][6] = Utarget[4][4] = 0;
Utarget[6][4] = Utarget[4][6] = 1;
#perm = np.eye(8);
#perm[3][3] = perm[4][4] = 0;
#perm[3][4] = perm[4][3] = 1;

def getFidelity(E1, E2, E3, betas):
    Ucb = getUnitary(E1,E2,E3);
    
    rotPre = getRotator(betas[0]/1000, betas[1]/1000, betas[2]/1000);
    rotPos = getRotator(betas[3]/1000, betas[4]/1000, betas[5]/1000);
    
    Urotated = np.dot(np.dot(rotPos, Ucb), rotPre);
    
    M = np.dot(np.conjugate(Utarget), Urotated);
    fidelity = 0.125*abs(np.trace(M));

    return fidelity;

def getbest(P1, P2, P3, betas):
    best = 0;
    bErr = getFidelity(P1[0], P2[0], P3[0], betas[0]);
    for i in range(P):
        if getFidelity(P1[i], P2[i], P3[i], betas[i]) > bErr:
            best = i;
            bErr = getFidelity(P1[i], P2[i], P3[i], betas[i]);
            
    return best;

def wrapperGetBest(full):
    return getbest(full[:,0:N], full[:,N:2*N], full[:,2*N:3*N], full[:,3*N:3*N + 6] );

def getworst(P1, P2, P3, betas):
    best = 0;
    bErr = getFidelity(P1[0], P2[0], P3[0], betas[0]);
    for i in range(P):
        if getFidelity(P1[i], P2[i], P3[i], betas[i]) < bErr:
            best = i;
            bErr = getFidelity(P1[i], P2[i], P3[i], betas[i]);
    return best;

def wrapperGetWorst(full):
    return getworst(full[:,0:N], full[:,N:2*N], full[:,2*N:3*N], full[:,3*N:3*N + 6] );

def wrapperGetFidelity(full):
    return getFidelity(full[0:N], full[N:2*N], full[2*N:3*N], full[3*N:3*N + 6] );    


pops = 5000*np.random.rand(P,3*N + 6) - 2500;

B = wrapperGetBest(pops);
print(wrapperGetFidelity(pops[B]));
#plt.plot(pops[B]);

mUg = 0.8;
xig = 0.9;

mUU = 0.1;
mUL = 0.1;
kappa1 = 0.1;
kappa2 = 0.9;

inputSwitch = 0.75;

gen = 0;
while (gen<50):
    gen +=1;
    count = 0;
    if np.random.rand() < inputSwitch:
        #Only use a subspace
        sub = np.random.choice(range(3*N + 6), 1);
        for i in range(P):
            m = np.random.choice(range(P), 3, replace=False); #3 unique random population members
            mutat = pops[m[0]][sub] + mUg*(pops[m[1]][sub] - pops[m[2]][sub]);
            cross = np.zeros(3*N + 6);
            for j in range(3*N + 6):
                cross[j] = pops[i][j];
                #Initialize cross to be equal to the vector
            cross[sub] = mutat;
            if wrapperGetFidelity(cross) > wrapperGetFidelity(pops[i]):
                pops[i] = cross;
                count = count+1;
        print("gen:"+str(gen)+" mode:sub changed:"+str(count));
    else:
        #Use the full vector
        for i in range(P):
            m = np.random.choice(range(P), 3, replace=False); #3 unique random population members
            mutat = pops[m[0]] + mUg*(pops[m[1]] - pops[m[2]]);
            cross = np.zeros(3*N + 6);
            for j in range(3*N + 6):
                if np.random.rand() < xig:
                    cross[j] = mutat[j];
                else:
                    cross[j] = pops[i][j];
            if wrapperGetFidelity(cross) > wrapperGetFidelity(pops[i]):
                pops[i] = cross;
                count = count+1;
                #Cross is a better vector
        print("gen:"+str(gen)+" mode:tot changed:"+str(count));
        #Updating parameters:
    if np.random.rand() < kappa1:
        mUg = mUL + np.random.rand()*mUU;
    if np.random.rand() < kappa2:
        xig = np.random.rand();
    if gen%4 == 0:
        print(wrapperGetFidelity(pops[wrapperGetBest(pops)]));
        print(wrapperGetFidelity(pops[wrapperGetWorst(pops)]));

B = wrapperGetBest(pops);
print(wrapperGetFidelity(pops[B]));
#plt.plot(pops[B]);