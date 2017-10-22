import numpy as np;
from BarryHamiltonianProvider import getHamiltonian;
import scipy;
#import matplotlib.pyplot as plt;

N = 30; #size of epsilons
P = 40; #population

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

#Fredkin:
Utarget = np.eye(8);
Utarget[6][6] = Utarget[4][4] = 0;
Utarget[6][4] = Utarget[4][6] = 1;

def getFidelity(E1, E2, E3):
    Ucb = getUnitary(E1,E2,E3);
    #Utarget[6][6] = Utarget[7][7] = 0;
    #Utarget[6][7] = Utarget[7][6] = 1;
    M = np.dot(np.conjugate(Utarget), Ucb);
    fidelity = 0.125*abs(np.trace(M));

    return fidelity;

def getbest(P1, P2, P3):
    best = 0;
    bErr = getFidelity(P1[0], P2[0], P3[0]);
    for i in range(P):
        if getFidelity(P1[i], P2[i], P3[i]) > bErr:
            best = i;
            bErr = getFidelity(P1[i], P2[i], P3[i]);
    return best;

def wrapperGetBest(full):
    return getbest(full[:,0:N], full[:,N:2*N], full[:,2*N:3*N] );

def getworst(P1, P2, P3):
    best = 0;
    bErr = getFidelity(P1[0], P2[0], P3[0]);
    for i in range(P):
        if getFidelity(P1[i], P2[i], P3[i]) < bErr:
            best = i;
            bErr = getFidelity(P1[i], P2[i], P3[i]);
    return best;

def wrapperGetWorst(full):
    return getworst(full[:,0:N], full[:,N:2*N], full[:,2*N:3*N] );

def wrapperGetFidelity(full):
    return getFidelity(full[0:N], full[N:2*N], full[2*N:3*N] );    


pops = 5000*np.random.rand(P,3*N) - 2500;

B = wrapperGetBest(pops);
print(wrapperGetFidelity(pops[B]));
#plt.plot(pops[B]);

mUg = 0.8;
xig = 0.9;

mUU = 0.1;
mUL = 0.1;
kappa1 = 0.1;
kappa2 = 0.9;

inputSwitch = 0.76;

gen = 0;
while (gen<50):
    gen +=1;
    count = 0;
    if np.random.rand() < inputSwitch:
        #Only use a subspace
        sub = np.random.choice(range(3*N), 1);
        for i in range(P):
            m = np.random.choice(range(P), 3, replace=False); #3 unique random population members
            mutat = pops[m[0]][sub] + mUg*(pops[m[1]][sub] - pops[m[2]][sub]);
            cross = np.zeros(3*N);
            for j in range(3*N):
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
            cross = np.zeros(3*N);
            for j in range(3*N):
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

#0.94732
"""
array([-3113.73512963, -3165.04337832,  2643.08224052,   505.11689546,
        5049.21754292, -6438.01683889,    72.65871049, -4116.65682153,
        4482.99067048,   826.8356443 , -1871.26244357, -1973.75623227,
        5639.96738033,   896.74772688,   217.48589822,  -306.59059297,
       -1010.28318904, -3914.07836274, -2670.03698638,  2952.71627229,
        2368.50968547,   353.53899826,    77.0098577 , -3907.10640688,
       -4363.14548695,  1129.91634968,  -309.02825989,  1753.36164219,
        2592.90999939, -2619.79159492, -4132.65049373,  -885.14656345,
       -2116.35229658, -1729.34315415,  1547.87914042,   509.9387357 ,
        -731.44337386,  2073.53484638, -4302.10316234, -2050.29691821,
       -3976.74101089,  1664.92425265, -1725.64140871,  -593.13934417,
         432.76020777, -4866.3744691 ,  -560.47556002,  2907.33730175,
        2208.87250924,  2269.21202941,  2807.97871911,   129.9124748 ,
       -4388.68025816,  4412.856491  ,  1485.50830298,  -801.47900771,
       -1532.8729352 , -1008.71010846,   622.39162787,  3812.30751765,
        1252.24602376, -1240.96371331, -1065.00697673, -4246.14501223,
       -5547.3283535 ,   336.34408053,  -869.68714192, -1939.04533264,
       -2270.52379896,  -950.78689783,  4929.43545263, -1455.31572161,
       -1658.31160611, -3883.63736067, -2568.003028  , -4000.77714374,
        1225.09432704,  4489.61654048,   -59.44495932,  2355.12421854,
        4244.06828847, -1179.07226555,  1595.32786556,  -385.92674082,
       -1109.12381816,  -359.91962561,  1932.18272126, -2146.8292756 ,
        4240.7010985 ,  2255.29993446])
    
P10,000;G1400
0.949047375215

array([  -29.05784526,   416.16088898,   131.50489226, -1502.57256855,
       -1268.3249109 ,  1513.76869748,  1545.11162997,  1358.34784541,
         819.79830589,  1679.23958471,  1307.46386915, -2436.73271832,
       -1569.53061347,   448.14763511,  -781.76793022, -1322.46640651,
         709.90803628, -1908.06623925, -1237.74689626,  1905.1423214 ,
       -1050.08053965,  -859.36048181, -1593.687145  , -1636.26104979,
        1747.09383206, -2212.48155685,  1998.72542662, -2229.25501745,
         426.32609314, -1643.03426831,  1705.44040897,  -596.21312345,
        2056.17390709, -1103.09941063, -1944.62456968,  -501.70200759,
       -2175.52161928, -2187.82599253,  -626.73996651,   -18.08695386,
        2307.41620454, -1639.74862997,  2307.97025738, -1050.43176528,
       -2035.40531958,   367.53490807,   922.0242461 ,    -6.14986107,
         904.09801775,    93.07076185,  -217.80309956,  -841.06993805,
       -1315.68777602,   323.81587745,   557.57361032,  1907.91116174,
        -189.46610555,  1934.61722587, -1196.85471754,   951.0448221 ,
        -527.46715185, -2428.04389872, -1838.38630719,   732.81679442,
        -576.21146836, -1384.37511657,    13.01580897,  2033.06564428,
        2376.29560501,   421.73215979, -1907.13017926,  -783.36738792,
         342.63648104,  -611.91140745, -2334.4230808 ,  -476.90920116,
         406.34738024, -2356.87722809,   854.25299875, -1200.65836377,
        -208.72491363, -1889.91852064,  -287.23812155,  1067.71729959,
         764.77391045,  2029.06071955,  1221.84899421,  -697.8834586 ,
       -1057.62101822,  1529.40248159])
    """
