import numpy as np
import pandas as pd
from random import randrange
import math
import matplotlib.pyplot as plt
import itertools
from itertools import permutations
from itertools import combinations
import copy
from pandas import *

# scipy optimize
# Pulp Intrga

V = [([1],100), ([2],125), ([3],50), ([1,2],270), ([1,3],375), ([2,3],350), ([1,2,3],500)]
z = 1
P = 1628706
# P = [1628706, 1628706, 1628706, 1628706, 1628706]
file = pd.read_csv("covid_confirmed_NY_july.csv")
mid_B = 3.0

alpha = 0.05
beta = 0.55
sigma = 0.25
gamma = 0.1

def main(z, Time = None):
    """
        z: number of zones
        Time: number of simulation
    """
    # creating N zones with initial random SEIRD values
    Z_Total = create_zone(z)
    print("number of zones:", len(Z_Total))
    print("Initial Zones:")
    print(pd.DataFrame(Z_Total))
    print("-------------------------------Coalition Game-------------------------------")
    coalition_game(Z_Total)
    print("-------------------------------")
    
    for _ in range(int(input("enter number of iterations:"))):
        newZ = [OPT(z) for z in Z_Total]
        print(pd.DataFrame(newZ))
        coalition_game(newZ)
        print("-------------------------------")
        Z_Total = newZ
        
            
    

def apply_OPT(Z, Time):
    newZ = []
    newZ.append(Z)
    print(newZ)
    T = [0]
    for t in range(1, Time):
        T.append(t)
        newZ.append(OPT(newZ[-1], t))
    print(DataFrame(newZ))
    PlottingZone(newZ, T)
    return newZ
    


def create_zone(t):
    """return one zone SEIRD values"""
    # t: number of zones
    z_total = []
    for z in range(t):
        temp = []
        people = 100
        consumed = 0
        for i in range(5):
            if i< 4:
                consumed = randrange(people)
                people = people - consumed
                temp.append(consumed)
            else:
                temp.append(people)
        z_total.append(temp)
    return z_total        




def OPT(Z):
    global beta, sigma, gamma, alpha, P
    
    if sum(Z)>100 or sum(Z)<99:
        print("Z = {}".format(sum(Z)))
    # newP = P[randrange(len(P))]
    newP = sum(Z)
    S, E, I, R, D = Z[0], Z[1], Z[2], Z[3], Z[4]
    new_infected  = sigma * E
    # S: Susceptible, E: Exposed, I: Infectious, R: Recovered, D: Dead
    newS = S - ((beta * S * I) /newP)
    newE = E + (((beta * S * I) / newP) - new_infected)
    newI = I + (new_infected - gamma * I)
    newR = R + (gamma * (1.0 - alpha) * I)
    newD = D + (gamma * alpha * I)

    return [newS, newE, newI, newR, newD]


def PlottingSEIRD(C, T, m):
    """C: given Curve matrix, T: time, m: type of model"""
    # print(DataFrame(C))
    M = ['Susceptible', 'Exposed', 'Infectious', 'Recovered', 'Dead']
    # print("-------------------------------")

    _, col = C.shape
    for z in range(col-1):
        t = C[:, z]
        plt.plot(T, t)
    plt.xlabel("time")
    plt.ylabel('Population')
    
    plt.title(M[m])
    plt.legend(loc='best')
    plt.show()


def PlottingZone(C, T):
    """C: given Curve matrix, T: time, m: type of model"""
    print("-------------------------------")
    a = np.array(C)
    M = ['Susceptible', 'Exposed', 'Infectious', 'Recovered', 'Dead']

    for m in range(5):
        plt.plot(T, a[:, m], label=M[m])
    plt.xlabel("time")
    plt.ylabel('Population')
    
    plt.title(M[m])
    plt.legend()
    plt.show()



# -------------------------------------------------------------------------------------------------
# ---------------------------------------SEIRD---------------------------------------
# -------------------------------------------------------------------------------------------------


def SEIRD(Zones):
    """Divide zones simulation to separated categories"""
    newZones = np.array(Zones)
    S = np.array([z[:,0] for z in newZones])
    E = np.array([z[:,1] for z in newZones])
    I = np.array([z[:,2] for z in newZones])
    R = np.array([z[:,3] for z in newZones])
    D = np.array([z[:,4] for z in newZones])
    return S, E, I, R, D


        
    



# -------------------------------------------------------------------------------------------------
# ---------------------------------------Coalitions Creation---------------------------------------
# -------------------------------------------------------------------------------------------------


def cpv(k):
    """Cost Per Vaccine 
    As number of vaccines order increases (k), cost (C) reduced
    """
    # cpv : Cost Per Vaccine
    # k : Number of vaccine in the order of 100.000
    A = 20
    B = 10
    result = A * ( B - k + 1)
    if result>0:
        return result
    else:
        return 20

def reward(k, k_th):
    """
    K: number of vaccines == number of infected ppl. in zone i.
    k_th: min number of vaccine needed for zone"""
    if k < k_th:
        return 0
    else:
        #print("k = {} , cpv({})= {}, k_th = {}".format(k,k, cpv(k), k_th))
        return math.log(k, cpv(k))
    
    
def LogFunction():
    k_th = 5
    # k = number of vaccine that a zone orders
    # cpv = cost per vaccine
    plt.plot([k for k in range(1,10)], 
             [reward(k, k_th) for k in range(1,10)],
             label = 'Cost per Vaccine: $' )
    plt.xlabel("number of vaccines purchased (in order of 100K)")
    plt.ylabel('Reward')
    
    plt.title('Minimum vaccines needed' + str(k_th) + '00,000')
    plt.legend()
    plt.show()


# -------------------------------------------------------------------------------------------------
# ---------------------------------------Coalitions Creation---------------------------------------
# -------------------------------------------------------------------------------------------------

def coalition_game(Z, v = None):
    """
    Z : Zones
    N : list of players
    v(S) : 2^N --> R, payoff (Reward) function that can be distribute among players of coalition."""
    all_coalitions = Generating_Coalitions(Z)
    uniqe_coalitions = T_reduction(all_coalitions)
    e_core(uniqe_coalitions, Z)

def Generating_Coalitions(Z):
    """
    Z : Zones
    N : list of players
    v(S) : 2^N --> R, payoff (Reward) function that can be distribute among players of coalition."""
    N = [i for i in range(len(Z))]
    # print("List of Players (zones) = ", N)
    T = []
    for i in range(1,len(N)+1):
        for j in permutations(N):
            c = list(create_coalitions(j, i))
            # print(j, c)
            for item in c:
                T.append(item)
    # print("\n1- List of all possible coalitions (T)\n")
    # print(DataFrame(T))
    return T

    
def T_reduction(T):
    """Removing redundant coalitions"""
    R = np.zeros(len(T))
    for i in range(0, len(T)-1):
        for j in range(i+1, len(T)):
            if R[j]!=1:
                if set(T[i])== set(T[j]):
                    R[j] = 1
    newT = []
    for i in range(0,len(T)):
        if R[i]==0:
            newT.append(T[i])
    return newT


def create_coalitions(data, n):
    """
    Generating all possible combination of coalition 
    data : list of zone ids
       n : number of groups per choice"""
    from itertools import combinations, chain
    for splits in combinations(range(1, len(data)), n-1):
        result = []
        prev = None
        for split in chain(splits, [None]):
            result.append(data[prev:split])
            prev = split
        yield result
    return list(result)




def e_core(CH, Z, thr=None):
    """Return the best Choice for coalitions that provide the min epsilon value"""
    # CH: list of CHOICES for Coalition 
    # Z : Zones
    # thr : given threshold.
    
    # print(DataFrame(CH),"\n")
    
    E = []
    R = []
    # R =[(coalition, reward)]
    # Part one: Calculate rewards for all combinations of coalitions
    print('************************************************************************************************')
    for ch in CH:
        # ch : current option as a list of coalitions 
        CR = []
        for coalition in ch:
            # print("Coalition : ", coalition)

            infected = [Z[i][2] for i in coalition]
            # print("Infected : ", infected)
            # print("individual Reward   : ", [reward(i,5) for i in infected])

            coalition_reward = pre_Shapley(infected, coalition)
            
            # print("Coalition Reward : ", coalition_reward,"\n")
            
            CR.append(coalition_reward)
        R.append((ch, CR))
        # c: coalition set
    # printing(R)

    # Part Two: Calculate best coalition for each zone based on its reward:
    eps = 10
    best_choice, best_reward = R[0][0], sum_choice(R[0][1])
    for i in range(1, len(R)):
        choice, re = R[i][0], sum_choice(R[i][1])
        
        current_eps = best_reward - re
        if eps >= current_eps and current_eps > 0:
            eps = current_eps
            best_reward = re
            best_choice = choice
            print(choice,"\t", re,"\t", best_choice,"\t", best_reward,"\t", eps)
    print("Best choice of coalition: ", best_choice, best_reward)
    

                
def sum_choice(A):
    s = 0 
    for a in A:
        for i in a:
            s = s + i
    return s



def printing(D):
    for d in D:
        print(d)
    

    
    
def pre_Shapley(I, C):
    """I: infected ppl. C: Coalition"""
    Result = [reward(I[0],5)]
    S = I[0]
    for i in range(1,len(I)):
        new_S = S + I[i]
        Result.append(reward(new_S, 5) - reward(S,5))
        # print("{} - {} = {}".format(reward(new_S, 5), reward(S, 5), reward(new_S, 5) - reward(S,5)))
        S = new_S
    return Result
    
    
    
    
def M1(item, CH):
    """Returning the coalition number of item in CH"""
    print(item , CH)
    for c in CH:
        if item in c:
            return CH.index(c)
    
def M2(CH):
    """Return list of [(item: coalition)]"""
    U = []
    for i in range(len(CH)):
        for c in CH[i]:
            U.append((c, i))
        # print(i, CH[i])
    return U

            
def GDP(z):
    """ Z : number of vaccines that a player can afford."""   
    return 10 + randrange(z)

def GrossDomesticProduct(C, I, G, X, M):
    """𝐶 = Consumption, 
       𝐼 = Investment, 
       𝐺 = Government Spending, & 
       (𝑋–𝑀) = Net Exports."""
    return C + I + G + (X-M)

# -------------------------------------------------------------------------------------------------
# -----------------------------------------Shapley values------------------------------------------
# -------------------------------------------------------------------------------------------------

def start_shapley(A, C):
    """A: list of zones
    C: Coalitions"""
    # 1- Generating V values for zones and coalitions.
    Av = v_zone(A)
    print("V(A) = ", Av)
    for c in C:
        if len(c)>1:
            Av.append((c, v_coalition(c, Av)))

    print("V(A and C) = ", Av)       
    
    
def v_zone(A):
    return [([i], randrange(100)) for i in A]
    
def v_coalition(C, V):
    # C: coalition (list of zones of one coalition)
    # V: v values for all zones
    s = 0
    for i in C:
        for v in V:
            if v[0][0]==i:
                s = s + v[1]
    newS = round(s * 0.5)
    return s + randrange(newS)
        
    

def Shapley(V, C):
    """
    V: v values of Z and coalitions
    C: coalition to measure contribution of its zones based on order of arrival"""
    # print("V=", V)
    # print("C=", C)

    contribution = [get_v([C[0]], V)]
    coalition = [C[0]]
    for i in range(1, len(C)):
        new_coalition = coalition.copy()
        new_coalition.append(C[i])
        
        old_contr= get_v(coalition, V)
        new_contr= get_v(new_coalition, V)
        
        contribution.append(new_contr - old_contr )
        coalition = new_coalition.copy()
    
    return sort_tuples([(C[i], contribution[i]) for i in range(0, len(C))])
        
                    

def get_v(i, V):
    for v in V:
        if set(i) == set(v[0]):
            # print("v({}) = {}".format(i, v[1]))
            return v[1]
        

def sort_tuples(A):
    return sorted(A, key=lambda x: x[0])
