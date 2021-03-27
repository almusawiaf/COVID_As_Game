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
    print("Initial Zones:\n{}".format(Z_Total))
    print(DataFrame(Z_Total))
    print("-------------------------------")

    coalition_game(Z_Total)
    print("-------------------------------")
    
    # for z in range(len(Z_Total)):
    #     current_zone = Z_Total[z]
    #     print(current_zone)
    #     infected = current_zone[2]
    #     print(Reward(infected, 10))
    # print("-------------------------------")

        # apply_OPT(current_zone, Time)



    # for t in range(1, Time):
    #     # t : time slot
    #     T.append(t)
    #     # print("t = {}".format(t))
    #     Z = []
    #     current_zone = Z_Total[-1]
    #     # print("-------------------------------")
    #     for z in range(N):
    #         newZ = OPT(z, current_zone, t)
    #         # print ("\tsum(   Z[{}])={}\n\tsum(newZ[{}])={} ".format(z, sum(current_zone[z]), z, sum(newZ)))
    #         # print ("\tdifference (sum(Z[{}]),sum(newZ[{}]))={} ".format(z, z, sum(current_zone[z])-sum(newZ)))
    #         Z.append(newZ)
    #     Z_Total.append(Z)
    # # print("-------------------------------")
    # # Plotting(Z_Total, T)
    # seird= SEIRD(Z_Total)
    # PlottingZone(Z_Total, T, 0)
    # # for m in range(0,5):
    # #     # PlottingSEIRD(seird[m], T, m)
    # #     PlottingZone(seird, T)

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




def OPT(Z, repeat):
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


        
    


# def beta_values():

#     global file, mid_B

#     B = np.array(file['Population Density'].values)
#     beta_arr = [((B[i] - np.mean(B)) / np.sum(B) + 1.0) * mid_B for i in range(z)]
#     return beta_arr


# -------------------------------------------------------------------------------------------------
# ---------------------------------------Coalitions Creation---------------------------------------
# -------------------------------------------------------------------------------------------------


def C_M(k):
    """Cost Per Vaccine 
    As number of vaccines order increases (k), cost (C) reduced
    """
    # C_M : Cost Per Vaccine
    # k : Number of vaccine in the order of 100.000
    A = 20
    B = 10
    return A * ( B - k + 1)

def reward(Ki, k_i_th):
    """
    Ki: number of vaccines == number of infected ppl. in zone i.
    k_i_th: min vaccine needed for zone"""
    if Ki < k_i_th:
        return 0
    else:
        return math.log(Ki, C_M(Ki))
    
    
def LogFunction():
    k = 5
    # k = number of vaccine that a zone orders
    # cpv = cost per vaccine
    for cpv in [20, 30, 40, 50]:
        plt.plot([k for k in range(1,10)], 
                 [Reward(k, k) for k in range(1,10)],
                 label = 'Cost per Vaccine: $' + str(cpv))
    plt.xlabel("number of vaccines purchased (in order of 100K)")
    plt.ylabel('Reward')
    
    plt.title('Minimum vaccines needed' + str(k) + '00,000')
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
    N = [i for i in range(len(Z))]
    print("N = ", N)
    T = []
    c = [list(create_coalitions(N, n)) for n in range(1, len(N))]        
    for item in c:
        for subitem in item:
            if subitem not in T:
                T.append(subitem) 
    print(DataFrame(T))
    print("-------------------------------")
    e_core(T, Z)


        
def create_coalitions(data, n):
    """
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

def e_core(CH, Z, thr=None):
    """Return the best Choice for coalitions that provide the min epsilon value"""
    # CH: CHOICES for Coalition 
    # Z : Zones
    # thr : given threshold.
    V = [GDP(i[2]) for i in Z]
    E = []
    R = []
    # R =[(zone, coalition, reward)]
    # Part one: Calculate rewards for all combinations of coalitions
    print("Choices of possible combinations", CH)
    print('************************************************************************************************')
    for ch in CH:
        # ch : current combination
        CR = []
        for coalition in ch:
            print("Coalition : ", coalition)
            infected = [Z[i][2] for i in coalition]
            zones_reward = [reward(i, i-1) for i in infected]
            print("Infected : ", infected)
            print("Zones Reward : ", zones_reward)
           
            total_reward = sum(zones_reward)
            coalition_reward = [Shapley(total_reward, zones_reward, i) for i in range(0, len(coalition))]
            print("Coalition Reward : ", coalition_reward,"\n")
            
            CR.append(coalition_reward)
        print('************************************************************************************************')
        # c: coalition set
    # Part Two: Calculate best epsilon for each zone!
    
def GDP(z):
    """ Z : number of vaccines that a player can afford."""   
    return randrange(z)
    
    
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
        print(i, CH[i])
    return U

        
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
    



def Shapley(R, V, i):
    """members should receive payments or shares proportional to their marginal contributions
    R: total reward, R = sum of reward of all player i in a coalition, 
    V: value function
    i: current zone (player)"""
    return (R-V[i])/len(V)
    
def divide_payoff():
    
    return 0
    
    
    
    
    
