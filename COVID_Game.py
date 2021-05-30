# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 21:13:45 2021

@author: Ahmad Al Musawi
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from random import randrange
from itertools import permutations
import math
from shapley import Shapley

# from itertools import combinations
# import copy
# from pandas import *


z = 1
P = 1628706
# P = [1628706, 1628706, 1628706, 1628706, 1628706]
file = pd.read_csv("covid_confirmed_NY_july.csv")
mid_B = 3.0

alpha = 0.05
beta = 0.55
sigma = 0.25
gamma = 0.1

Z, GDP = [],[]
boroughs = ['The Bronx','Brooklyn','Manhattan','Queens','Staten Island']
ppl =      [1418207    , 2559903  , 1628706   , 2253858, 476143]
GDP =      [42.695,      91.559,    600.244,    93.310,  14.514]

VB = [0,0,0,0,0]


my_path = 'D:/Documents/Research Projects/Complex Networks Researches/COVID/COVID_as_GAME/Results/'

    
# Z = [[100,150, 50,  20, 10],
#      [50 ,10,  5,   5,  0],
#      [500,250, 100, 200,100]]
# GDP = [100, 5,  50]


# Z = [[100,150, 50,  20, 10],
#       [50 ,10,  5,   5,  0],
#       [500,250, 100, 200,100],
#       [250,100, 100, 50, 0]]
# GDP = [100, 5, 50, 200 ]



def main():
    global Z, GDP, boroughs, ppl, VB

    Z = set_zones(ppl)
    print(DataFrame(ppl))
    print("--------------------------------------------------------------------")
    print(DataFrame(Z))

    uniqe_coalitions = Generating_Coalitions(Z)
    Results = ''

    _ = input("Check Zones statistics? (Enter)")

    tt = input("enter number of iterations:")
    
    Q = 1000
    V = [i*Q for i in GDP]
    for L in range(int(tt)):
        print()
        patients = [i[1] for i in Z]
        newZ = [OPT(z) for z in Z]
        X_Prime, final_C, selected_e, final_R = e_core3(uniqe_coalitions, patients, GDP, V)

        # Now we should process the vaccines to the susceptibles.
        Results = Results + "{} - C = {}\tR = {}\te = {}\tX' = {}\n".format(L, final_C, final_R, selected_e, X_Prime)
        print("{} - C = {}\nR = {}\ne = {}\nX' = {}\n".format(L, final_C, final_R, selected_e, X_Prime))
        # print(pd.DataFrame(newZ))

        M = after_coalition(final_C, GDP, V)
        print('V = {}\nCons. = {}'.format(V, [i[1] for i in M]))
        
        VB = [VB[i]+(V[i]-M[i][1]) for i in range(len(V))]
        print('New VB=', VB)
        
        Z = newZ.copy()
        
        
        
    print('Mission Completed.')
    ff = input("Save?(y/n)")
    if ff=='y':
        gg = input('Enter file name:')
        f = open(my_path + gg+'.txt', "w")
        f.write(Results)
        f.close()

def after_coalition(C, GDP, V):
    res = []
    for c in C:
        s = sum([GDP[i] for i in c])
        R = CM(s)
        print("c = ", c,'\tsum = ', s,'\tR [CM(S)]= ', R)

        for i in c:
            res = res + [(i, V[i]/R)]
    return sort_tuples(res)
            
    
    
def set_zones(ppl):
    Z = []
    for p in ppl:
        found = False
        while not found:            
            i = randrange(round(p*0.25))
            e = randrange(i*4)
            r = randrange(round(i*0.8))
            d = randrange(round(i*0.02))
            s = p-i-e-r-d
            if s>0 :
                found = True
                Z.append([s,e,i,r,d])
    return Z



def get_zone(a):
    print(a)

    return Z, GDP
        
        
# -------------------------------------------------------------------------------------------------
# --------------------------------------- OPT + SEIRD Model ---------------------------------------
# -------------------------------------------------------------------------------------------------


def OPT(Z):
    global beta, sigma, gamma, alpha, P
    
    # if sum(Z)>100 or sum(Z)<99:
    #     print("Z = {}".format(sum(Z)))
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

# -------------------------------------------------------------------------------------------------
# ---------------------------------------Coalitions Creation---------------------------------------
# -------------------------------------------------------------------------------------------------


def Generating_Coalitions(Z):
    """
    Z : Zones
    N : list of players
    v(S) : 2^N --> R, payoff (Reward) function that can be distribute among players of coalition."""
    print("Generating list of coalitions....")
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
    return T_reduction(T)

    
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
# -------------------------------------------------------------------------------------------------
# ---------------------------------------End of Coalitions Creation---------------------------------------
# -------------------------------------------------------------------------------------------------



def GetReward(C, patients, GDP, V):
    # print("----------------------------------------------------------------------")
    newV = []
    newR = []
    K_C = S_coalition(C, patients)
    K_th = [i*0.15 for i in patients]
    
    Kth_C = S_coalition(C, K_th)
    V_C = S_coalition(C, V)
    GDP_C = S_coalition(C, GDP)
    # print("C = {}\nV(C) = {}\nGDP(C) = {}\nK(C) = {}--(total patients)\nK_th(C) = {}".format(C, V_C, GDP_C, K_C, Kth_C))
    for i in range(len(K_C)):
        zoneID = K_C[i][0]
        kth = Kth_C[i][1]
        R = CM(GDP_C[i][1])
        k = V_C[i][1]/R       
        Xi = PI(R, k, kth)
        newV.append((zoneID, Xi))
        newR.append((zoneID, R))
    # print("new V = ", newV)
    return newV, newR


def get_details_C(C, patients, GDP, V):
    # print("----------------------------------------------------------------------")
    res = []
    K_C = S_coalition(C, patients)
    K_th = [i*0.15 for i in patients]
    
    Kth_C = S_coalition(C, K_th)
    V_C = S_coalition(C, V)
    GDP_C = S_coalition(C, GDP)
    # print("C = {}\nV(C) = {}\nGDP(C) = {}\nK(C) = {}--(total patients)\nK_th(C) = {}".format(C, V_C, GDP_C, K_C, Kth_C))
    for i in range(len(K_C)):
        zoneID = K_C[i][0]
        kth = Kth_C[i][1]
        R = CM(GDP_C[i][1])
        k = V_C[i][1]/R       
        res.append((zoneID, R, k, kth))
    return res
    
    

# -------------------------------------------------------------------------------------------------
# ---------------------------------------       The Core    ---------------------------------------
# -------------------------------------------------------------------------------------------------
def e_core3(list_of_coalitions, patients, GDP, V):
    final_C = list_of_coalitions[-1] # (1), (2), (3), (4), (5)
    
    newV = []
    resulted_shapley = []
    final_R = []

    for C in final_C:
        newV, newR = GetReward(C, patients, GDP, V)
        # print("Reward of {} = {}".format(C, newV))
        resulted_shapley.append(Shapley(newV + V, C))

    X_Prime = get_net_reward(resulted_shapley)
    selected_e = 0
    for ep in range(25):
        e = ep / 10

        for coalitions in list_of_coalitions:            
            newV = [] 
            newR = []
            resulted_shapley = []
            for C in coalitions:
                newV, R  = GetReward(C, patients, GDP, V)
                newR.append(R)
                resulted_shapley.append(Shapley(newV + V, C))
            X = get_net_reward(resulted_shapley)

            for C in coalitions:
                flag = False
                sX = sum([X[c][1] for c in C])
                sXP = sum([X_Prime[c][1]-e for c in C])
                if  not (sX >= sXP):
                    X_Prime, final_C = X.copy(), coalitions.copy()
                    final_R = newR
                    selected_e = e
                    flag = True
                if flag:
                    break
    return X_Prime, final_C, selected_e, final_R

            

def get_net_reward(SH):
    res = []
    for sh in SH:
        for subsh in sh:
            res.append(subsh)
    return sort_tuples(res)
        
        
def S_coalition(C, S):
    """C: Coalition, S: susceptible list
    return sum of susceptible for the C coalition"""
    result = []
    # print(C)
    for i in range(1, len(C)+1):
        m = list(C[0:i])
        ssum = sum([S[c] for c in m])
        result.append((m, ssum))
    return result


def kith_coalition(C, available_vaccines):
    """C: Coalition, available vaccine:-)
    return sum of available vaccines for coalitions"""
    # print(C)
    result = []
    for i in range(1, len(C)+1):
        m = list(C[0:i])
        s = sum([available_vaccines[i] for i in m])
        result.append((m, s))
    return result

    
def Sum_coalition(C, S):
    """C: Coalition, S: susceptible list
    return sum of susceptible for the C coalition"""
    result = []
    # print(C)
    for i in range(1, len(C)+1):
        m = list(C[0:i])
        ssum = sum([S[c] for c in m])
        result.append((m, ssum))
    return result



# -------------------------------------------------------------------------------------------------
# -----------------------------------    PI, Reward and CM    ------------------------------------
# -------------------------------------------------------------------------------------------------

def PI(R, ki, kith):
    """Reward function!!!
    R: number of vaccine to buy!
    k_i: number of vaccines
    kith: minimum number of vaccine"""
    result = 0
    if ki < kith:
        result =  0
    else:
        result = math.log(ki, R)
    # print("Reward function(R = {}, ki = {}, kith = {}) = {}".format(R, ki, kith, result))
    return result


def CM(number_of_vaccine):
    """Cost Per Vaccine 
    As number of vaccines order increases (k), cost (C) reduced
    """
    # CM : Cost Per Vaccine
    # k : Number of vaccine in the order of 100.000
    A = 20
    B = 10
    result = A * ( B - number_of_vaccine + 1)
    if result>0:
        return result
    else:
        return 20
# -------------------------------------------------------------------------------------------------
# -----------------------------------    helpful functions    -------------------------------------
# -------------------------------------------------------------------------------------------------
def sort_tuples(A):
    return sorted(A, key=lambda x: x[0])

# -------------------------------------------------------------------------------------------------
# -----------------------------------    To be deleted if not needed   ----------------------------
# -------------------------------------------------------------------------------------------------


# def apply_OPT(Z, Time):
#     newZ = []
#     newZ.append(Z)
#     # print(newZ)
#     T = [0]
#     for t in range(1, Time):
#         T.append(t)
#         newZ.append(OPT(newZ[-1], t))
#     print(DataFrame(newZ))
#     PlottingZone(newZ, T)
#     return newZ





# def v_function(V, C):
#     print("V = {}\tC = {}".format(V, C))
#     XX = sum([V[c] for c in C])
#     return XX + randrange(xx*0.5)


def create_zone():
    """return one zone SEIRD values, t: number of zones"""
    t = 3
    z_total = []
    for z in range(t):
        temp = []
        people = 250
        consumed = 0
        for i in range(5):
            if i< 4:
                consumed = randrange(people)
                people = people - consumed
                temp.append(consumed)
            else:
                temp.append(people)
        z_total.append(temp)
    return z_total, [randrange(500) for _ in range(t)]    



# -------------------------------------------------------------------------------------------------
# ------------------------------------------    Shapley   -----------------------------------------
# -------------------------------------------------------------------------------------------------

def shapley1(R, V, i):
    return R-V[i]

def shapley2(V, C):
    """V: v function, C: Coalition, i: player"""
    c = [C[0]]
    v = [V[C[0]]]
    for i in range(1, len(C)):
        c.append(C[i])
        v.append(sum([V[cc] for cc in c]) + randrange(75)- V[i])
    # print("\ncoalition = ", C)
    # print ("shapley = ", v)
    return c, v

def shapley3(V, C):
    """distribute the reward based on their contribution in C,
    V: v function, C: Coalition, i: player"""
    print(V)
    print(C)
    c = [C[0]]
    v = [V[C[0]]]
    newV = []
    for c in C:
        newV.append()
    for i in range(1, len(C)):
        c.append(C[i])
        v.append(sum([V[cc] for cc in c]) + randrange(75)- V[i])
    # print("\ncoalition = ", C)
    # print ("shapley = ", v)
    return c, v



def groups(a):
    import itertools
    group = []
    for i in range(1,len(a)+1):
       b = list(itertools.combinations(a,i))
       for x in b:
           group.append(x)
    return(group)

def coalition(a, groups):
    coalitions = []    
    for i in range(0, len(groups)-1):
        x = groups[i]
        items1 = set(x)
        C = [x]
        for j in range(i+1, len(groups)):
            y = groups[j]
            items2 = set(y)
            if items1.intersection(items2) == 0 :
                C.append(y)
        print(x,'\t',y, C)
            
    
# def GetReward(sus, budget):
#     """based on total number of susceptable, we count the reward.
#     reward depends on the number of susceptible of given zone."""  
    
#     newV = []
#     print("\nsusceptible = ", sus)
#     print("budget = ", budget)
#     for i in range(len(sus)):
#         s = sus[i] #(zone id, zone susceptibles)
#         zone_id = s[0]
#         R = s[1]*budget[i][1]
#         ki = s[1]
#         kith = s[1]*0.1
#         XI = PI(R, ki, kith)
#         newV.append((zone_id, XI))
#     return newV
    
    
    
