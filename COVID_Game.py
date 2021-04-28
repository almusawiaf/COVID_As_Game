# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 21:13:45 2021

@author: Ahmad Al Musawi
"""
import numpy as np
import pandas as pd
from random import randrange
from itertools import permutations
from itertools import combinations
import copy
from pandas import *
import math

z = 1
P = 1628706
# P = [1628706, 1628706, 1628706, 1628706, 1628706]
file = pd.read_csv("covid_confirmed_NY_july.csv")
mid_B = 3.0

alpha = 0.05
beta = 0.55
sigma = 0.25
gamma = 0.1

def main( Time = None):
    """
        z: number of zones
        Time: number of simulation
    """
    # creating N zones with initial random SEIRD values
    tt = input("enter number of iterations:")
    Z_Total = create_zone()
    V = create_V()
    coalition_game(Z_Total, V)
    print("-------------------------------")
    
    for _ in range(int(tt)):
        newZ = [OPT(z) for z in Z_Total]
        print(pd.DataFrame(newZ))
        V = create_V()
        coalition_game(newZ, V)
        print("-------------------------------")
        Z_Total = newZ
        
def create_V():
    """return one zone SEIRD values, t: number of zones"""
    return [100, 5, 50, 200]
    # return [randrange(500) for _ in range(t)]    

def v_function(V, C):
    print("V = {}\tC = {}".format(V, C))
    XX = sum([V[c] for c in C])
    return XX + randrange(xx*0.5)
        

def create_zone():
    """return one zone SEIRD values, t: number of zones"""
    return [[100,150, 50,  20, 10],
            [50 ,10,  5,   5,  0],
            [500,250, 100, 200,100],
            [250,100, 100, 50, 0]]
    # t = 4
    # z_total = []
    # for z in range(t):
    #     temp = []
    #     people = 250
    #     consumed = 0
    #     for i in range(5):
    #         if i< 4:
    #             consumed = randrange(people)
    #             people = people - consumed
    #             temp.append(consumed)
    #         else:
    #             temp.append(people)
    #     z_total.append(temp)
    # return z_total        
            
    

# -------------------------------------------------------------------------------------------------
# ---------------------------------------OPT + SEIRD Model---------------------------------------
# -------------------------------------------------------------------------------------------------
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

def coalition_game(Z, V):
    """
    Z : Zones
    N : list of players
    v(S) : 2^N --> R, payoff (Reward) function that can be distribute among players of coalition."""
    all_coalitions = Generating_Coalitions(Z)
    uniqe_coalitions = T_reduction(all_coalitions)
    print("--------------------------------------------------------------------")
    print("X = ", e_core2(uniqe_coalitions, Z, V))
    print("--------------------------------------------------------------------")

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
# -------------------------------------------------------------------------------------------------
# ---------------------------------------End of Coalitions Creation---------------------------------------
# -------------------------------------------------------------------------------------------------






# -------------------------------------------------------------------------------------------------
# ---------------------------------------       The Core    ---------------------------------------
# -------------------------------------------------------------------------------------------------
def e_core2(list_of_coalitions, Zones, V):
    print("Coalitions:- \n", DataFrame(list_of_coalitions))
    print("Zones:- \n", DataFrame(Zones))
    print("V-value:- \t", V)
    _ = input("press Enter to continue")
    best_coalitions = []
    flag = True
    susceptible = [i[0] for i in Zones]
    X = []
    for i in range(len(Zones)):
        R = shapley1(sum([x for x in X]), V, i)    
        K_i = susceptible[i]
        X.append(PI(R, K_i, K_i*0.1))
    print("X = ", X)
    for e in range(0,10):
        print("------------------------------------------------------")
        m = 0
        for coalitions in list_of_coalitions:
            m = m + 1
            print("\ne = ", e)
            X_Prime = []
            for C in coalitions:
                c, sh_v = shapley2(V, C)
                print("coalition({}) = {}\nC_sharpley = {}".format(m, c, sh_v))
                for i in range(len(c)):
                    z_i = c[i] # z_i : zone index
                    r_i = sh_v[i]
                    K_i = susceptible[z_i]
                    X_Prime.append((z_i, PI(r_i, K_i, K_i*0.1)))
            newX_Prime = [i[1] for i in sort_tuples(X_Prime)]
            print("Old reward = ", X)
            print("New reward = ", newX_Prime)
            s1 = sum(X)
            s2 = sum(newX_Prime)
            print("s1({}) >= s2({}) - e({})".format(s1,s2,e))
            if not (s1 >= s2-e):
                X, best_coalitions = newX_Prime.copy(), c.copy()
                print("Breaking now--------Coalition = {}, reward = {}".format(best_coalitions, X))
                flag = False
                break
        if flag:
            print("FLAG-------FLAG----------FLAG-------FLAG----------")
            print("X = ", X)
    return X, best_coalitions 

                   
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

def PI(R, ki, kith):
    """Reward function!!!"""
    rr = reward(ki, kith, R)
    print("Reward function(R = {}, ki = {}, kith = {}) = {}".format(R, ki, kith, rr))
    return rr


def reward(k, k_th, r):
    """K: number of vaccines == number of infected ppl. in zone i. k_th: min number of vaccine needed for zone
    r: result of shapley"""
    if k < k_th:
        return 0
    else:
        return math.log(k, cpv(r))

def cpv(R):
    """Cost Per Vaccine 
    As number of vaccines order increases (k), cost (C) reduced
    """
    # cpv : Cost Per Vaccine
    # k : Number of vaccine in the order of 100.000
    A = 20
    B = 10
    result = A * ( B - R + 1)
    if result>0:
        return result
    else:
        return 20
# -------------------------------------------------------------------------------------------------
# -----------------------------------    End of The Core    ---------------------------------------
# -------------------------------------------------------------------------------------------------
def sort_tuples(A):
    return sorted(A, key=lambda x: x[0])
