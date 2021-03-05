import numpy as np
import pandas as pd
from random import randrange
import math
import matplotlib.pyplot as plt
import copy


sigma, gamma, alphas = [0.001, 0.2], 0.1, 0.05
P = [1628706, 1628706, 1628706, 1628706, 1628706]
file = pd.read_csv("covid_confirmed_NY_july.csv")
z = 45
mid_B = 3.0
beta = 0

def main():
    global beta 
    beta = beta_values()
    # creating 10 zones with initial random SEIRD values
    Z_Total = create_Z_total(10)
    print("number of zones:", len(Z_Total))
    print("-------------------------------")
    
    for zone in range(len(Z_Total)):
        Z_Total  = opt(zone, Z_Total, 0)

def create_Z_total(t):
    # t: number of zones
    z_total = []
    for z in range(t):
        temp = []
        for i in range(5):
            temp.append(randrange(250))
        z_total.append(temp)
    return z_total        


def opt(zid, z_total, repeat):
    global beta, sigma, gamma, alphas, P
    z0 = copy.deepcopy(z_total[zid])
    newP = P[randrange(len(P))]
    print("zid = {}\np[zid] = {}\nbeta[zid] = {}\nz0 = {}\n".format(zid, newP, beta[zid], z0))
    
    new_infected  = sigma[repeat]*z0[1]

    # S: Susceptible, E: Exposed, I: Infectious, R: Recovered, D: Dead
    S = - (beta[zid] * z0[0] * z0[2]) /newP
    E = (beta[zid] * z0[0] * z0[2]) / newP - new_infected
    I = new_infected - gamma * z0[2]
    R = gamma * (1.0 - alphas) * z0[2]
    D = gamma * alphas * z0[2]

    z0[0] = z0[0] + S
    z0[1] = z0[1] + E
    z0[2] = z0[2] + I
    z0[3] = z0[3] + R
    z0[4] = z0[4] + D
    
    print("new Z0 = {}\n--------------------------\n".format(z0))
    z_total[zid]=copy.deepcopy(z0)
    return z_total

def beta_values():

    global file, mid_B

    B = np.array(file['Population Density'].values)
    beta_arr = [((B[i] - np.mean(B)) / np.sum(B) + 1.0) * mid_B for i in range(z)]
    return beta_arr

def CPV(A, B, nov):
    """Cost Per Vaccine 
    As number of vaccines order increases (nov), cost (C) reduced
    """
    # CPV : Cost Per Vaccine
    # A, B : constant
    # nov : Number of vaccine in the order of 100.000
    return A * ( B - nov + 1)

def Reward(nov, cpv, pi):
    """
    Reward of cost
    nov : number of vaccine a zone order
    cpv : cost per vaccine
    PI : number of vaccine I must have
    """
    if nov < pi :
        return 0
    if nov >= pi:
        return math.log(nov, cpv)
    
    
def LogFunction():
    k = 5
    # nov = number of vaccine that a zone orders
    # cpv = cost per vaccine
    for cpv in [20, 30, 40, 50]:
        plt.plot([nov for nov in range(1,10)], 
                 [Reward(nov, cpv, k) for nov in range(1,10)],
                 label = 'Cost per Vaccine: $' + str(cpv))
    plt.xlabel("number of vaccines purchased (in order of 100K)")
    plt.ylabel('Reward')
    
    plt.title('Minimum vaccines needed' + str(k) + '00,000')
    plt.legend()
    plt.show()

def coalition_game(N, v):
    """N : list of players
    v(S) : 2^N --> R, payoff function that can be distribute among players of coalition."""
    S = Shapley(N)
    return S

def Shapley(N):
    """members should receive payments or shares proportional to their marginal contributions"""
    if len(N)==0:
        return 0
    
def divide_payoff():
    
    return 0
    
    
    
    
    
    
