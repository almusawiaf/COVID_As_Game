import numpy as np
import pandas as pd
from random import randrange
import math

sigma, gamma, alphas = [0.001, 0.2], 0.1, 0.05
P = [1628706, 1628706, 1628706, 1628706, 1628706]
file = pd.read_csv("Data/covid_confirmed_NY_july.csv")
z = 45
mid_B = 3.0
beta = 0

def main():
    global beta 
    beta = beta_values()
    Z_Total = create_Z_total(10)
    for zone in range(len(Z_Total)):
        print(opt(zone, Z_Total, 0))
    

def create_Z_total(t):
    z_total = []
    for z in range(t):
        temp = []
        for i in range(5):
            temp.append(randrange(250))
        z_total.append(temp)
    return z_total        


def opt(zid, z_total, repeat):
    global beta, sigma, gamma, alphas, P
    z0 = z_total[zid]
    print("z0 = {}\nzid = {}\np[zid] = {}\nbeta[zid] = {}\n".format(z0, zid, P[zid], beta[zid]))
    
    new_infected  = sigma[repeat]*z0[1]
    
    # S: Susceptible, E: Exposed, I: Infectious, R: Recovered, D: Dead
    S = - (beta[zid] * z0[0] * z0[2]) / P[zid]
    E = (beta[zid] * z0[0] * z0[2]) / P[zid] - new_infected
    I = new_infected - gamma * z0[2]
    R = gamma * (1.0 - alphas) * z0[2]
    D = gamma * alphas * z0[2]

    z0[0] = z0[0] + S
    z0[1] = z0[1] + E
    z0[2] = z0[2] + I
    z0[3] = z0[3] + R
    z0[4] = z0[4] + D
    
    z_total[zid]=z0    
    return z_total

def beta_values():

    global file, mid_B

    B = np.array(file['Population Density'].values)
    beta_arr = [((B[i] - np.mean(B)) / np.sum(B) + 1.0) * mid_B for i in range(z)]
    return beta_arr

def CoV(A, B, K):
    # CoV : Cost of Vaccine
    # A, B : constant
    # K : Number of vaccine in the order of 100.000
    return A * ( B - K + 1)

def R_c(K, c, Th):
    # R_c : Reward of cost
    # Th : number of vaccine I must have
    if K < Th :
        return 0
    if K >= Th:
        return math.log(K, c)
    
    
    
    
    
    
    
    
    
    
    
    