import pulp
import random
import networkx as nx
import simpy
import numpy as np
import pickle
import random
import math
import operator
import itertools
import pandas as pd
import decimal
import time

import seaborn as sns

from copy import *
from geopy import distance
from scipy import optimize
from sklearn.cluster import KMeans
from scipy.spatial.distance import *
from scipy import stats
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import silhouette_score, calinski_harabasz_score

np.printoptions(precision = 2)


def pearsonr_ci(L, alpha=0.05):

    x = [pt[0] for pt in L]
    y = [pt[1] for pt in L]

    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi


def find_queue_probability(b_GDP, M, mus, mode):

    L = []
    for i in range(b_GDP):
        if mode == 0:
            lambdas = M[i, T]
        else:
            lambdas = np.mean(M[i, T - int(window): T])

        rho = float(lambdas) / float(mus)

        if rho <= 1:
            p0 = 1.0 - rho
            p1 = p0 * rho
            pq = 1.0 - (p0 + p1)

        else:
            pq = 1.0

        L.append(pq)

    mean_pq = np.mean(L)
    return mean_pq


def find_reward(kappa, mean_pq):

    # global v, small_reward
    global kappas, small_reward

    # return math.log(beta + small_reward)/(mean_pq + small_reward)
    # return (v.index(velocity) + 1) * math.exp(- int(math.floor(mean_pq * 10)))
    # return float(v.index(velocity) + 1)/float(len(v)) * math.exp(- mean_pq)
    return float(kappa)/(max(kappas)) * math.exp(- mean_pq)


def combine_heatmaps(M1, M2, B):

    df = pd.DataFrame(M1)
    df2 = pd.DataFrame(M2)

    fig, (ax, ax2) = plt.subplots(ncols=2)
    fig.subplots_adjust(wspace=0.01)
    sns.heatmap(df, cmap ="icefire", ax=ax, cbar=False)
    fig.colorbar(ax.collections[0], ax=ax, location="left", use_gridspec=False, pad=0.2)
    sns.heatmap(df2, cmap="icefire", ax=ax2, cbar=False)
    fig.colorbar(ax2.collections[0], ax=ax2, location="right", use_gridspec=False, pad=0.2)
    ax2.yaxis.tick_right()
    ax2.tick_params(rotation = 0)

    ax.set_title('Mobility')
    ax2.set_title('Migration')

    ax.set_xticks([i + 0.5 for i in range(5)])
    ax.set_xticklabels(['Man', 'Brx', 'Bkln', 'Qns', 'SI'], fontsize = 7)
    ax.set_yticks([i + 0.5 for i in range(5)])
    ax.set_yticklabels(['Man', 'Brx', 'Bkln', 'Qns', 'SI'], fontsize = 10)

    ax2.set_xticks([i + 0.5 for i in range(5)])
    ax2.set_xticklabels(['Man', 'Brx', 'Bkln', 'Qns', 'SI'], fontsize = 7)
    ax2.set_yticks([i + 0.5 for i in range(5)])
    ax2.set_yticklabels(['Man', 'Brx', 'Bkln', 'Qns', 'SI'], fontsize = 10)

    # plt.tight_layout()
    plt.savefig('Migrate.png', dpi = 300)
    plt.show()


class Node(object):

    def __init__(self, env, ID, kappa, dense, Z, pop, b_GDP, Q):
        global T, Duration

        self.ID = ID
        self.env = env
        self.z_total = Z

        # self.velocity = velocity
        self.kappa = kappa

        self.dense = dense
        self.pop = pop
        self.Q = Q

        # Number of ICUs proportional to GDP
        self.b_GDP = int(b_GDP)
        self.bed_queue = {j: [] for j in range(self.b_GDP)}

        self.last_state = (0, 0)
        self.current_state = (0, 0)
        self.next_state = None
        self.reward = 0

        self.beta = 1.0
        self.last_mean_pq = None

        # Arrival rates per bed
        self.M = np.zeros((self.b_GDP, Duration + 1))

        if self.ID == 1:
            self.env.process(self.time_increment())

        self.env.process(self.opt())
        self.env.process(self.inject_new_infection())
        self.env.process(self.treatment())
        # self.env.process(self.move())
        self.env.process(self.learn())

    def time_increment(self):

        global T, iho, epsilon, decay, window, x, yR, yV, v, Z_list, T_list, eB, Proportions

        while True:

            T = T + 1
            if T % 100 == 0:
                for i in range(eB):
                    Proportions[i].append(sum(entities[i].z_total))

            # print (T)
            # if T % iho == 0:
            #     print (T, epsilon)

            if T > 0 and T % window == 0:
                epsilon *= decay

            # if T % 10 == 0:
            #     Z_list.append([entities[b].z_total[2] for b in range(eB)])
            #     T_list.append(T)

            # if T > 0 and T % (100 * 24) == 0:
            #     epsilon = 0.6
            #     self.Q = [np.zeros((state_size, state_size)) for _ in range(len(capacity))]

            yield self.env.timeout(minimumWaitingTime)

    def treatment(self):

        global T, iho, epsilon, decay, window, recovery_time, hospital_recover

        while True:

            for i in range(self.b_GDP):
                if len(self.bed_queue[i]) == 0:
                    continue

                t = self.bed_queue[i][0]
                if t - T > recovery_time:
                    self.bed_queue[i].pop(0)
                    if random.uniform(0, 1) < hospital_recover:
                        self.z_total = [self.z_total[0], self.z_total[1], self.z_total[2] - 1,
                                        self.z_total[3] + 1, self.z_total[4]]
                    else:
                        self.z_total = [self.z_total[0], self.z_total[1], self.z_total[2] - 1,
                                        self.z_total[3], self.z_total[4] + 1]

            yield self.env.timeout(minimumWaitingTime)

    def inject_new_infection(self):

        global pI, cI

        while True:
            if T % fI == 0:
                n = float(cI) / float(T + 1) * 5
                self.z_total = [self.z_total[0], self.z_total[1], self.z_total[2] + n, self.z_total[3], self.z_total[4]]

            yield self.env.timeout(minimumWaitingTime)

    def opt(self):

        global iho, sigma, gamma, alphas, rho, p, pH, DHP, kappas
        while True:

            if T % iho == 0:

                if T % (20 * iho) == 0:

                    # Note p_queue
                    # PQ[self.ID].append((T, self.last_mean_pq))
                    PQ[self.ID].append((T, kappas.index(self.kappa)))

                if self.ID == 2 and T % 300 == 0:
                    print ('Before SEIRD:', T, np.sum(self.z_total), [int(val) for val in self.z_total])

                z0 = deepcopy(self.z_total)

                # self.beta = p * math.sqrt(2) * math.pi * self.velocity * (self.dense * math.pow(10, -6)) * 1
                self.beta = self.kappa * self.dense

                new_infected = sigma * z0[1]

                update0 = - (self.beta * z0[0] * z0[2]) / self.pop
                update1 = (self.beta * z0[0] * z0[2]) / self.pop - new_infected
                update2 = new_infected - gamma * z0[2]
                update3 = gamma * (1.0 - alphas) * z0[2]
                update4 = gamma * alphas * z0[2]

                z0[0] = z0[0] + update0

                z0[1] = z0[1] + update1
                z0[2] = z0[2] + update2
                z0[3] = z0[3] + update3
                z0[4] = z0[4] + update4

                self.z_total = deepcopy(z0)
                if self.ID == 2 and T % 300 == 0:
                    print ('Afters SEIRD: T=', T, np.sum(self.z_total), [int(val) for val in self.z_total], '\n')

                # Number of patients hospitalized
                nH = int(pH * new_infected)

                # Empty beds
                empty_beds = []
                for j in range(self.b_GDP):
                    if len(self.bed_queue[j]) == 0:
                        empty_beds.append(j)

                # Assign random hospital beds to patients if no beds are empty
                for i in range(nH):
                    if len(empty_beds) > 0:
                        bed = empty_beds.pop(0)
                    else:
                        bed = int(random.uniform(0, self.b_GDP - 1))

                    self.M[bed, T] += 1
                    self.bed_queue[bed].append(T)

                # Find mean probability of queue (pq)
                mean_pq = find_queue_probability(self.b_GDP, self.M, mus, 0)
                DHP[self.ID].append((nH, mean_pq, T, self.kappa))

                # self.reward = find_reward(self.velocity, mean_pq)
                self.reward = find_reward(self.kappa, mean_pq)

                # Find capacity index
                cp = None
                for cp in range(len(capacity)):
                    if mean_pq >= capacity[cp]:
                        break

                # self.Q[cp, v.index(self.velocity)] += self.reward
                self.Q[cp, kappas.index(self.kappa)] += self.reward

            yield self.env.timeout(minimumWaitingTime)

    def move(self):

        global mP, Mobility, B, Mobility_Trace, PQ
        while True:

            if T > 10 and T % window == 0:

                if self.ID == 2:
                    print ('Before Migrate:', T, np.sum(self.z_total), [int(val) for val in self.z_total])

                how_many_moving = int(mP * np.sum(self.z_total))
                for mover in range(how_many_moving):
                    destination = np.random.choice([i for i in range(len(B.keys()))], p = Mobility[:, self.ID], size = 1)
                    destination = destination[0]

                    epidemic_status = np.random.choice([i for i in range(4)],
                                                       p = [val/np.sum(self.z_total[:4]) for val in self.z_total[:4]],
                                                       size = 1)
                    epidemic_status = epidemic_status[0]

                    self.z_total[epidemic_status] -= 1
                    entities[destination].z_total[epidemic_status] += 1

                    Mobility_Trace[destination, self.ID] += 1

                if self.ID == 2:
                    print ('After Migrate:', T, np.sum(self.z_total), [int(val) for val in self.z_total])

            yield self.env.timeout(minimumWaitingTime)

    def learn(self):
        global epsilon, state_size, lr, gamma_RL, mus, window, v, \
               kappas, capacity, small_reward, yV, thr

        while True:
            if T > 10 and T % window == 0:

                mean_pq = find_queue_probability(self.b_GDP, self.M, mus, 1)

                if self.last_mean_pq is not None and abs(self.last_mean_pq - mean_pq) < thr:
                    self.last_mean_pq = mean_pq
                else:

                    # Find capacity index
                    cp = None
                    for cp in range(len(capacity)):
                        if mean_pq >= capacity[cp]:
                            break

                    if random.uniform(0, 1) < epsilon:
                        self.kappa = kappas[random.choice([i for i in range(state_size)])]
                    else:
                        self.kappa = kappas[np.argmax(self.Q[cp])]

                    if self.ID == 4:
                        yV.append(self.kappa)
                        yR.append(mean_pq)
                        x.append(T)

                    self.last_mean_pq = mean_pq

            yield self.env.timeout(minimumWaitingTime)


# Number of boroughs
eB = 5

# Note PQ
PQ = {i: [] for i in range(eB)}

# Interact how often in hours
iho = 12

# Simulation time in hours
Duration = 24 * 60

# Fraction of patients needing hospitalization
pH = 0.2

# SEIRD parameters
sigma, gamma, alphas = 0.25, 0.5, 0.05

# Minimum waiting time
minimumWaitingTime = 1

# B = {0: 'Bronx', 1: 'Brooklyn', 2: 'Manhattan', 3: 'Queens', 4: 'Staten Island'}
B = {0: 'Manhattan', 1: 'Bronx', 2: 'Brooklyn', 3: 'Queens', 4: 'Staten Island'}

# Mobility matrix
# Mobility = np.random.rand(len(B.keys()), len(B.keys()))
# Mobility = Mobility / Mobility.sum(axis = 0)

# the mobility matrix
Mobility =  np.array([[0.2, 0.2, 0.2, 0.2, 0.2],
                      [0.2, 0.2, 0.2, 0.2, 0.2],
                      [0.2, 0.2, 0.2, 0.2, 0.2],
                      [0.2, 0.2, 0.2, 0.2, 0.2],
                      [0.2, 0.2, 0.2, 0.2, 0.2]])

Mobility_Trace = np.zeros([eB, eB])
Proportions = [[] for i in range(eB)]

# Percentage of people moving
mP = 0.01

# Real trends in infection in NYC
actual_inf = [5465, 3878, 1889, 873, 665, 376, 442, 432, 349]
actual_inf = [float(v)/float(eB) for v in actual_inf]

# Population
# P = [1418207, 2559903, 1628706, 2253858, 476143]
# P = [1628706, 1418207, 2559903, 2253858, 476143]
P = [1628706, 1628706, 1628706, 1628706, 1628706]

# Infected
# I = [50120, 63086, 30921, 68548, 14909]
# I = [40857, 59265, 82090, 83311, 20022]
I = [59265, 59265, 59265, 59265, 59265]

# I = [actual_inf[0] for i in range(eB)]

# Death
# D = [4865, 7257, 3149, 7195, 1077]
D = [0, 0, 0, 0, 0]

# Beta parameters
# Density
# density = [13957, 13957, 13957, 13957, 13957]
density = [10000, 20000, 30000, 40000, 50000]

kappas = [0 + (i * 4.4428829381583655e-06) for i in range(1, 11, 3)]
print (kappas)
input('')

# Infection rate
p = 0.01

# Proportion of exposed
pe = 1.82911550e-04

# monitoring system for duration
small_reward = 0.00001
window = 10 * iho + iho/2

# Threshold to invoke RL
thr = 0

# Action space
# Velocity
v = [100.0, 250.0, 400.0, 650.0, 800.0, 1000.0]

# Queue probability less than
capacity = [0.66, 0.33, 0.0]
state_size = len(kappas)
indices = [(i, j) for i in range(len(capacity)) for j in range(len(v))]
# print (indices)
# exit(1)

# Percentage and count of new infections
pI = 0.0005
cI = 100000

# Frequency of new infections
fI = 24 * 30

# GDP parameter (in billion USD)
mid_B = 100
# GDP = [100, 300, 500, 700, 900]
GDP = [300, 300, 300, 300, 300]

# queue model parameter
recovery_time = 14.0
hospital_recover = 0.8
mus = 1.0/float(recovery_time * 24)

iterate = 100

v_change = []
for iter in range(1):

    # Set the percent you want to explore (RL)
    epsilon = 0.75
    decay = 0.99
    lr = 0.3
    gamma_RL = 0.8

    Z_list, T_list = [], []
    print ('Iteration:', iter)

    # List for hospitalization and p(queue) correlation over time for each borough
    DHP = {b: [] for b in range(eB)}

    # Global time
    T = 0

    # Initial population
    Z = []
    for b in B.keys():
        zb = [P[b] - (pe * P[b] + I[b] + D[b]), pe * P[b], I[b], 0, D[b]]
        # print (zb, sum(zb))
        Z.append(zb)

    # Visualization
    yR = []
    yV = []
    x = []

    # Create SimPy environment and assign nodes to it.
    env = simpy.Environment()
    # entities = [Node(env, i, v[0], density[i], Z[i], P[i], GDP[i], np.zeros((len(capacity), state_size))) for i in range(eB)]
    entities = [Node(env, i, kappas[0], density[i], Z[i], P[i], GDP[i], np.zeros((len(capacity), state_size))) for i in range(eB)]
    env.run(until = Duration)

    for i in range(eB):
        if i == 0 or i == eB - 1:
            L = PQ[i]
            x = [pt[0] for pt in L]
            y = [pt[1] for pt in L]

            plt.plot(x, y, label = str('Zone' + str(i)) + ' density level' + str(i + 1) + ' units', marker = 'o', linewidth = i + 1)

    plt.xlabel('Time', fontsize = 12)
    plt.ylabel('Kappa', fontsize = 12)
    plt.legend()

    plt.tight_layout()
    plt.savefig('Dense_Disc.png', dpi = 300)
    plt.show()
