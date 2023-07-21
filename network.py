import sys
import copy
import math
import numpy as np
from utils import get_distance
from constants import *
from repository import Repository

repo = Repository()


def received_power(point1, point2, power_transmitter):
    dist = max(0.1, get_distance(point1, point2))
    tmp = LAMBDA / (4 * PI * dist)
    p = power_transmitter * DT * DR * pow(tmp, 2)
    return p


# received power for all users
def calculate_P(bss, bss_indexes, bss_weights, min_pt):
    P = np.zeros(len(bss_indexes))
    for i in range(len(bss_indexes)):
        power_transmitter = max(min_pt, PT * bss_weights[bss_indexes[i]])
        P[i] = received_power(bss[bss_indexes[i]], repo.users[i], power_transmitter)
    return P


# interference for all users
def calculate_I(bss, bss_indexes, bss_weights, min_pt):
    sum_time = 0
    I = np.zeros(len(repo.users))
    for i in range(len(repo.users)):
        sum = 0
        for j in range(len(bss)):
            if j != bss_indexes[i]:
                power_transmitter = max(min_pt, PT * bss_weights[j])
                sum += received_power(bss[j], repo.users[i], power_transmitter)
        I[i] = sum
    return I


# gama ---> SINR
# calculate gama for all users
def calc_gama(bss, bss_indexes, bss_weights, min_pt):
    gama = calculate_P(bss, bss_indexes, bss_weights, min_pt) / (
        SIGMA2 + calculate_I(bss, bss_indexes, bss_weights, min_pt)
    )
    return gama


# calculate rate for all users
def calc_R(bss, bss_indexes, bss_weights, min_pt):
    R = np.zeros(len(bss_indexes))
    gama = calc_gama(bss, bss_indexes, bss_weights, min_pt)
    R = B * (np.log2(1 + gama))
    return R


def objective_function(R):
    return np.sum(np.log10(R))


# Indexes of sorted average rate of each base station
def get_bss_priority(R, bss, bss_indexes):
    sum_R_array = np.zeros(len(bss))
    avg_R_array = np.zeros(len(bss))
    bss_user_number_array = np.zeros(len(bss))
    for i, bsi in enumerate(bss_indexes):
        sum_R_array[bsi] += R[i]
        bss_user_number_array[bsi] += 1

    for j in range(len(sum_R_array)):
        avg_R_array[j] = sum_R_array[j] / bss_user_number_array[j]

    indexes_sorted_avg_R = np.argsort(avg_R_array)

    return reversed(indexes_sorted_avg_R)


# sorting from worst SINR to best SINR users
def get_sorted_gama_index(gama):
    return np.argsort(gama)


def change_user_link(user_index, bss, bss_indexes, bss_weights, min_pt):
    bs_index = bss_indexes[user_index]
    indicator = sys.maxsize
    index = -1
    for i in range(len(bss)):
        if i != bs_index:
            power_transmitter = max(min_pt, PT * bss_weights[i])
            indicator_temp = (
                math.pow(get_distance(repo.users[user_index], bss[i]), 2)
                / power_transmitter
            )
            if indicator_temp < indicator:
                indicator = indicator_temp
                index = i
    if index == -1:
        return None
    bss_indexes_final = copy.copy(bss_indexes)
    bss_indexes_final[user_index] = index
    return bss_indexes_final


def total_power(bss_weights):
    power = 0
    for weight in bss_weights:
        power += weight * PT
    return power
