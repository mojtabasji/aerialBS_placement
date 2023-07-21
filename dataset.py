import sys
import json
import numpy as np
import math
from random import randint
from constants import *
from utils import get_distance, cluster
from network import calc_gama, get_sorted_gama_index
from repository import Repository

repo = Repository()


def generate_users(width, height, number_users):
    window_size = np.array([width, height])
    users = np.random.rand(number_users, 2) * window_size
    return users


def is_attachable(region_list, pos, small_width, small_height):
    attachable = True
    for r in region_list:
        x1 = r[0] * small_width
        x2 = x1 + small_width
        y1 = r[1] * small_height
        y2 = y1 + small_height
        if pos[0] > x1 and pos[0] < x2 and pos[1] > y1 and pos[1] < y2:
            attachable = False
            break
    return attachable


def make_empty_region(
    users, width, height, grid_w, grid_h, number_empty, region_list=None
):
    new_users = []
    small_width, small_height = (width / grid_w), (height / grid_h)
    if region_list is None:
        region_list = []
        for i in range(number_empty):
            while True:
                random_x = randint(0, grid_w - 1)
                random_y = randint(0, grid_h - 1)
                if (random_x, random_y) not in region_list:
                    break
            region_list.append((random_x, random_y))
    for user in users:
        if is_attachable(region_list, user, small_width, small_height):
            new_users.append(user)

    return np.array(new_users)


def set_dataset_bss(dataset_index):
    if dataset_index:
        number_bs = "bss" + str(dataset_index + 1)
        with open(BSS_PATH) as bss_file:
            ground_bss = np.array(json.load(bss_file)[str(number_bs)])
        return ground_bss
    return None


def set_dataset_users(dataset_index):
    path = USERS_PATH_BASE + str(dataset_index + 1) + ".json"
    with open(path) as users_file:
        users = np.array(json.load(users_file))
    return users


# assigning each user to nearest bss of itself
def user_association(bss):
    ground_bss_indexes = []
    for user in repo.users:
        index = -1
        dist = sys.maxsize
        for j, bs in enumerate(bss):
            dist2 = get_distance(user, bs)
            if dist2 < dist:
                index = j
                dist = dist2
        ground_bss_indexes.append(index)
    return ground_bss_indexes


# Selecting users according to their SINR ,and disconnect them from their bss
def specify_aerial_users(
    mixed_bss, mixed_association_array, mixed_bss_weights, min_pt, aerial_user_number
):
    new_association_array = mixed_association_array.copy()
    gama = calc_gama(
        mixed_bss,
        mixed_association_array,
        mixed_bss_weights,
        min_pt,
    )
    sorted_gama_index = get_sorted_gama_index(gama)
    aerial_user_indexes = sorted_gama_index[0:aerial_user_number]
    for user_index in aerial_user_indexes:
        new_association_array[user_index] = -1
    return aerial_user_indexes, np.array(new_association_array)


# locating aerial bss using clustering
def setup_aerial_bss_auto(
    aerial_bss_number,
    mixed_bss=None,
    mixed_association_array=None,
    mixed_bss_weights=None,
    worst_users_percent=None,
):
    aerial_bss = np.array([[0, 0] for i in range(aerial_bss_number)])
    mixed_bss_number = 0 if (mixed_bss is None) else len(mixed_bss)
    aerial_bss_indexes = [i + mixed_bss_number for i in range(aerial_bss_number)]
    if mixed_bss is not None:
        bss = np.concatenate((mixed_bss, aerial_bss), axis=0)
        aerial_user_number = int(len(repo.users) * worst_users_percent)
        min_pt = PIRU * PT
        aerial_user_indexes, association_array = specify_aerial_users(
            mixed_bss,
            mixed_association_array,
            mixed_bss_weights,
            min_pt,
            aerial_user_number,
        )
        aerial_users = []
        for index in aerial_user_indexes:
            aerial_users.append(repo.users[index])
        aerial_association_array, centroids = cluster(
            np.array(aerial_users), aerial_bss_number, np.array(aerial_bss)
        )
        for j, index in enumerate(aerial_user_indexes):
            association_array[index] = aerial_bss_indexes[aerial_association_array[j]]
        for j, index in enumerate(aerial_bss_indexes):
            bss[index] = centroids[j]
    else:
        aerial_user_indexes = [i for i in range(len(repo.users))]
        bss = aerial_bss.copy()
        association_array, centroids = cluster(
            np.array(repo.users), len(bss), np.array(bss)
        )
        bss = centroids.copy()
    return bss, association_array, aerial_bss_indexes, aerial_user_indexes


def get_aerial_user_indexes(association_array, aerial_bss_indexes):
    aerial_user_indexes = []
    for i in range(len(association_array)):
        if association_array[i] in aerial_bss_indexes:
            aerial_user_indexes.append(i)
    return np.array(aerial_user_indexes)


# Setup aerial bss based on specified location and allocate users based on distance from all bss
def setup_aerial_bss_manual(ground_bss, aerial_bss, ground_association_array):
    aerial_bss_indexes = [i + len(ground_bss) for i in range(len(aerial_bss))]
    bss = np.concatenate((ground_bss, aerial_bss), axis=0)
    association_array = user_association(bss)
    aerial_user_indexes = get_aerial_user_indexes(association_array, aerial_bss_indexes)
    for i in range(len(association_array)):
        if not association_array[i] in aerial_bss_indexes:
            association_array[i] = ground_association_array[i]
    return bss, association_array, aerial_bss_indexes, aerial_user_indexes


def get_number_of_assigned_users(association_array, bss_numbers):
    number_list = [0] * bss_numbers
    for i in association_array:
        number_list[i] += 1
    return number_list
