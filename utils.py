import os
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from constants import *
from repository import Repository

repo = Repository()


def get_distance(point1, point2):
    # euclidean distance
    return np.linalg.norm(point1 - point2)


def get_pic_path(dir):
    now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    # return dir + "/" + str(now)[:-4] + ".png"
    return dir + "/" + str(now).replace(":", "-") + ".png"


def create_dir(dir):
    if not os.path.exists(dir):
        Path(dir).mkdir(parents=True)


def write_json(json_path, data):
    with open(json_path, "w") as outfile:
        json.dump(data, outfile)


def save_mop_results(file_path, mop_details):
    save_dict = {
        "global_counter": repo.global_counter,
        "obf_init": repo.obf_init,
        "obf": mop_details["obf"],
        "total_association_changes": repo.total_association_changes,
        "obf_time_list": repo.obf_time_list,
        "obf_list": repo.obf_list,
        "alg_list": repo.alg_list,
        "piru_list": repo.piru_list,
        "detail_list": repo.detail_list,
        "bss_weights": mop_details["bss_weights"],
        "bss_positions": mop_details["bss_positions"],
    }
    write_json(file_path, save_dict)


def save_best_fpa_results(file_path, fpa_details):
    save_dict = []
    if os.path.exists(file_path):
        with open(file_path, "r") as input_file:
            save_dict = json.load(input_file)
    save_dict.append(
        {
            "p": fpa_details["p"],
            "aerial_number": fpa_details["aerial_number"],
            "k": fpa_details["k"],
            "centroid": fpa_details["centroid"],
            "global_counter": fpa_details["global_counter"],
            "obf_init": fpa_details["obf_init"],
            "obf": fpa_details["obf"],
            "obf_list": fpa_details["obf_list"],
            "alg_list": fpa_details["alg_list"],
            "piru_list": fpa_details["piru_list"],
        }
    )
    with open(file_path, "w") as outfile:
        json.dump(save_dict, outfile)


def cluster(data, n_clusters, initial_centroids):
    if KMEANS_PP_MODE:
        kmeans = KMeans(
            n_clusters=n_clusters, random_state=0, init="k-means++", n_init=30
        ).fit(data)
    else:
        kmeans = KMeans(
            n_clusters=n_clusters, random_state=0, init=initial_centroids, n_init=1
        ).fit(data)
    association_array = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return association_array, centroids


def cluster_aerial_users(aerial_user_indexes, k):
    try:
        aerial_users = []
        for i in range(len(repo.users)):
            if i in aerial_user_indexes:
                aerial_users.append(repo.users[i])
        aerial_users = np.array(aerial_users)
        association_array_temp, centroids = cluster(aerial_users, k, [])
        return association_array_temp, centroids
    except:
        return None, None


def reset_repository():
    repo.global_counter = 0
    repo.obf_init = 0
    repo.obf_time_list = []
    repo.obf_list = []
    repo.obf_z_list = []
    repo.obf_y_list = []
    repo.alg_list = []
    repo.piru_list = []
    repo.detail_list = []
    repo.positions_list = []


def delete(np_2d_array, index):
    return np.delete(np_2d_array, index, 0)


def algorithms_contribution(alg_list, obf_list, init_obf):
    k1 = "PROB"
    k2 = "UAC"
    k3 = "UPAS"
    it_prob = alg_list.count(k1)
    it_uac = alg_list.count(k2)
    it_upas = alg_list.count(k3)
    all_it = len(alg_list)
    print(
        "iterations ---> prob:%d (%.2f%%) uac %d (%.2f%%) upas %d (%.2f%%)"
        % (
            it_prob,
            it_prob * 100 / all_it,
            it_uac,
            it_uac * 100 / all_it,
            it_upas,
            it_upas * 100 / all_it,
        )
    )

    imp_dict = {k1: 0, k2: 0, k3: 0}

    imp_dict[k1] = obf_list[0] - init_obf
    for i in range(1, len(obf_list)):
        imp_dict[alg_list[i]] = imp_dict[alg_list[i]] + (obf_list[i] - obf_list[i - 1])

    all_imp = imp_dict[k1] + imp_dict[k2] + imp_dict[k3]
    print(
        "improvements ---> prob:%.2f (%.2f%%) uac %.2f (%.2f%%) upas %.2f (%.2f%%)"
        % (
            imp_dict[k1],
            imp_dict[k1] * 100 / all_imp,
            imp_dict[k2],
            imp_dict[k2] * 100 / all_imp,
            imp_dict[k3],
            imp_dict[k3] * 100 / all_imp,
        )
    )


def decibel_to_watt(db):
    return 10 ** (db / 10)


def watt_to_decibel(watt):
    return 10 * np.log10(watt)
