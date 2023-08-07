import time
import copy
import numpy as np
from network import (
    calc_gama,
    get_sorted_gama_index,
    change_user_link,
    calc_R,
    objective_function,
    objective_function_z,
    objective_function_y,
    get_bss_priority,
)
from constants import *
from log import logger
from utils import get_distance, cluster
from repository import Repository


repo = Repository()


def inc_counter(obf, alg, piru, detail_tuple, obf_z=None, obf_y=None):
    repo.global_counter += 1
    repo.obf_list.append(obf)
    repo.obf_z_list.append(obf_z)
    repo.obf_y_list.append(obf_y)
    repo.alg_list.append(alg)
    repo.piru_list.append(piru)

    if detail_tuple[0] is None:
        detail = "-"
    else:
        if alg == "PROB":
            detail = "bss = %d , pre_pi = %.3f , new_pi = %.3f" % (
                detail_tuple[0],
                detail_tuple[1],
                detail_tuple[2],
            )
        if alg == "UAC":
            detail = "user = %d , pre_bss = %d , new_bss = %d" % (
                detail_tuple[0],
                detail_tuple[1] + 1,
                detail_tuple[2] + 1,
            )
        if alg == "UPAS":
            detail = "bss = %d , pre_point = %s , new_point = %s" % (
                detail_tuple[0],
                detail_tuple[1],
                detail_tuple[2],
            )
    repo.detail_list.append(detail)


# update_bss_power   ----> PROB
def PROB(bss, bss_weights, association_array, R, piru, global_obf: float):
    logger.info("--------------> Updating bss powers with piru = %0.2f" % piru)
    start = time.time()
    min_pt = None   # PIRU * PT
    number_bs = len(bss)
    window_power = min(number_bs, int(PERCENT_POWER * number_bs))
    for iter in range(MAX_ITERATION_POWER):
        print("iteration PROB:", iter)
        bss_priority = get_bss_priority(R, bss, association_array)
        bss_weights_temp = copy.copy(bss_weights)
        obf_new = global_obf
        obf_z = None
        obf_y = None
        index_max = None
        count = 0
        for index in bss_priority:
            if bss_weights_temp[index] <= piru:
                continue
            else:
                bss_weights_temp[index] -= piru
                R_temp = calc_R(bss, association_array, bss_weights_temp, min_pt)
                bss_weights_temp[index] += piru
                obf_temp = objective_function(R_temp)
                obf_z_temp = objective_function_z(R_temp)
                obf_y_temp = objective_function_y(R_temp)
                if obf_temp > obf_new:
                    index_max = index
                    obf_new = obf_temp
                    obf_z = obf_z_temp
                    obf_y = obf_y_temp
                count += 1
                if count > window_power:
                    break
        if index_max is None:
            inc_counter(obf_new, "PROB", piru, (None, None, None), obf_z=obf_z, obf_y=obf_y)
            end = time.time()
            t = (end - start) / 60
            repo.obf_time_list.append((global_obf, t))
            logger.info(
                "--------------> Finished updating bss powers in time %.3f" % (t)
            )
            return bss_weights, global_obf
        else:
            old_weight = copy.copy(bss_weights[index_max])
            bss_weights[index_max] -= piru
            inc_counter(
                obf_new,
                "PROB",
                piru,
                (index_max, old_weight, bss_weights[index_max]),
                obf_z=obf_z,
                obf_y=obf_y,
            )
            print(
                "better result ---> changing power of bss %d with weight %.2f ---> obf_old= %.4f obf_new= %.4f"
                % (index_max, bss_weights[index_max], global_obf, obf_new)
            )
            global_obf = obf_new
    end = time.time()
    t = (end - start) / 60
    print("result", bss_weights, global_obf, t)
    repo.obf_time_list.append((global_obf, t))
    logger.info("--------------> Finished updating bss powers in time %.3f" % (t))
    return bss_weights, global_obf


# UAC ---> update_user_association
def UAC(bss, bss_weights, association_array, min_pt, global_obf):
    logger.info("--------------> Updating user association")
    start = time.time()
    window_user = min(len(repo.users), int(PERCENT_USER * len(repo.users)))
    n_association_change = 0
    obf_new = global_obf
    obf_z = None
    obf_y = None
    for iter in range(MAX_ITERATION_USER):
        gama = calc_gama(bss, association_array, bss_weights, min_pt)
        users_priority = get_sorted_gama_index(gama)
        print("iteration:", iter)
        association_array_max = None
        n_better = 0
        for i in range(len(repo.users)):
            association_array_temp = change_user_link(
                users_priority[i], bss, association_array, bss_weights, min_pt
            )
            if association_array_temp is not None:
                gama_temp = calc_gama(bss, association_array_temp, bss_weights, min_pt)
                R_temp = calc_R(bss, association_array_temp, bss_weights, min_pt)
                obf_temp = objective_function(R_temp)
                obf_z_temp = objective_function_z(R_temp)
                obf_y_temp = objective_function_y(R_temp)
                if obf_temp > obf_new:
                    obf_new = obf_temp
                    obf_z = obf_z_temp
                    obf_y = obf_y_temp
                    association_array_max = copy.copy(association_array_temp)
                    n_better += 1
                    if n_better >= window_user:
                        break
        if association_array_max is None:
            inc_counter(obf_new, "UAC", -1, (None, None, None), obf_z=obf_z, obf_y=obf_y)
            print("final user ", i)
            end = time.time()
            t = (end - start) / 60
            repo.obf_time_list.append((global_obf, t))
            logger.info(
                "--------------> Finished updating user association in time %.3f  ,number of association changes %d"
                % (t, n_association_change)
            )
            return association_array, global_obf
        else:
            n_association_change += 1
            association_array = copy.copy(association_array_max)
            inc_counter(
                obf_new,
                "UAC",
                -1,
                (
                    users_priority[i],
                    association_array[users_priority[i]],
                    association_array_temp[users_priority[i]],
                ),
                obf_z=obf_z,
                obf_y=obf_y,
            )
            print(
                "better result ---> user %d pre_bss= %d  gama1= %.2f   next_bss= %d  gama2= %.2f --->  obf_old= %.4f  obf_new= %.4f"
                % (
                    users_priority[i],
                    association_array[users_priority[i]],
                    gama[users_priority[i]],
                    association_array_temp[users_priority[i]],
                    gama_temp[users_priority[i]],
                    global_obf,
                    obf_new,
                )
            )
            global_obf = obf_new
    end = time.time()
    t = (end - start) / 60
    repo.total_association_changes += n_association_change
    repo.obf_time_list.append((global_obf, t))
    logger.info(
        "--------------> Finished updating user association in time %.3f  ,number of association changes %d"
        % (t, n_association_change)
    )
    return association_array, global_obf


# update_bss_position  ---> UPAS
def UPAS(bss, association_array, bss_weights, aerial_bss_indexes, min_pt, global_obf):
    logger.info("--------------> updating bss position")
    start = time.time()
    # print('old bss positions:',bss)
    users_dict = get_user_groups(repo.users, association_array)
    new_bss = bss.copy()
    changed_bss = []
    obf_new = global_obf
    obf_z = None
    obf_y = None
    if len(users_dict) == len(bss):
        for iter in range(len(bss)):
            index_max = -1
            for j, bs in enumerate(bss):
                if j not in aerial_bss_indexes:
                    continue
                temp_bss = new_bss.copy()
                array_bs = np.array([bs])
                array_users = np.array(users_dict[str(j)])
                # print('j',j,'length array users', len(array_users))
                _, centroids = cluster(array_users, len(array_bs), array_bs)
                temp_bs = centroids[0]
                temp_bss[j] = temp_bs
                if get_distance(temp_bs.astype(int), bs.astype(int)) > 0:
                    R_new = calc_R(temp_bss, association_array, bss_weights, min_pt)
                    obf_temp = objective_function(R_new)
                    obf_z_temp = objective_function_z(R_new)
                    obf_y_temp = objective_function_y(R_new)
                    if obf_temp > obf_new:
                        index_max = j
                        new_bs = temp_bs.copy()
                        obf_new = obf_temp
                        obf_z = obf_z_temp
                        obf_y = obf_y_temp
            if index_max != -1:
                old_position = new_bss[index_max].copy()
                new_bss[index_max] = new_bs.copy()
                new_position = new_bss[index_max].copy()
                inc_counter(
                    obf_new,
                    "UPAS",
                    -1,
                    (
                        index_max,
                        [int(old_position[0]), int(old_position[1])],
                        [int(new_position[0]), int(new_position[1])],
                    ),
                    obf_z=obf_z,
                    obf_y=obf_y
                )
                changed_bss.append(index_max)
            else:
                inc_counter(obf_new, "UPAS", -1, (None, None, None), obf_z=obf_z, obf_y=obf_y)
                break
        logger.info("obf_old = %0.4f , obf_new = %0.4f" % (global_obf, obf_new))
        global_obf = obf_new
        changed_positions = [bss[index].tolist() for index in changed_bss]
        logger.info(
            "moved bss %s , new positions %s" % (changed_bss, changed_positions)
        )
        positions_dict = {"indexes": changed_bss, "positions": changed_positions}
        repo.positions_list.append(positions_dict)
    else:
        print("number of bss and users clusters are not equal")
    end = time.time()
    t = (end - start) / 60
    repo.obf_time_list.append((obf_new, t))
    logger.info("--------------> finished updating bss position in time %.3f" % (t))
    return new_bss, global_obf


# specifing users of each bss ---> key: bs index   value: users of that bs
def get_user_groups(users, association_array):
    list_bss = []
    users_dict = {}
    for i, user in enumerate(users):
        if association_array[i] in list_bss:
            users_dict[str(association_array[i])].append(user)
        else:
            list_bss.append(association_array[i])
            users_dict[str(association_array[i])] = [user]
    return users_dict
