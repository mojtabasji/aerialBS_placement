import os
import sys, time, numpy as np
import json
from constants import *
from network import calc_R, objective_function
from repository import Repository
from updates import PROB, UAC, UPAS
from dataset import (
    setup_aerial_bss_manual,
    setup_aerial_bss_auto,
    get_number_of_assigned_users,
    get_aerial_user_indexes,
    set_dataset_users,
    set_dataset_bss,
    user_association,
    specify_aerial_users,
)
from visualize import plot_network, plot_convergence, plot_results, plot_upas_changes
from utils import (
    get_pic_path,
    cluster_aerial_users,
    reset_repository,
    create_dir,
    write_json,
    delete,
    save_mop_results,
    save_best_fpa_results,
)
from log import logger
from pso_algorithm import PSO

repo = Repository()


def get_methods(method):
    method_prob = "ours"
    method_upas = "ours"
    if METHODS[method] == "pso_prob":
        method_prob = "pso"
    if METHODS[method] == "pso_upas":
        method_upas = "pso"
    if METHODS[method] == "both":
        method_prob = "pso"
        method_upas = "pso"
    return method_prob, method_upas


def MOP(
    bss,
    association_array,
    bss_weights,
    aerial_bss_indexes,
    method,
    dynamic_bss=False,
    degradation=True,
):
    method_prob, method_upas = get_methods(method)
    start = time.time()
    min_pt = PIRU * PT
    piru = PIRU
    R = calc_R(bss, association_array, bss_weights, min_pt)
    repo.obf_init = objective_function(R)
    obf = repo.obf_init
    repo.obf_time_list.append((obf, 0))
    logger.info("objective= %.3f" % obf)
    while True:
        min_pt = piru * PT
        if piru <= 0.01:
            break
        if method_prob == "ours":
            bss_weights, obf_new = PROB(
                bss.copy(),
                bss_weights.copy(),
                association_array.copy(),
                R.copy(),
                piru,
                obf,
            )
        elif method_prob == "pso":
            fit_func_data = {
                "bss": bss.copy(),
                "association_array": association_array.copy(),
                "min_pt": min_pt,
            }
            pso_prob = PSO(
                fit_func_data=fit_func_data,
                init_pos=bss_weights.tolist(),
                n_particles=50,
                c0=1.0,
                c1=1.5,
                w=0.1,
                limit=(0, 1),
                alg="PROB",
            )
            res_prob = pso_prob.optimize()
            bss_weights, obf_new = np.array(res_prob[0]), res_prob[1]
        if obf_new == obf:
            print("Cut the PIRU in half")
            piru = piru / 2
            continue
        else:
            obf = obf_new
        association_array, obf_new = UAC(
            bss.copy(), bss_weights.copy(), association_array.copy(), min_pt, obf
        )
        if obf_new == obf:
            print("Cut the PIRU in half")
            piru = piru / 2
            continue
        else:
            obf = obf_new
            if dynamic_bss:
                if method_upas == "ours":
                    bss, obf_new = UPAS(
                        bss.copy(),
                        association_array.copy(),
                        bss_weights,
                        aerial_bss_indexes,
                        min_pt,
                        obf,
                    )
                elif method_upas == "pso":
                    aerial_bss = np.array([bss[index] for index in aerial_bss_indexes])
                    ground_bss = [
                        bss[index]
                        for index in range(len(bss))
                        if index not in aerial_bss_indexes
                    ]
                    fit_func_data = {
                        "ground_bss": ground_bss.copy(),
                        "bss_weights": bss_weights,
                        "association_array": association_array.copy(),
                        "min_pt": min_pt,
                    }
                    pso_upas = PSO(
                        fit_func_data=fit_func_data,
                        init_pos=aerial_bss.flatten().tolist(),
                        n_particles=100,
                        c0=1.0,
                        c1=1.5,
                        w=0.1,
                        limit=(0, 4000),
                        alg="UPAS",
                    )
                    res_upas = pso_upas.optimize()
                    new_aerial_bss, obf_new = res_upas[0], res_upas[1]
                    bss = np.concatenate(
                        (ground_bss, np.array(new_aerial_bss).reshape(-1, 2)), axis=0
                    )
                if obf_new > obf:
                    obf = obf_new
    end = time.time()
    logger.info(
        "***************** finished , final obf = %.2f  association changes = %d   run time = %.1f minutes*****************"
        % (obf, repo.total_association_changes, (end - start) / 60)
    )
    return (
        obf,
        bss,
        bss_weights,
        association_array,
        repo.total_association_changes,
        repo.obf_time_list,
    )


def mop_wrapper(
    mixed_bss,
    association_array,
    aerial_bss_indexes,
    plot_title,
    plot_dir,
    method=0,
    ground_bss_weights=None,
    gnd_is_none=True,
):
    aerial_bss_number = len(aerial_bss_indexes)
    ground_bss_number = len(mixed_bss) - aerial_bss_number
    if gnd_is_none:
        bss_weights = np.ones(aerial_bss_number)
    else:
        aerial_bss_weights = np.array([1.0 for i in range(aerial_bss_number)])
        bss_weights = np.concatenate((ground_bss_weights, aerial_bss_weights), axis=0)
    user_number_list1 = get_number_of_assigned_users(association_array, len(mixed_bss))
    title = "%s (ng = %d , na = %d)" % (
        plot_title,
        ground_bss_number,
        aerial_bss_number,
    )
    aerial_user_indexes = get_aerial_user_indexes(association_array, aerial_bss_indexes)
    plot_network(
        mixed_bss,
        association_array,
        title,
        aerial_bss_indexes=aerial_bss_indexes,
        aerial_user_indexes=aerial_user_indexes,
        bss_weights=bss_weights,
        user_numbers=user_number_list1,
        result_path=get_pic_path(plot_dir),
    )
    final_obf, final_bss, final_bss_weights, final_association_array, _, _ = MOP(
        mixed_bss.copy(),
        association_array.copy(),
        bss_weights.copy(),
        aerial_bss_indexes,
        method=method,
        dynamic_bss=True,
        degradation=True,
    )
    final_aerial_user_indexes = get_aerial_user_indexes(
        final_association_array, aerial_bss_indexes
    )
    user_number_list = get_number_of_assigned_users(
        final_association_array, len(final_bss)
    )
    title = "%s (ng = %d , na = %d , obf = %.2f)" % (
        plot_title,
        ground_bss_number,
        aerial_bss_number,
        final_obf,
    )
    plot_network(
        final_bss,
        final_association_array,
        title,
        aerial_bss_indexes=aerial_bss_indexes,
        aerial_user_indexes=final_aerial_user_indexes,
        bss_weights=final_bss_weights,
        user_numbers=user_number_list,
        result_path=(get_pic_path(plot_dir)),
    )
    if aerial_bss_number > 0:
        plot_upas_changes(
            mixed_bss,
            final_bss,
            repo.positions_list,
            "",
            aerial_bss_indexes=aerial_bss_indexes,
            result_path=(get_pic_path(plot_dir)),
        )
    plot_convergence(
        list(range(1, repo.global_counter + 1)),
        (repo.obf_list),
        (repo.alg_list),
        result_path=(get_pic_path(plot_dir)),
    )
    return final_obf, final_bss, final_bss_weights


def run_mop(dataset_index_bss, aerial_bss_number, method):
    aerial_bss_indexes = None
    dataset_index_users = int(dataset_index_bss / 3)  # 0 to 2
    ground_bss = set_dataset_bss(dataset_index_bss)
    repo.users = set_dataset_users(dataset_index_users)
    states_dir = PROCESS_STATES_BASE_PATH + DATASET_TITLES[dataset_index_bss]
    result_dir = f"{RESULTS_BASE_PATH}/{METHODS[method]}"
    result_path = f"{result_dir}/{DATASET_TITLES[dataset_index_bss]}_MOP.json"
    result_path2 = f"{result_dir}/{DATASET_TITLES[dataset_index_bss]}_aerials.json"
    create_dir(LOG_DIR)
    create_dir(states_dir)
    create_dir(result_dir)
    create_dir(RESULTS_BASE_PATH)
    ground_association_array = user_association(ground_bss)
    ground_bss_weights = np.array([1.0 for i in range(len(ground_bss))])
    if aerial_bss_number == 0:
        mixed_bss = ground_bss
        association_array = user_association(ground_bss)
        aerial_bss_indexes = []
    else:
        if os.path.exists(result_path2):
            with open(result_path2, "r") as file:
                results = json.load(file)
            if len(results) >= aerial_bss_number:
                aerial_bss_found = results[:aerial_bss_number]
                (
                    mixed_bss,
                    association_array,
                    aerial_bss_indexes,
                    _,
                ) = setup_aerial_bss_manual(
                    ground_bss, np.array(aerial_bss_found), ground_association_array
                )
    if aerial_bss_indexes is None:
        logger.info("error ----->  aerial bss not found!!!")
        return
    title = DATASET_TITLES[dataset_index_bss]
    final_obf, final_bss, final_bss_weights = mop_wrapper(
        mixed_bss,
        association_array,
        aerial_bss_indexes,
        title,
        states_dir,
        method,
        ground_bss_weights,
        False,
    )
    mop_details = {
        "obf": final_obf,
        "bss_positions": final_bss.tolist(),
        "bss_weights": final_bss_weights.tolist(),
    }
    save_mop_results(result_path, mop_details)


def locate_ground_bss(topo_index, min_n, max_n):
    repo.users = set_dataset_users(topo_index)
    obf_number_dict = {}
    bss_number_dict = {}
    for bss_number in range(min_n, max_n + 1):
        (
            bss,
            association_array,
            aerial_bss_indexes,
            aerial_user_indexes,
        ) = setup_aerial_bss_auto(bss_number)
        plot_title = TOPO_TITLES[topo_index]
        plot_dir = GROUND_PLACEMENT_PATH + TOPO_TITLES[topo_index]
        create_dir(plot_dir)
        final_obf, final_bss, _ = mop_wrapper(
            bss, association_array, aerial_bss_indexes, plot_title, plot_dir
        )
        obf_number_dict[str(bss_number)] = final_obf
        bss_number_dict[str(bss_number)] = final_bss.tolist()
        reset_repository()

    plot_results(
        obf_number_dict.keys(),
        obf_number_dict.values(),
        "bss number",
        "obf",
        get_pic_path(plot_dir),
    )
    json_path = f"{plot_dir}/{GROUND_BSS_FILE}"
    json_dict = {}
    for number in obf_number_dict.keys():
        json_dict[number] = {
            "obf": obf_number_dict[number],
            "bss_locations": bss_number_dict[number],
        }

    write_json(json_path, json_dict)
    return json_dict


def remove_best_bss(topo_index, ground_bss):
    bss_number = len(ground_bss)
    min_obf = sys.maxsize
    min_index = -1
    bss_temp = None
    temp_obf_list = []
    repo.users = set_dataset_users(topo_index)
    plot_dir = GROUND_PLACEMENT_PATH + TOPO_TITLES[topo_index]
    create_dir(plot_dir)
    for i in range(bss_number):
        bss_temp = delete(ground_bss, i)
        ground_association_array = user_association(bss_temp)
        bss_weights = np.ones(bss_number)
        user_numbers = get_number_of_assigned_users(
            ground_association_array, bss_number
        )
        plot_network(
            bss_temp,
            ground_association_array,
            (TOPO_TITLES[topo_index]),
            bss_weights=bss_weights,
            user_numbers=user_numbers,
            result_path=(get_pic_path(plot_dir)),
        )
        temp_obf, _, temp_bss_weights, temp_association_array, _, _ = MOP(
            bss_temp,
            ground_association_array.copy(),
            np.ones(len(bss_temp)),
            aerial_bss_indexes=[],
            method=0,
            dynamic_bss=False,
            degradation=True,
        )
        temp_user_numbers = get_number_of_assigned_users(
            temp_association_array, bss_number
        )
        plot_network(
            bss_temp,
            temp_association_array,
            (TOPO_TITLES[topo_index]),
            bss_weights=temp_bss_weights,
            user_numbers=temp_user_numbers,
            result_path=(get_pic_path(plot_dir)),
        )
        temp_obf_list.append(temp_obf)
        if temp_obf < min_obf:
            min_obf = temp_obf
            min_index = i
        reset_repository()

    reduced_bss = []
    for i in range(len(ground_bss)):
        if i != min_index:
            reduced_bss.append(ground_bss[i].tolist())

    json_path = f"{plot_dir}/{REDUCED_BSS_FILE}"
    json_dict = {
        "obf": min_obf,
        "bss_locations": reduced_bss,
        "removed": ground_bss[min_index].tolist(),
    }
    write_json(json_path, json_dict)


def FPA(dataset_index, max_aerial, p, max_k, method):
    result_dir = f"{RESULTS_BASE_PATH}/{METHODS[method]}"
    fpa_path = f"{result_dir}/{DATASET_TITLES[dataset_index]}_FPA.json"
    aerials_path = f"{result_dir}/{DATASET_TITLES[dataset_index]}_aerials.json"
    plot_dir = PROCESS_STATES_BASE_PATH + DATASET_TITLES[dataset_index]
    create_dir(result_dir)
    create_dir(plot_dir)
    ground_bss = set_dataset_bss(dataset_index)
    dataset_index_users = int(dataset_index / 3)  # 0 to 2
    repo.users = set_dataset_users(dataset_index_users)
    ground_bss_weights = np.array([1.0 for i in range(len(ground_bss))])
    ground_association_array = user_association(ground_bss)
    aerial_bss_found = None
    for aerial_bss_number in range(0, max_aerial):
        logger.info("1.finding aerial :%d" % (aerial_bss_number + 1))
        max_obf = -1
        fpa_details = None
        if aerial_bss_number == 0:
            mixed_bss = ground_bss
            association_array = user_association(ground_bss)
            aerial_bss_indexes = []
            bss_weights = ground_bss_weights
        else:
            (
                mixed_bss,
                association_array,
                aerial_bss_indexes,
                _,
            ) = setup_aerial_bss_manual(
                ground_bss, np.array(aerial_bss_found), ground_association_array
            )
            aerial_bss_weights = np.array([1.0 for i in range(aerial_bss_number)])
            bss_weights = np.concatenate(
                (ground_bss_weights, aerial_bss_weights), axis=0
            )
        # ------------------------------
        aerial_user_number = int(len(repo.users) * p)
        min_pt = PIRU * PT
        aerial_user_indexes, _ = specify_aerial_users(
            mixed_bss,
            association_array,
            bss_weights,
            min_pt,
            aerial_user_number,
        )
        # ------------------------------
        for k in range(2, max_k + 1):
            logger.info("2.starting for k %d" % k)
            _, centroids = cluster_aerial_users(aerial_user_indexes, k)
            if centroids is None:
                logger.info("number of users is Insufficient , going to next percent")
                break
            for i, centroid in enumerate(centroids):
                logger.info(
                    "3.centroid (%d/%d): %s " % (i + 1, len(centroids), centroid)
                )
                aerial_bss_temp = np.array([centroid.tolist()])
                if aerial_bss_number > 0:
                    aerial_bss = []
                    for i in aerial_bss_indexes:
                        aerial_bss.append(mixed_bss[i].tolist())
                    aerial_bss_temp = np.concatenate(
                        (np.array(aerial_bss), aerial_bss_temp), axis=0
                    )
                (
                    mixed_bss_temp,
                    association_array_temp,
                    aerial_bss_indexes_temp,
                    _,
                ) = setup_aerial_bss_manual(
                    ground_bss, aerial_bss_temp, ground_association_array
                )
                final_obf, final_bss, _ = mop_wrapper(
                    mixed_bss_temp,
                    association_array_temp,
                    aerial_bss_indexes_temp,
                    DATASET_TITLES[dataset_index],
                    plot_dir,
                    method,
                    ground_bss_weights,
                    False,
                )
                if final_obf > max_obf:
                    max_obf = final_obf
                    fpa_details = {
                        "p": p,
                        "aerial_number": aerial_bss_number,
                        "k": k,
                        "centroid": centroid.tolist(),
                        "global_counter": repo.global_counter,
                        "obf_init": repo.obf_init,
                        "obf": final_obf,
                        "obf_list": repo.obf_list,
                        "alg_list": repo.alg_list,
                        "piru_list": repo.piru_list,
                    }
                    aerial_bss_found = []
                    for index in aerial_bss_indexes_temp:
                        aerial_bss_found.append(final_bss[index].tolist())
                # logger.info(
                #     "p = %.2f , k = %d , centroid = %s ,final_pos = %s , user_number = %d , obf = %.2f"
                #     % (
                #         p,
                #         k,
                #         c,
                #         final_bss[len(final_bss) - 1],
                #         unl[len(unl) - 1],
                #         final_obf,
                #     )
                # )
                reset_repository()
        if max_obf == -1:
            break
        else:
            with open(aerials_path, "w") as file:
                json.dump(aerial_bss_found, file)
            save_best_fpa_results(fpa_path, fpa_details)
