from dataset import generate_users
from constants import *
generated_users = generate_users(WIDTH, HEIGHT,200)


from dataset import make_empty_region
region_list= [(0, 0),(0, 1),(0, 2),(0, 3),(1, 0),(1, 1),(1, 2),(1, 3),(2, 0),(2, 1),(2, 2),(2, 3),(3, 0),(3, 1),(3, 2),(3, 3)]
users = make_empty_region(generated_users, WIDTH, HEIGHT, 4, 4, 5, region_list=None)

from visualize import plot_topology
from dataset import set_dataset_users
from constants import *
title = "D"
color = [97, 21, 159]
plot_topology(
        users,
        title ,
        color,
        [0, HEIGHT],
        [0, WIDTH],
    )


from algorithms import locate_ground_bss
from dataset import set_dataset_users
topo_index = 0
result_dict = locate_ground_bss(topo_index,min_n=10, max_n=17)


import json
import numpy as np
from algorithms import remove_best_bss
from constants import *

topo_index = 0
ground_number = "16"

json_path = f"{GROUND_PLACEMENT_PATH}{TOPO_TITLES[topo_index]}/{GROUND_BSS_FILE}"
print(json_path)
with open(json_path, "r") as file:
    data = json.load(file)
    ground_bss = np.array(data[ground_number]["bss_locations"])

remove_best_bss(topo_index, ground_bss)

from visualize import plot_topology
from dataset import set_dataset_users
from constants import *
topo_index = 1
title = TOPO_TITLES[topo_index]
color = COLORS_TOPO[topo_index]
users = set_dataset_users(topo_index)
result_path = GENERAL_PATH + title + ".png"
plot_topology(
        users,
        title ,
        color,
        [0, HEIGHT],
        [0, WIDTH],
        result_path,
    )


from algorithms import run_mop
run_mop(
        dataset_index_bss=1,
        aerial_bss_number=2,
        method=0
    )


import json
from constants import *
from utils import algorithms_contribution

dataset_index = 7
method = "ours"

file_path = f"{RESULTS_BASE_PATH}/{method}/{DATASET_TITLES[dataset_index]}_MOP.json"
with open(file_path, "r") as file:
        results = json.load(file)
alg_list = results.get("alg_list")
obf_list = results.get("obf_list")
obf_init = results.get("obf_init")
algorithms_contribution(alg_list, obf_list, obf_init)


import json
from visualize import plot_bss_weights
from constants import *

dataset_index = 7
method = "ours"

file_path = f"{RESULTS_BASE_PATH}/{method}/{DATASET_TITLES[dataset_index]}_MOP.json"
with open(file_path, "r") as file:
        results = json.load(file)
weights = results.get("bss_weights")
if weights is not None:
    result_path = GENERAL_PATH + "weights_" +DATASET_TITLES[dataset_index] + ".png"
    plot_bss_weights(weights,result_path)


from visualize import multi_plot
from constants import *
import json

dataset_index = 7
title = DATASET_TITLES[dataset_index]
legends = ["PROB(ours) + UAC(ours) + UPAS(ours)",
            "PROB(PSO) + UAC(ours) + UPAS(ours)",
            "PROB(ours) + UAC(ours) + UPAS(PSO)",
            "PROB(PSO) + UAC(ours) + UPAS(PSO)"]

color_list = ["b","k","r","g"]

obfs_list = []
for method in METHODS:
    mop_path = f"{RESULTS_BASE_PATH}/{method}/{title}_MOP.json"
    with open(mop_path, "r") as f:
        saved_results = json.load(f)
    obfs_list.append(saved_results["obf_list"])

# obfs_list = [[20,25,30,46,42],[23,25,30,46,42],[20,25,30,43,42],[20,25,38,46,42],[30,25,30,46,42],[20,25,30,46,42]]

save_path = f"./outputs/general/obf_methods_{title}.png"
multi_plot(None,"iteration", obfs_list, None, color_list,legends ,result_path=None)


from algorithms import FPA

dataset_index = 1
max_aerial = 7
max_k = 2
percents = [0.20]
# percents = [.05,.07,.09,.11,.13,.15]
# percents = [0.05,.10,.15,.20,.25,.30,.35,.40]
for p in percents:
    print("starting FPA with percent", p)
    FPA(dataset_index,max_aerial,p,max_k,method=0)


from visualize import multi_plot
from constants import *
import json

method = "ours"
result_dir = f"{RESULTS_BASE_PATH}/{method}"
dataset_indexes = [1, 2, 4, 5, 7, 8]
dataset_titles = [DATASET_TITLES[index] for index in dataset_indexes]
legends = [f"Dataset {title}" for title in dataset_titles]
marker_list = ["o", "P","D",7,"*",6]
color_list = ["r", "b","g","c","m","k"]
aerial_numbers = [1, 2, 3, 4, 5]
p = 0.20  # percent of users

obfs_list = []
for title in dataset_titles:
    obfs = [0] * len(aerial_numbers)
    fpa_path = f"{result_dir}/{title}_FPA.json"
    with open(fpa_path, "r") as f:
        saved_results = json.load(f)
    for result in saved_results:
        if result["p"] == p:
            obfs[result["aerial_number"]] = result["obf"]
    obfs_list.append(obfs)

# obfs_list = [[20,25,30,46,42],[23,25,30,46,42],[20,25,30,43,42],[20,25,38,46,42],[30,25,30,46,42],[20,25,30,46,42]]

save_path = f"./outputs/general/obf_aerials_{method}.png"
multi_plot(aerial_numbers,"number of aerial base stations", obfs_list, marker_list, color_list,legends ,result_path=save_path)

from visualize import plot_pi_values
from constants import *
import numpy as np

method = "ours"
result_dir = f"{RESULTS_BASE_PATH}/{method}"
dataset_indexes = [1, 2, 4, 5, 7, 8]
dataset_titles = [DATASET_TITLES[index] for index in dataset_indexes]
p_list =[ 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40 ]  # percent of users
colors = [ 'r', 'g', 'b', 'c', 'm', 'y', 'gray', 'tab:orange']
legends = [ 'π = 0.05', 'π = 0.10', 'π = 0.15', 'π = 0.20', 'π = 0.25', 'π = 0.30', 'π = 0.35', 'π = 0.40', ]

obfs_list = []
for title in dataset_titles:
    obfs = [0] * len(p_list)
    fpa_path = f"{result_dir}/{title}_FPA.json"
    with open(fpa_path, "r") as f:
        saved_results = json.load(f)
    for result in saved_results:
        if result["aerial_number"] == 1:
            index = p_list.index(result["p"])
            obfs[index] = result["obf"]
    obfs_list.append(obfs)

obfs_list = np.swapaxes(np.array(obfs_list),0,1)
save_path = ""

# obf_list = [
#     [37.15, 35.99, 42.6, 37.42, 35.99, 35.96],
#     [36.81, 35.55, 42.74, 39.56, 40.68, 35.6],
#     [36.81, 35.55, 42.67, 43.21, 40.74, 35.57],
#     [36.81, 35.55, 42.52, 43.31, 40.83, 35.57],
#     [36.81, 35.55, 42.52, 42.79, 40.74, 35.63],
#     [36.81, 35.58, 42.45, 43.05, 40.74, 35.6],
#     [36.81, 35.58, 42.49, 43.12, 40.74, 35.6],
#     [36.81, 35.53, 42.52, 42.79, 40.68, 35.55],
# ]

plot_pi_values(legends,dataset_titles,colors,obfs_list,(34,44))



