from algorithms import FPA
from algorithms import run_mop

## just aerial A  A-2BSS  A-1BSS  B  B-2BSS  B-1BSS  C  C-2BSS  C-1BSS
##    None     0    1       2     3    4        5    6     7      8

# ----------------------------------------------------------------
# dataset_index = 7
# max_aerial = 4
# max_k = 3
# percents = [0.20]
# # percents = [.05,.07,.09,.11,.13,.15]
# # percents = [0.05,.10,.15,.20,.25,.30,.35,.40]
# for p in percents:
#     print("starting FPA with percent", p)
#     FPA(dataset_index, max_aerial, p, max_k, method=0)


# ----------------------------------------------------------------


# run_mop(dataset_index_bss=8, aerial_bss_number=2, method=0)


#
from algorithms import locate_ground_bss
from dataset import set_dataset_users

topo_index = 2
result_dict = locate_ground_bss(topo_index, min_n=10, max_n=11)
