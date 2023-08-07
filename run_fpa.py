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


