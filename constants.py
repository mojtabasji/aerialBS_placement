MAX_ITERATION_POWER = 1000
MAX_ITERATION_USER = 1000
MAX_AERIAL_NUMBERS = 0
MAX_K = 2

# Pij constants
PT = 2
## power index reduction unit
PIRU = 0.1
DT = 1
DR = 1
F = 1.6  # GHz
LAMBDA = 1 / F
PI = 3.14
SIGMA2 = 0
B = 1


PERCENT_POWER = 1
PERCENT_USER = 0.01
KMEANS_PP_MODE = True

TOPO_TITLES = ["A", "B", "C"]
DATASET_TITLES = ["A", "A2", "A1", "B", "B2", "B1", "C", "C2", "C1"]


LOG_DIR = "./log"
USERS_PATH_BASE = "./inputs/users"
BSS_PATH = "./inputs/base_stations.json"
PROCESS_STATES_BASE_PATH = "./outputs/process_states/"
RESULTS_BASE_PATH = "./outputs/saved_results/"
GROUND_PLACEMENT_PATH = "./outputs/ground_bss/"
GENERAL_PATH = "./outputs/general/"
GROUND_BSS_FILE = "ground_bss.json"
REDUCED_BSS_FILE = "reduced_bss.json"
METHODS = ["ours", "pso_prob", "pso_upas", "both"]
# visualize
FG_SIZE = (10, 10)
BS_SIZE = 100
XMIN, XMAX = -2000, 2000
YMIN, YMAX = -2000, 2000
WIDTH, HEIGHT = (XMAX - XMIN), (YMAX - YMIN)
COLORS = [
    [144, 91, 38],
    [238, 79, 42],
    [53, 50, 46],
    [83, 16, 238],
    [169, 7, 225],
    [236, 199, 109],
    [73, 140, 109],
    [222, 254, 21],
    [221, 28, 73],
    [155, 244, 95],
    [102, 178, 243],
    [49, 106, 39],
    [97, 21, 159],
    [109, 247, 188],
    [211, 131, 92],
    [255, 150, 246],
    [221, 114, 118],
    [71, 187, 5],
    [1, 203, 200],
    [202, 231, 176],
    [120, 0, 24],
    [32, 105, 3],
    [255, 119, 0],
]
COLORS_TOPO = [
    [144, 91, 38],
    [238, 79, 42],
    [53, 50, 46],
]
MARKERS = [
    ".",
    ",",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "s",
    "p",
    "P",
    "*",
    "h",
    "H",
    "+",
    "x",
    "X",
    "D",
    "d",
    "|",
    "_",
]
NUMBER_MARKER = [
    "$1$",
    "$2$",
    "$3$",
    "$4$",
    "$5$",
    "$6$",
    "$7$",
    "$8$",
    "$9$",
    "$10$",
    "$11$",
    "$12$",
    "$13$",
    "$14$",
    "$15$",
    "$16$",
    "$17$",
    "$18$",
    "$19$",
    "$20$",
    "$21$",
    "$22$",
    "$23$",
    "$24$",
    "$25$",
]
