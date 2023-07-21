class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Repository(metaclass=SingletonMeta):
    global_counter = 0
    obf_init = 0
    total_association_changes = 0
    obf_time_list = []
    obf_list = []
    alg_list = []
    piru_list = []
    detail_list = []
    positions_list = []
    users = None
