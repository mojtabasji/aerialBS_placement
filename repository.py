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
    obf_init_z = 0
    total_association_changes = 0
    obf_time_list = []
    obf_time_list_z = []
    obf_list = []
    obf_z_list = []
    obf_y_list = []
    alg_list = []
    piru_list = []
    detail_list = []
    positions_list = []
    users = None

    def fix_obf_z(self):
        for i in range(len(self.obf_z_list)):
            if self.obf_z_list[i] is None:
                self.obf_z_list[i] = self.obf_z_list[i - 1]
            if self.obf_y_list[i] is None:
                self.obf_y_list[i] = self.obf_y_list[i - 1]


