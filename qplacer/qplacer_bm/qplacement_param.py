
class QplacementParam:
    def __init__(self, 
                 substrate_area, 
                 topology, 
                 scale_factor = 1000,
                 q_freq_range = (4.8, 5.2),
                 q_th = 0.1,
                 res_freq_range = (6, 7),
                 res_th = 0.1,
                 partition_size = 0.2,
                 qubit_pad_size = 0.4,
                 qubit_padding_size = 0.3,
                 padding_size = 0.1,
                 net_weight = 1, 
                 param_json_dir = 'test',
                 benchmark_dir = 'Qplace_benchmark',
                 placer_dir = 'placer',
                 freq_assign = True, 
                 partition = True,
                 debugging = True,
                 seed = None,
                 frequency_density_weight=9e-4,
                 density_weight=8e-4,
                 random_center_init_flag=1,
                 ):
        
        self.seed = seed
        self.debugging = debugging
        self.graph_debugging = False
        # scale graph layout for qubit placements in micrometers
        self.scale_factor = scale_factor  
        self.freq_assign = freq_assign
        self.partition = partition
        
        self.substrate_area = substrate_area
        self.topology = topology
        self.net_weight = net_weight

        self.frequency_density_weight = frequency_density_weight
        self.random_center_init_flag = random_center_init_flag
        self.density_weight = density_weight

        # Frequency in GHz
        self.q_freq_range = q_freq_range
        self.res_freq_range = res_freq_range
        self.q_th = q_th
        self.res_th = res_th

        # size in micro
        self.padding_size = padding_size
        self.partition_size = partition_size
        self.qubit_pad_size = qubit_pad_size
        self.qubit_padding_size = qubit_padding_size
        self.qubit_size = qubit_pad_size + 2*qubit_padding_size
        
        # file directory
        self.param_json_dir = param_json_dir
        self.benchmark_dir = benchmark_dir
        self.placer_dir = placer_dir
        self.file_paths = dict()
        self.file_name = "test"
        self.debugging_dir = ''

        self.pin_to_loc = {'nw': (-1, +1), 'sw': (-1, -1), 'se': (+1, -1), 'ne': (+1, +1)}


    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        print("\nBenchmark Params attributes : ")
        _info = []
        for attr, value in self.__dict__.items():
            _info.append(f" {attr:<20}: {value}")
        return "\n".join(_info)