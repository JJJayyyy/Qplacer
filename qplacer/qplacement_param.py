
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
                 qubit_size = 0.6,
                 padding_size = 0.1,
                 net_weight = 1, 
                 param_json_dir = 'test',
                 benchmark_dir = 'Qplace_benchmark',
                 placer_dir = 'placer',
                 freq_assign = True, 
                 partition = True,
                 debugging = True,
                 ):
        
        self.debugging = debugging
        # scale graph layout for qubit placements in micrometers
        self.scale_factor = scale_factor  
        self.freq_assign = freq_assign
        self.partition = partition

        # Frequency in GHz
        self.q_freq_range = q_freq_range
        self.q_th = q_th
        self.res_freq_range = res_freq_range
        self.res_th = res_th

        # size in micro
        self.partition_size = partition_size
        self.qubit_size = qubit_size
        self.padding_size = padding_size
        self.substrate_area = substrate_area
        self.topology = topology

        self.net_weight = net_weight

        # file directory
        self.param_json_dir = param_json_dir
        self.benchmark_dir = benchmark_dir
        self.placer_dir = placer_dir
        self.file_paths = dict()
        self.file_name = "test"


    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)