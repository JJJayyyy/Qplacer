from shapely.errors import ShapelyDeprecationWarning
import matplotlib.pyplot as plt
import networkx as nx
import warnings
import logging
import argparse
import json
import sys
import os

# root_dir = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/qplacer'
# if root_dir not in sys.path:
#     sys.path.append(f'{root_dir}')

from connectivity import ConnectivityGraphBuilder
from frequency_assignment import FrequencyAssigner
from design_format import DesignFormator
from collision_check import FreqCollisionChecker
from qplacement_param import QplacementParam
from qplacement_database import QplacementDatabase

from qplacer_io.lef_file_writer import LefFileWriter
from qplacer_io.def_file_writer import DefFileWriter
from qplacer_io.json_file_writer import JsonFileWriter


warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.root.name = 'QPlacer'
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)-7s] %(name)s - %(message)s',
                    stream=sys.stdout)


""" parameters """
area_dict = {
    "grid-25": (7800, 7800),
    "grid-64": (12900, 12900),
    "falcon": (6900,6900),
    "hummingbird": (10800, 10800),
    "eagle" : (15300, 15300),
    "Aspen-11" : (9000, 9000),
    "Aspen-M" : (12900, 12900),
    'xtree-53' : (9300, 9300)
}

qubit_num_dict = {
    'grid-4': ('grid', 4), 
    'grid-25': ('grid', 25),  
    'grid-64':('grid', 64),
    'falcon':('ibm_falcon', 27), 
    'hummingbird':('ibm_hummingbird', 65), 
    'eagle':('ibm_eagle', 127),
    'Aspen-11':('oxtagon', 40), 
    'Aspen-M':('oxtagon', 80), 
    'xtree-53': ('xtree', 53),
    'xtree-17': ('xtree', 17),
    'xtree-5': ('xtree', 5)
}


class BenchmarkGenerator:
    def __init__(self, 
                 bm_json,
                 suffix='wp_wf'
                 ):
        self.suffix = suffix
        with open(bm_json, 'r') as json_file:
            self.bm_setting = json.load(json_file)

        self.debugging_dir = 'logs'
        if not os.path.isdir(self.debugging_dir):
            os.makedirs(self.debugging_dir, exist_ok=True)
            logging.info(f"The directory {self.debugging_dir} has been created.")
        
    def __call__(self, topology, substrate_area=(10000, 10000)):
        self.params = QplacementParam(substrate_area=substrate_area, 
                                      topology=topology,
                                      scale_factor = self.bm_setting ['scale_factor'],
                                      q_freq_range = self.bm_setting['q_freq_range'],
                                      res_freq_range = self.bm_setting['res_freq_range'],
                                      q_th = self.bm_setting['q_th'],
                                      res_th = self.bm_setting['res_th'],
                                      padding_size = self.bm_setting['padding_size'],
                                      partition_size = self.bm_setting['partition_size'],
                                      qubit_pad_size = self.bm_setting['qubit_pad_size'],
                                      qubit_padding_size = self.bm_setting['qubit_padding_size'],
                                      benchmark_dir = self.bm_setting['benchmark_dir'],
                                      freq_assign = self.bm_setting['freq_assign'],
                                      partition = self.bm_setting['partition'],
                                      debugging = self.bm_setting['debugging'],
                                      net_weight = self.bm_setting['net_weight'],
                                      seed = self.bm_setting['seed'],
                                      )
        self.params.debugging_dir = self.debugging_dir
        self.db = QplacementDatabase()
        # if self.params.debugging:
        #     logging.getLogger().setLevel(logging.DEBUG)
        # Build connectivity graph
        graph_builder = ConnectivityGraphBuilder(qubits=qubit_num_dict[topology][1], 
                                                 topology=qubit_num_dict[topology][0], 
                                                 debugging=self.params.debugging)
        self.db.c_graph, self.db.c_graph_pos_map = graph_builder.get_connectivity_graph()
        logging.info(f'Topology: {topology}, #Qubit: {len(self.db.c_graph.nodes())}')
        """ Assign Frequency """ 
        self.f_assigner = FrequencyAssigner(self.db.c_graph)
        self.db.qubit_to_freq_map = self.f_assigner.assign_qubit_frequencies(self.params.q_freq_range, self.params.q_th)
        self.db.edge_to_freq_map =  self.f_assigner.assign_resonator_frequencies(self.params.res_freq_range, self.params.res_th)

        logging.debug(f'pos : {self.db.c_graph_pos_map}')
        logging.debug(f'qubit_to_freq_map ({len(self.db.qubit_to_freq_map.keys())}): {self.db.qubit_to_freq_map}')
        logging.debug(f'edge_to_freq_map ({len(self.db.edge_to_freq_map.keys())}): {self.db.edge_to_freq_map}') 
        if self.params.graph_debugging:
            fig, ax = plt.subplots()
            nx.draw(self.db.c_graph, self.db.c_graph_pos_map, with_labels=True, node_size=300, node_color="skyblue")
            ax.set_title(topology)
            plt.tight_layout()
            fig_path = f'{self.debugging_dir}/{topology}.png'
            plt.savefig(fig_path)
            logging.info(f'save figure {fig_path}')

        """ Prepare the data for Placer """ 
        self.formator = DesignFormator(self.params, self.db)
        self.formator(debugging=self.params.debugging)
                
        loaded_component_to_freq_map = {**self.db.qubit_to_freq_map, **self.db.poly_to_freq_map}
        logging.info('Num wireblk: {}, Num components: {}'.format(
            len(self.db.poly_to_freq_map.keys()), 
            len(loaded_component_to_freq_map.keys()),
            ))
        logging.info(f'Qubit size: {next(iter(self.db.qubit_geo_data.values()))}')
        logging.info(f'Wireblock size: {next(iter(self.db.wireblk_def_data.values()))[0]}')
        
        area_x, area_y = substrate_area[0], substrate_area[1]
        # file_name = f'{topology}_wp' if self.params.partition else topology
        # file_name = f'{file_name}_wf'if self.params.freq_assign else file_name
        file_name = f'{topology}_{self.suffix}'
        benchmark_dir = f"benchmarks/{self.params.benchmark_dir}/{file_name}"

        if not os.path.isdir(benchmark_dir):
            os.makedirs(benchmark_dir, exist_ok=True)
            logging.info(f"The directory {benchmark_dir} has been created.")

        file_paths = {
            "lef": f"{benchmark_dir}/{file_name}.lef",
            "def": f"{benchmark_dir}/{file_name}.def", 
            "json_dir": f'{self.params.param_json_dir}/{topology}/{self.suffix}',
        }
        self.params.file_paths = file_paths
        self.params.file_name = file_name

        """  Write the design files for placer """ 
        self.lefwriter  = LefFileWriter(self.params)
        self.defwriter  = DefFileWriter(self.params, area_x=area_x, area_y=area_y)
        self.jsonwriter = JsonFileWriter(self.params)
        self.db.lef_edge_macro_map = self.lefwriter(self.params.file_paths["lef"], db=self.db, debugging=False)
        self.db.def_nodes_order, self.db.net_weight_list, self.db.wireblk_in_group = self.defwriter(self.params.file_paths["def"], self.db)

        self.collision_checker = FreqCollisionChecker(self.params.file_paths["def"], self.params, self.db)
        self.db.potential_collision_map = self.collision_checker.build_potential_freq_collisions_map(self.db.def_nodes_order)
        # print(f'p_collision_map : {p_collision_map}')

        self.jsonwriter(self.db)

        """ Check Collision """ 
        self.collision_checker.plot_collisions()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default='benchmark_params.json', help='benchmark parameters json file')
    parser.add_argument('--suffix', default='', help='the suffix for experiment')
    parser.add_argument('--topology', default='grid-25', help='the connectivity topology of the design')
    args = parser.parse_args()
    bmg = BenchmarkGenerator(args.benchmark, suffix=args.suffix)
    bmg(topology=args.topology, substrate_area=area_dict[args.topology])