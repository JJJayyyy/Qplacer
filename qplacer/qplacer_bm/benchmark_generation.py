from shapely.errors import ShapelyDeprecationWarning
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import warnings
import logging
import argparse
import json
import sys
import os

from qplacer_bm.qplacement_param import QplacementParam
from qplacer_bm.qplacement_database import QplacementDatabase
from qplacer_bm.connectivity import ConnectivityGraphBuilder
from qplacer_bm.frequency_assignment import FrequencyAssigner
from qplacer_bm.design_format import DesignFormator
from qplacer_bm.collision_check import FreqCollisionChecker

from qplacer_bm.bm_io.lef_file_writer import LefFileWriter
from qplacer_bm.bm_io.def_file_writer import DefFileWriter
from qplacer_bm.bm_io.json_file_writer import JsonFileWriter


warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.root.name = 'QPlacer'
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)-7s] %(name)s - %(message)s',
                    stream=sys.stdout)


""" parameters """
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
        if isinstance(bm_json, str):
            with open(bm_json, 'r') as json_file:
                self.bm_setting = json.load(json_file)
        elif isinstance(bm_json, dict):
            self.bm_setting = bm_json
        else:
            raise Exception(f"Unaccept type :{type(bm_json)}")

        self.debugging_dir = 'logs'
        if not os.path.isdir(self.debugging_dir):
            os.makedirs(self.debugging_dir, exist_ok=True)
            logging.info(f"The directory {self.debugging_dir} has been created.")
        
    def __call__(self, topology, substrate_area=(10000, 10000), frequency_database=None):
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
                                      frequency_density_weight = self.bm_setting["frequency_density_weight"],
                                      density_weight = self.bm_setting["density_weight"],
                                      random_center_init_flag = self.bm_setting["random_center_init_flag"],
                                      )
        self.params.debugging_dir = self.debugging_dir
        self.db = QplacementDatabase(chip_size_x=substrate_area[0], chip_size_y=substrate_area[1])
        area_x, area_y = substrate_area[0], substrate_area[1]
        file_name = f'{topology}_{self.suffix}'
        benchmark_dir = f"benchmarks/{self.params.benchmark_dir}/{file_name}"
        # if self.params.debugging:
        #     logging.getLogger().setLevel(logging.DEBUG)
        
        """ Build connectivity graph """
        graph_builder = ConnectivityGraphBuilder(qubits=qubit_num_dict[topology][1], 
                                                 topology=qubit_num_dict[topology][0], 
                                                 debugging=self.params.debugging)
        self.db.c_graph, self.db.c_graph_pos_map = graph_builder.get_connectivity_graph()
        logging.info(f'Topology: {topology}, #Qubit: {len(self.db.c_graph.nodes())}')

        """ Assign Frequency """ 
        if frequency_database == None:
            logging.warning("<CREATE> Frequency Assigner, assign frequency to Qubits and Couplers")
            self.f_assigner = FrequencyAssigner(self.db.c_graph, seed=self.params.seed)
            self.db.qubit_to_freq_map = self.f_assigner.assign_qubit_frequencies(self.params.q_freq_range, self.params.q_th)
            self.db.edge_to_freq_map =  self.f_assigner.assign_resonator_frequencies(self.params.res_freq_range, self.params.res_th)
        else:
            logging.warning("<LOAD> Qubits/Couplers frequency from database")
            graph_qubits = sorted(list(self.db.c_graph.nodes()))
            db_qubits = sorted(set(frequency_database.qubit_to_freq_map.keys()))
            assert graph_qubits == db_qubits, "graph node: {}, db freq_q: {}".format(graph_qubits, db_qubits)
            assert set(self.db.c_graph.edges()) == set(frequency_database.edge_to_freq_map.keys()), \
                "graph edge: {}, db freq_e: {}".format(set(self.db.c_graph.edges()), 
                                                     set(frequency_database.edge_to_freq_map.keys()))
            self.db.qubit_to_freq_map = frequency_database.qubit_to_freq_map
            self.db.edge_to_freq_map = frequency_database.edge_to_freq_map

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
        logging.info('#Wireblk: {}, #Components: {}'.format(
            len(self.db.poly_to_freq_map.keys()), 
            len(loaded_component_to_freq_map.keys()),
            ))
        logging.info("Qubit size: {} mm^2, Wireblock size: {} mm^2".format(
            next(iter(self.db.qubit_geo_data.values()))['geometry'].area,
            round(next(iter(self.db.wireblk_def_data.values()))[0][1].area, 6)
            ))

        if not os.path.isdir(benchmark_dir):
            os.makedirs(benchmark_dir, exist_ok=True)
            logging.info(f"The directory {benchmark_dir} has been created.")

        file_paths = {
            "lef": f"{benchmark_dir}/{file_name}.lef",
            "def": f"{benchmark_dir}/{file_name}.def", 
            "json_dir": f'{self.params.param_json_dir}/{topology}/{self.suffix}',
            "freq": f'{self.params.param_json_dir}/{topology}/freq_{topology}.pkl',
            "result_def" : f"results/{file_name}/{file_name}.gp.def",
        }
        self.params.file_paths = file_paths
        self.params.file_name = file_name

        """  Create the Benchmark files for placer """ 
        self.lefwriter  = LefFileWriter(self.params)
        self.defwriter  = DefFileWriter(self.params, area_x=area_x, area_y=area_y)
        self.jsonwriter = JsonFileWriter(self.params)
        self.db.lef_edge_macro_map = self.lefwriter(self.params.file_paths["lef"], db=self.db, debugging=False)
        self.db.def_nodes_order, self.db.net_weight_list, self.db.wireblk_in_group = self.defwriter(self.params.file_paths["def"], self.db)

        self.collision_checker = FreqCollisionChecker(self.params.file_paths["def"], self.params, self.db)
        self.db.potential_collision_map = self.collision_checker.build_potential_freq_collisions_map(self.db.def_nodes_order)

        self.jsonwriter(self.db)

        """ Check Collision """ 
        self.collision_checker.plot_collisions(suffix=f"_{self.suffix}_human")
        min_bounding_rect = self.collision_checker.get_min_bounding_rect()
        poly_area = self.collision_checker.get_total_area()
        logging.info("Area (polygons): {} mm^2, Side Length of total area: {:.1f} mm, Area (MBR): {} mm^2".format(
            poly_area/(self.params.scale_factor**2),
            np.sqrt(poly_area)/self.params.scale_factor,
            min_bounding_rect.area/(self.params.scale_factor**2)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default='benchmark_params.json', help='benchmark parameters json file')
    parser.add_argument('--suffix', default='', help='the suffix for experiment')
    parser.add_argument('--topology', default='grid-25', help='the connectivity topology of the design')
    parser.add_argument('--area_x', type=int, default=10000, help='x axie size of the substrate')
    parser.add_argument('--area_y', type=int, default=10000, help='y axie size of the substrate')
    args = parser.parse_args()
    bmg = BenchmarkGenerator(args.benchmark, suffix=args.suffix)
    bmg(topology=args.topology, substrate_area=(args.area_x, args.area_y))