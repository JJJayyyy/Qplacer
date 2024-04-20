import os
import json
import logging
import pickle

from qplacer_bm.qplacement_database import QplacementDatabase, FrequencyDatabase
from qplacer_bm.qplacement_param import QplacementParam


data_template = {
    "lef_input": [],
    "def_input": "",
    "gpu" : 0,
    "num_bins_x" : 512,
    "num_bins_y" : 512,
    "global_place_stages" : [
        {"num_bins_x" : 512, 
        "num_bins_y" : 512, 
        "iteration" : 1000, 
        "learning_rate" : 0.01, 
        "wirelength" : "weighted_average", 
        "optimizer" : "nesterov"}
    ],
    "target_density" : 1.0,
    "density_weight" : 8e-4,
    #  "density_weight" : 8e-5,
    "gamma" : 4.0,
    "random_seed" : 100,
    "ignore_net_degree" : 100,
    "enable_fillers" : 1,
    "gp_noise_ratio" : 0.025,
    "random_center_init_flag" : 0,
    "global_place_flag" : 1,
    "legalize_flag" : 1,
    "qplacer_legalize_flag" : 0,
    "stop_overflow" : 0.1,
    "dtype" : "float32",
    "plot_flag" : 1,
    "sort_nets_by_degree" : 0,
    "num_threads" : 8,
    "deterministic_flag" : 0,
    "frequency_assign" : 1,
    "frequency_density_weight": [],
}
        


class JsonFileWriter:
    def __init__(self,
                 params: QplacementParam,
                 ):
        self.data = data_template
        self.params = params
        self.data['qubit_dist_threhold'] = params.q_th*params.scale_factor      # convert to nano
        self.data['coupler_dist_threhold'] = params.res_th*params.scale_factor  # convert to nano
 
    def __call__(self, 
                 db: QplacementDatabase, 
                 save_freq=True):
        assert len(db.potential_collision_map.keys()) > 0, f'potential_collision_map is empty1'
        file_name = self.params.file_name
        benchmark_dir = self.params.benchmark_dir
        json_dir = self.params.file_paths["json_dir"]

        if not os.path.isdir(json_dir):
            os.makedirs(json_dir, exist_ok=True)
            logging.info(f"The directory {json_dir} has been created.")
        
        # self.data["lef_input"] = [f"benchmarks/{benchmark_dir}/{file_name}/{file_name}.lef"]
        # self.data["def_input"] = f"benchmarks/{benchmark_dir}/{file_name}/{file_name}.def"
        self.data["lef_input"] = [self.params.file_paths['lef']]
        self.data["def_input"] = self.params.file_paths['def']
        json_filename = f'{json_dir}/{file_name}.json'
        self.data["num_qubit"] = len(db.qubit_to_freq_map.keys())
        self.data["num_coupler"] = len(db.poly_to_freq_map.keys())
        self.data["density_weight"] = self.params.density_weight
        
        if self.params.freq_assign:
            self.data["net_weight_list"] = db.net_weight_list
            assert len(db.wireblk_in_group) > 0
            self.data["wireblk_in_group"] = db.wireblk_in_group
            self.data["potential_collision_map"] = db.potential_collision_map
            self.data["frequency_density_weight"] = self.params.frequency_density_weight
            self.data["random_center_init_flag"] = self.params.random_center_init_flag
            self.data["global_place_flag"] = 1
            self.data["legalize_flag"] = 1
            self.data["qplacer_legalize_flag"] = 1
        elif "legal" in self.params.file_name:
            print("classical placer with qplacer legalization")
            self.data["frequency_density_weight"] = 0
            self.data["frequency_assign"] = 0
            self.data["random_center_init_flag"] = self.params.random_center_init_flag
            self.data["qplacer_legalize_flag"] = 1
            self.data["wireblk_in_group"] = db.wireblk_in_group
        else:
            self.data["frequency_density_weight"] = 0
            self.data["frequency_assign"] = 0
            self.data["random_center_init_flag"] = self.params.random_center_init_flag
            self.data["qplacer_legalize_flag"] = 0

        with open(json_filename, 'w') as f:
            json.dump(self.data, f, indent=4)
            logging.info(f"Content written to {json_filename}")

        with open(f'{json_dir}/{file_name}_params.pkl', 'wb') as file:
            pickle.dump(self.params, file)
            logging.info(f"Save params to {json_dir}/{file_name}_params.pkl")

        with open(f'{json_dir}/{file_name}_db.pkl', 'wb') as file:
            pickle.dump(db, file)
            logging.info(f"Save data   to {json_dir}/{file_name}_db.pkl")

        if save_freq:
            freq_dir = self.params.file_paths["freq"]
            freq_db = FrequencyDatabase(db.qubit_to_freq_map, db.edge_to_freq_map)
            with open(freq_dir, 'wb') as file:
                pickle.dump(freq_db, file)
                logging.info(f"Save freqdb to {self.params.file_paths['freq']}")
            
