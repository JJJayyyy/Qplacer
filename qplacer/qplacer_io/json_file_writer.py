import os
import json
import logging
import pickle
import dill

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
    "density_weight" : 8e-5,
    "gamma" : 4.0,
    "random_seed" : 3,
    "ignore_net_degree" : 100,
    "enable_fillers" : 1,
    "gp_noise_ratio" : 0.025,
    "global_place_flag" : 1,
    "legalize_flag" : 1,
    "stop_overflow" : 0.1,
    "dtype" : "float32",
    "plot_flag" : 1,
    "random_center_init_flag" : 0,
    "sort_nets_by_degree" : 0,
    "num_threads" : 8,
    "deterministic_flag" : 0,
    "frequency_assign" : 1,
    "frequency_density_weight": [],
}


class JsonFileWriter:
    def __init__(self,
                 params,
                 ):
        self.data = data_template
        self.params = params
        self.data['qubit_dist_threhold'] = params.q_th*params.scale_factor      # convert to nano
        self.data['coupler_dist_threhold'] = params.res_th*params.scale_factor  # convert to nano
 
    def __call__(self, db):
        assert len(db.potential_collision_map.keys()) > 0, f'potential_collision_map is empty1'
        file_name = self.params.file_name
        benchmark_dir = self.params.benchmark_dir
        json_dir = self.params.file_paths["json_dir"]

        if not os.path.isdir(json_dir):
            os.makedirs(json_dir, exist_ok=True)
            logging.info(f"The directory {json_dir} has been created.")
        
        self.data["lef_input"] = [f"benchmarks/{benchmark_dir}/{file_name}/{file_name}.lef"]
        self.data["def_input"] = f"benchmarks/{benchmark_dir}/{file_name}/{file_name}.def"
        json_filename = f'{json_dir}/{file_name}.json'
        
        if self.params.freq_assign:
            self.data["num_qubit"] = len(db.qubit_to_freq_map.keys())
            self.data["num_coupler"] = len(db.poly_to_freq_map.keys())
            self.data["net_weight_list"] = db.net_weight_list
            assert len(db.wireblk_in_group) > 0
            self.data["wireblk_in_group"] = db.wireblk_in_group
            self.data["potential_collision_map"] = db.potential_collision_map
            self.data["frequency_density_weight"] = 8e-5
            self.data["random_center_init_flag"] = 1
        else:
            self.data["num_frequency_options"] = []
            self.data["frequency_density_weight"] = 0
            self.data["frequency_assign"] = 0

        with open(json_filename, 'w') as f:
            json.dump(self.data, f, indent=4)
            logging.info(f"Content written to {json_filename}")

        # def find_non_pickleable_attributes(obj):
        #     non_pickleable = []
        #     for attr in dir(obj):
        #         if not attr.startswith('__'):
        #             try:
        #                 pickle.dumps(getattr(obj, attr))
        #             except (pickle.PicklingError, TypeError):
        #                 non_pickleable.append(attr)
        #     return non_pickleable
        # non_pickleable_attrs = find_non_pickleable_attributes(db)
        # print("Non-pickleable attributes:", non_pickleable_attrs)

        with open(f'{json_dir}/{file_name}_params.pkl', 'wb') as file:
            pickle.dump(self.params, file)

        with open(f'{json_dir}/{file_name}_db.pkl', 'wb') as file:
            pickle.dump(db, file)
