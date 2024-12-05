from qiskit_metal import designs

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
    'grid-25': ('new_grid', 25),  
    'grid-64':('new_grid', 64),
    'falcon':('ibm_falcon', 27), 
    'hummingbird':('ibm_hummingbird', 65), 
    'eagle':('ibm_eagle', 127),
    'Aspen-11':('oxtagon', 40), 
    'Aspen-M':('oxtagon', 80), 
    'xtree-53': ('xtree', 53)
}

class QplacementDatabase:
    def __init__(self, chip_size_x=10000, chip_size_y=10000):
        self.metal_design = designs.DesignPlanar(overwrite_enabled=True)
        self.metal_design.chips['main']['size']['size_x'] = f"{chip_size_x}um" 
        self.metal_design.chips['main']['size']['size_y'] = f"{chip_size_y}um"
        self.metal_design.chips['main']['size']['center_x'] = f"{chip_size_x/2}um" 
        self.metal_design.chips['main']['size']['center_y'] = f"{chip_size_y/2}um"

        self.c_graph = None
        self.c_graph_pos_map = dict()

        # component to frequency
        self.qubit_to_freq_map = dict()
        self.edge_to_freq_map = dict()
        self.wireblk_to_freq_map = dict()
        self.poly_to_freq_map = dict()

        # component to edge
        self.wireblk_edge_map = dict()
        self.poly_to_edge = dict() 

        self.wireblk_polygon_map = dict()

        # design geometry
        self.qubit_to_metal_map = dict()
        self.qubit_geo_data = dict() 
        self.path_geo_data = dict()

        self.wireblk_def_data = dict(), 
        
        # pin distribution
        self.qubit_pin_distribution = dict()
        self.pin_distribution = dict()
        
        self.lef_edge_macro_map = dict()
        self.def_nodes_order = dict()
        self.net_weight_list = list()
        self.wireblk_in_group = list()

        self.potential_collision_map = dict()
        


    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('metal_design', None) 
        state.pop('qubit_to_metal_map', None)
        return state
    


class FrequencyDatabase:
    def __init__(self,
                 qubit_to_freq_map : dict, 
                 edge_to_freq_map : dict):
        self.qubit_to_freq_map = qubit_to_freq_map
        self.edge_to_freq_map = edge_to_freq_map