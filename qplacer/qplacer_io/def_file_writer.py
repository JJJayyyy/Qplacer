
import logging

class DefFileWriter:
    def __init__(self, 
                 params, 
                 design_name="METAL_TEST", 
                 area_x=10000, 
                 area_y=10000,
                 ): 
        
        self.design_name = design_name
        self.partition = params.partition
        self.site_size = params.partition_size
        self.microns = params.scale_factor
        self.net_weight = params.net_weight
        self.area_x = area_x
        self.area_y = area_y
        


    def __call__(self, 
                 file_path, 
                 db,
                 ):
        
        qubit_def_data=db.qubit_geo_data
        wireblk_def_data=db.wireblk_def_data
        pin_connections=db.pin_distribution
        edge_macro_map=db.lef_edge_macro_map
                
        def_str = self.def_header()
        assert len(edge_macro_map.keys()) > 0, f'edge_macro_map is empty, run Lef file writer first'
        comp_def_str, nodes_order, wireblk_in_group = self.format_components_to_def(
            qubit_def_data=qubit_def_data, 
            wireblk_def_data=wireblk_def_data,
            edge_macro_map=edge_macro_map,
            )
        def_str = def_str + comp_def_str
        def_str += self.format_pin_to_def()
        net_string, net_weight_list = self.format_connections_to_def(pin_connections)
        def_str += net_string
        def_str += self.def_footer()
        with open(file_path, 'w') as file:
            file.write(def_str)
            logging.info(f"Content written to {file_path}")
        return nodes_order, net_weight_list, wireblk_in_group


    def def_header(self):
        def_str  =  'VERSION 5.8 ;\n\nBUSBITCHARS "[]" ;\n\nDIVIDERCHAR "/" ;\n\n'
        def_str += f"DESIGN {self.design_name} ;\n\n"
        def_str += f"UNITS DISTANCE MICRONS {self.microns} ;\n\n"
        def_str += f"DIEAREA ( 0 0 ) ( {self.area_x} {self.area_y} ) ;\n\n"
        num_row = int(self.area_y / (self.site_size * self.microns))
        num_col = int(self.area_x / (self.site_size * self.microns))
        for i in range(num_row):
            direction = "N" if i % 2 == 0 else "FS"
            step = int(self.site_size*self.microns)
            core_site_y = int(self.site_size*self.microns)*(i + 1)
            def_str += f"ROW CORE_ROW_{i:<2} CoreSite 0 {core_site_y:<5} {direction:<2} DO {num_col} BY 1 STEP {step} 0 ;\n"
        def_str += "\n"
        return def_str


    def format_components_to_def(self, 
                                 qubit_def_data:dict, 
                                 wireblk_def_data:dict={}, 
                                 q_type='QUBIT', 
                                #  w_type='WIRE_BLK', 
                                 edge_macro_map:dict={},
                                 scale=1000
                                 ):
        nodes_order = dict()
        num = 0
        def_str = f"COMPONENTS {len(qubit_def_data.keys())+len(wireblk_def_data.keys())} ;\n"
        for q_name, q_data in qubit_def_data.items():
            q_centroid = q_data["geometry"].centroid
            x, y = int(round(q_centroid.x, 2) * scale), int(round(q_centroid.y, 2) * scale)
            def_str += f'- {q_name:<3} {q_type} + PLACED ( {x:<5} {y:<5} ) N ;\n'
            nodes_order[q_name] = num
            num += 1

        if self.partition:
            wireblk_in_group = []
            w_type = edge_macro_map["wireblk"]
            for edge, w_data_list in wireblk_def_data.items():
                wireblk_in_num = []
                for w_data in w_data_list:
                    w_centroid = w_data[1].centroid
                    x, y = int(round(w_centroid.x, 2) * scale), int(round(w_centroid.y, 2) * scale)
                    def_str += f'- {w_data[0]} {w_type} + PLACED ( {x:<5} {y:<5} ) N ;\n'
                    nodes_order[w_data[0]] = num
                    wireblk_in_num.append(num)
                    num += 1
                wireblk_in_group.append(wireblk_in_num)
        else:
            wireblk_in_group = []
            for edge, w_data_list in wireblk_def_data.items():
                w_type = edge_macro_map[edge]
                for w_data in w_data_list:
                    w_centroid = w_data[1].centroid
                    x, y = int(round(w_centroid.x, 2) * scale), int(round(w_centroid.y, 2) * scale)
                    def_str += f'- {w_data[0]} {w_type} + PLACED ( {x:<5} {y:<5} ) N ;\n'
                    nodes_order[w_data[0]] = num
                    num += 1

        def_str += f"END COMPONENTS\n\n"
        return def_str, nodes_order, wireblk_in_group


    def format_pin_to_def(self, pins=[]):
        def_str = f"PINS {len(pins)} ;\n"
        def_str += f"END PINS\n\n"
        return def_str


    def format_connections_to_def(self, connection:dict):
        net_weight_list = []
        def_str = f"NETS {len(connection.keys())} ;\n"
        for i, (edge, pins) in enumerate(connection.items()):
            if 'Q' not in edge[0] and 'Q' not in edge[1]:
                net_weight_list.append(self.net_weight)
            else:
                net_weight_list.append(1)
            def_str += f'- net{i}\n'
            def_str += f'( {edge[0]} {pins[0]} ) ( {edge[1]} {pins[1]} ) ;\n'
        def_str += f"END NETS\n\n"
        return def_str, net_weight_list


    def def_footer(self):
        def_str = "END DESIGN\n"
        return def_str