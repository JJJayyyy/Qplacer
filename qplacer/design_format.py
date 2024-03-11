import uuid
import math
import logging
from collections import defaultdict, OrderedDict
from shapely.geometry import Polygon, LineString
import os
os.environ['QISKIT_METAL_HEADLESS'] = '1'

from qiskit_metal import designs
from qiskit_metal.qlibrary import QRoute, BaseQubit
from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
import numpy as np
# from Qplacer.route import PolygonConnector
# from Qplacer.route import RouteMixed
import matplotlib.pyplot as plt

pin_to_loc = {'nw': (-1, +1), 'sw': (-1, -1), 'se': (+1, -1), 'ne': (+1, +1)}


class DesignFormator:
    def __init__(self, 
                 params,
                 database,
                 ):
        self.db = database
        self.params = params
        self.coresite_size_dict = dict()
        self.wireblk_polygon_map = dict()



    def __call__(self, debugging=False):
        self.debugging = debugging
        self.scale_factor = self.params.scale_factor
        self.partition = self.params.partition
        self.qubit_size = self.params.qubit_size
        self.partition_size = self.params.partition_size
        self.padding_size = self.params.padding_size

        self.db.metal_design.delete_all_components()

        self.db.qubit_to_metal_map = self.create_metal_qubits()
        self.db.qubit_geo_data, self.db.path_geo_data = self.get_geo_data_from_design(debugging=self.debugging)
        self.db.qubit_pin_distribution = self.distribute_pins_equally()
        self.db.wireblk_def_data, self.db.pin_distribution, self.db.poly_to_edge, self.db.poly_to_freq_map = self.create_edges_def_data()
        self.db.wireblk_polygon_map = self.wireblk_polygon_map


    def create_metal_qubits(self):
        qubit_to_metal_map = dict()
        for q_name, coordinates in self.db.c_graph_pos_map.items():
            x, y = coordinates
            q_options = dict(
                pos_x = f"{round(x * self.scale_factor)}um",
                pos_y = f"{round(y * self.scale_factor)}um",
                gds_cell_name = q_name, 
                pad_width='400 um',
                pocket_width='600um',
                pocket_height='600um',
                connection_pads={pin: dict(loc_W=loc_w, 
                                    loc_H=loc_h, 
                                    pad_width='80um', 
                                    pad_gap='50um', 
                                    cpw_extend='0um') for pin, (loc_w, loc_h) in pin_to_loc.items()},
            )
            qubit = TransmonPocket(self.db.metal_design, q_name, options=q_options)
            qubit_to_metal_map[q_name] = qubit
        return qubit_to_metal_map
    

    def create_metal_resonator(self, 
                       design, 
                       wireblk_polygons,
                       edges_to_poly_map,
                       ):
        scaled_wireblk_polygons = dict()
        for wire_name, poly in wireblk_polygons.items():
            scaled_polygon_points = [(x/self.scale_factor, y/self.scale_factor) for x, y in poly.exterior.coords]
            scaled_polygon = Polygon(scaled_polygon_points)
            scaled_wireblk_polygons[wire_name] = scaled_polygon

        edge_path_dict = dict()
        for edge, pins in self.db.qubit_pin_distribution.items():
            head = self.db.qubit_geo_data[edge[0]]['pins'][pins[0]].centroid
            tail = self.db.qubit_geo_data[edge[1]]['pins'][pins[1]].centroid
            poly_list = edges_to_poly_map[edge]

            print(f'edge: {edge}, head: {head}, tail: {tail}, poly_list: {poly_list}')
            for p_name in poly_list:
                print(f'{p_name} : {scaled_wireblk_polygons[p_name]}')

            polygons_dict = {p_name: scaled_wireblk_polygons[p_name] for p_name in poly_list}
            p_connector = PolygonConnector()
            path, distance = p_connector(head=head, tail=tail, polygons_dict=polygons_dict, debugging=True)
            edge_path_dict[edge] = path
            print(path, poly_list)
            coupler_name = f'{edge[0]}-{edge[1]}'
            if coupler_name in design.components:
                print(f"Component {coupler_name} exists in the design.")
                design.delete_component(component_name=coupler_name)
            else:
                print(f"Component {coupler_name} does not exist in the design.")

            ops=dict(fillet='90um')
            anchors = OrderedDict()
            print(len(path))
            for p in path:
                print(p)

            for i, point in enumerate(path[1:-1]):
                anchors[i] = np.array([point.x, point.y])

            between_anchors = OrderedDict() # S, M, PF
            for i in range(len(poly_list)):
                between_anchors[2*i] = "PF"
                between_anchors[2*i+1] = "M"
            between_anchors[2*len(poly_list)] = "PF"
            print(between_anchors, len(between_anchors))
            print(anchors, len(anchors))
            assert len(between_anchors) == len(anchors) + 1

            options = {'pin_inputs': {
                        'start_pin': {'component': edge[0], 'pin': pins[0]}, 
                        'end_pin'  : {'component': edge[1], 'pin': pins[1]}
                        },
                        'total_length': f'{len(poly_list)*4}mm',
                        'chip' : 'main',
                        'layer': '1',
                        'trace_width': 'cpw_width',
                        'step_size'  : '0.25mm',
                        'anchors': anchors,
                        'between_anchors': between_anchors,
                        'advanced': {'avoid_collision': 'true'},
                        'meander' : {
                        'spacing': '200um',
                        'asymmetry': '0um'
                        },
                        'snap': 'true',
                        'lead': {
                        'start_straight': '0.1mm',
                        'end_straight'  : '0.1mm',
                        },
                        **ops
                    }

            route = RouteMixed(design, name=coupler_name, options=options)

        print(f'len of design component: {len(design.components.items())}')
        qubit_geo_data, path_geo_data = self.get_geo_data_from_design()
        print(path_geo_data)

        for q_name, path_data in path_geo_data.items():
            if isinstance(path_data['geometry'], LineString):
                line = path_data['geometry']
                line_str = 'LINESTRING (' + ', '.join(f'{x:.3f} {y:.3f}' for x, y in zip(*line.xy)) + ')'
                print(f'{q_name} is LineString: {line_str}')
                x, y = line.xy
                plt.figure(figsize=(12, 4))
                plt.plot(x, y, marker='o', color='blue', linestyle='-', linewidth=2)
                plt.title(f'LineString for {q_name}')
                plt.xlabel('X-axis')
                plt.ylabel('Y-axis')
                plt.grid(True)
                plt.show()
            else:
                buffer_width = 0.2
                polygon = path_data['geometry'].buffer(buffer_width)
                coords = list(polygon.exterior.coords)
                polygon_str = 'POLYGON ((' + ', '.join(f'{x:.3f} {y:.3f}' for x, y in coords) + '))'
                print(polygon_str)

                x, y = polygon.exterior.xy
                plt.figure(figsize=(12, 4))
                plt.fill(x, y, alpha=0.5, fc='blue', ec='black')
                plt.title(f'Polygon for {q_name}')
                plt.xlabel('X-axies')
                plt.ylabel('Y-axies')
                plt.grid(True)
                plt.show()


    def get_geo_data_from_design(self, debugging=False):
        qubit_geo = self.db.metal_design.qgeometry.tables['poly']
        path_geo = self.db.metal_design.qgeometry.tables['path']
        qubit_geo_data = dict()
        path_geo_data = dict()

        for q_name, component in self.db.metal_design.components.items():
            if isinstance(component, BaseQubit):
                qubit_geometry_value = qubit_geo.loc[
                    (qubit_geo['component'] == component.id) & (qubit_geo['name'] == 'rect_pk'), 'geometry'].iloc[0]
                logging.debug(f"Qubit name: {q_name}, {component.id}")
                logging.debug(f"Qubit Loc: {qubit_geometry_value}", type(qubit_geometry_value))
                qubit_geo_data[q_name] = {'id': component.id, 'geometry': qubit_geometry_value, 'pins': dict()}
                pin_list = list(component.options.get('connection_pads', 'N/A').keys())
                for pin in pin_list:
                    pin_geometry_value = path_geo.loc[
                        (path_geo['component'] == component.id) & (path_geo['name'] == f'{pin}_wire'), 'geometry'].iloc[0]
                    qubit_geo_data[q_name]['pins'][pin] = pin_geometry_value
                    logging.debug(f"pin name : {pin:<10}\tPin Geo : {pin_geometry_value}")             
            elif isinstance(component, QRoute):
                path_geometry_value = path_geo.loc[
                    (path_geo['component'] == component.id) & (path_geo['name'] == 'trace'), 'geometry'].iloc[0]
                path_geo_data[q_name] = {'id': component.id, 'geometry': path_geometry_value}
                logging.debug(f"Component name: {q_name}, {component.id}")
                logging.debug(type(path_geometry_value), path_geometry_value)

            logging.debug("---------")
        return qubit_geo_data, path_geo_data


    def distribute_pins_equally(self):
        edges = list(self.db.c_graph.edges())
        node_pins = defaultdict(lambda: {'ne': 0, 'nw': 0, 'se': 0, 'sw': 0})
        pin_connections = defaultdict(list)

        def get_least_used_pin(node, pins_to_compare):
            filtered_pins = {pin: node_pins[node][pin] for pin in pins_to_compare}
            return min(filtered_pins, key=filtered_pins.get)

        for edge in edges:
            node_a, node_b = edge
            pos_a = self.db.c_graph_pos_map[node_a]
            pos_b = self.db.c_graph_pos_map[node_b]
            if pos_b[0] > pos_a[0] and pos_b[1] == pos_a[1]:
                pin_a = get_least_used_pin(node_a, ['ne', 'se'])
                pin_b = get_least_used_pin(node_b, ['nw', 'sw'])
            elif pos_b[0] < pos_a[0] and pos_b[1] == pos_a[1]:
                pin_a = get_least_used_pin(node_a, ['nw', 'sw'])
                pin_b = get_least_used_pin(node_b, ['ne', 'se'])
            elif pos_b[1] > pos_a[1] and pos_b[0] == pos_a[0]:
                pin_a = get_least_used_pin(node_a, ['ne', 'nw'])
                pin_b = get_least_used_pin(node_b, ['se', 'sw'])
            elif pos_b[1] < pos_a[1] and pos_b[0] == pos_a[0]:
                pin_a = get_least_used_pin(node_a, ['sw', 'se'])
                pin_b = get_least_used_pin(node_b, ['nw', 'ne'])
            elif pos_a[0] < pos_b[0] and pos_a[1] < pos_b[1]:
                pin_a, pin_b = 'ne', 'sw'
            elif pos_a[0] > pos_b[0] and pos_a[1] > pos_b[1]:
                pin_a, pin_b = 'sw', 'ne'
            elif pos_b[0] < pos_a[0] and pos_b[1] > pos_a[1]:
                pin_a, pin_b = 'nw', 'se'
            elif pos_b[0] > pos_a[0] and pos_b[1] < pos_a[1]:
                pin_a, pin_b = 'se', 'nw'
            else:
                raise ValueError("Undefined relative position between nodes")    
            
            node_pins[node_a][pin_a] += 1
            node_pins[node_b][pin_b] += 1
            pin_connections[edge] = (pin_a, pin_b)

        return pin_connections


    def get_edge_wirelength(self, freq):
        # L = v_0/2f    v_0 = 1.3e8
        if isinstance(freq, float):
            return 1.3e8*1000/(2*freq)
        elif isinstance(freq, str):
            return 65/float(freq.lower().replace('ghz', '').strip())
        else:
            raise Exception(f"Type {type(freq)} is currently not acceptable")


    def create_edges_def_data(self):   
        def generate_polygon_id():
            return f"poly_{uuid.uuid4().hex[:5]}"       # increase digits if needed

        new_pin_distribution = defaultdict(list)
        wireblk_def_data = dict()
        poly_to_edge = dict()
        poly_to_freq_map = dict()

        for j, (edge, pins) in enumerate(self.db.qubit_pin_distribution.items()):
            edge_freq = self.db.edge_to_freq_map[edge]
            edge_wirelength = self.get_edge_wirelength(edge_freq)
            node_a, node_b = edge
            pin_a, pin_b = pins
            pos_a, pos_b = self.db.c_graph_pos_map[node_a], self.db.c_graph_pos_map[node_b]

            if self.partition:
                num_poly = math.ceil(edge_wirelength*self.padding_size/(self.partition_size**2))
                # num_poly = math.ceil(edge_wirelength*self.padding_size/(self.partition_size**2)) - 8    # for test only
                wireblk_size = (self.partition_size, self.partition_size)
                if self.debugging:
                    logging.debug(f'partition: {self.partition}, num_poly:{num_poly}')
                    logging.debug(f'edge_wl : {edge_wirelength:.2f}, wireblk_size: {wireblk_size}')
                assert num_poly != 0
                interm_pos = [(pos_a[0]+(pos_b[0]-pos_a[0]) * (i+1) / (num_poly+1),
                            pos_a[1]+(pos_b[1]-pos_a[1]) * (i+1) / (num_poly+1))
                            for i in range(num_poly)]
                last_node = node_a
                last_pin = pin_a
                wireblk_def_data[edge] = []
                for position in interm_pos:
                    poly_id = generate_polygon_id()
                    half_size = (wireblk_size[0]/2, wireblk_size[1]/2)
                    die_polygon = Polygon([
                        (position[0] - half_size[0], position[1] - half_size[1]),
                        (position[0] - half_size[0], position[1] + half_size[1]),
                        (position[0] + half_size[0], position[1] + half_size[1]),
                        (position[0] + half_size[0], position[1] - half_size[1]),
                    ])
                    wireblk_def_data[edge].append((poly_id, die_polygon))
                    poly_to_edge[poly_id] = edge
                    poly_to_freq_map[poly_id] = edge_freq
                    new_pin_distribution[(last_node, poly_id)] = (last_pin, 'IN')
                    last_node = poly_id
                    last_pin = 'OUT'
                new_pin_distribution[(last_node, node_b)] = (last_pin, pin_b)
                coordinates = [(0, 0), 
                               (0, wireblk_size[1]), 
                               (wireblk_size[0], wireblk_size[1]), 
                               (wireblk_size[0], 0)]
                if "wireblk" not in self.coresite_size_dict.keys():
                    self.coresite_size_dict["wireblk"] = self.partition_size
                if "wireblk" not in self.wireblk_polygon_map.keys():
                    self.wireblk_polygon_map["wireblk"] = Polygon(coordinates)

            else:
                wireblk_size = (round(self.partition_size*edge_wirelength/(self.qubit_size*2), 2), 2*(self.qubit_size*2))
                if self.debugging:
                    logging.debug(f'partition: {self.partition}, edge_wl : {edge_wirelength:.2f}, wireblk_size: {wireblk_size}')
                midpoint = ((pos_a[0] + pos_b[0]) / 2, (pos_a[1] + pos_b[1]) / 2)
                half_width, half_height = wireblk_size[0] / 2, wireblk_size[1] / 2
                edge_polygon = Polygon([
                    (midpoint[0] - half_width, midpoint[1] - half_height),
                    (midpoint[0] - half_width, midpoint[1] + half_height),
                    (midpoint[0] + half_width, midpoint[1] + half_height),
                    (midpoint[0] + half_width, midpoint[1] - half_height),
                ])
                poly_id = generate_polygon_id()
                wireblk_def_data[edge] = [(poly_id, edge_polygon)]
                poly_to_edge[poly_id] = edge
                poly_to_freq_map[poly_id] = edge_freq
                new_pin_distribution[(node_a, poly_id)] = (pin_a, "IN")
                new_pin_distribution[(poly_id, node_b)] = ("OUT", pin_b)
                
                coordinates = [(0, 0), 
                                (0, wireblk_size[1]), 
                                (wireblk_size[0], wireblk_size[1]), 
                                (wireblk_size[0], 0)]
                
                self.coresite_size_dict[edge] = 2*self.qubit_size
                self.wireblk_polygon_map[edge] = Polygon(coordinates)

        return wireblk_def_data, new_pin_distribution, poly_to_edge, poly_to_freq_map

