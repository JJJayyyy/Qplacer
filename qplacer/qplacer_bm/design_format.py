from qiskit_metal.qlibrary import QRoute, BaseQubit
from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket

from collections import defaultdict, OrderedDict
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import numpy as np
import logging
import random
import string
import uuid
import math
import os
# os.environ['QISKIT_METAL_HEADLESS'] = '1'

from qplacer_bm.utils import create_polygon


CONNETION_FORMAT = 'retangle'
# CONNETION_FORMAT = 'line'
# CONNETION_FORMAT = 'full'

LETTER_STRING = string.ascii_lowercase + string.digits

class DesignFormator:
    def __init__(self, 
                 params,
                 database,
                 ):
        self.db = database
        self.params = params
        self.coresite_size_dict = dict()
        self.wireblk_polygon_map = dict()
        self.poly_cnt = 0



    def __call__(self, debugging=False):
        self.debugging = debugging
        self.scale_factor = self.params.scale_factor
        self.partition = self.params.partition
        self.qubit_size = self.params.qubit_size
        self.partition_size = self.params.partition_size
        self.padding_size = self.params.padding_size

        self.db.metal_design.delete_all_components()

        self.db.qubit_to_metal_map = self.create_metal_qubits()
        self.db.qubit_geo_data, self.db.path_geo_data = self.get_geo_data_from_design()
        self.db.qubit_pin_distribution = self.distribute_pins_equally()
        (self.db.wireblk_def_data, 
         self.db.pin_distribution, 
         self.db.poly_to_edge, 
         self.db.poly_to_freq_map) = self.create_edges_def_data()
        self.db.wireblk_polygon_map = self.wireblk_polygon_map


    def create_metal_qubits(self):
        qubit_to_metal_map = dict()
        pad_size = round(self.params.qubit_pad_size * self.scale_factor)
        pocket_size = round(self.params.qubit_size * self.scale_factor) 
        for q_name, coordinates in self.db.c_graph_pos_map.items():
            x, y = coordinates
            q_options = dict(
                pos_x = f"{round(x * self.scale_factor)}um",
                pos_y = f"{round(y * self.scale_factor)}um",
                gds_cell_name = q_name, 
                pad_width = f'{pad_size} um',
                pocket_width = f'{pocket_size} um',
                pocket_height = f'{pocket_size} um',
                connection_pads = {
                    pin: dict(loc_W=loc_w, loc_H=loc_h, 
                              pad_width='80um', 
                              pad_gap='50um', 
                              cpw_extend='0um') for pin, (loc_w, loc_h) in self.params.pin_to_loc.items()
                },
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


    def get_geo_data_from_design(self):
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


    def create_pos_matrix(self, p1, p2, width, rect_width, num_rect_w, num_poly):
        """ 
            Calculate corners of the rectangle
                    corner0 __ __  corner3
            rect_w         |  .  |
                    corner1|__ __| corner2
                            rect_l
            return the matrix of point positions
        """

        def interpolate_points(start, end, num_points):
            return [(start[0] + i * (end[0] - start[0]) / (num_points - 1), 
                     start[1] + i * (end[1] - start[1]) / (num_points - 1)) for i in range(num_points)]
        
        assert num_rect_w > 0 and num_poly > 0
        tmp_x1, tmp_y1 = p1
        tmp_x2, tmp_y2 = p2
        if tmp_y1 > tmp_y2 or tmp_x1 > tmp_x2:
            x1, y1 = p2
            x2, y2 = p1
        else:
            x1, y1 = p1
            x2, y2 = p2
        rect_l = math.ceil(num_poly / num_rect_w)
        matrix = np.empty((rect_l, num_rect_w), dtype=object)

        if rect_l > 1:
            angle = math.atan2(y2 - y1, x2 - x1)
            offset_x = math.sin(angle) * width/2
            offset_y = math.cos(angle) * width/2
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) - rect_width

            corner0 = (x1 - offset_x, y1 + offset_y)
            corner1 = (x1 + offset_x, y1 - offset_y) 
            corner2 = (corner1[0]+length*math.cos(angle), corner1[1]+length*math.sin(angle))
            corner3 = (corner0[0]+length*math.cos(angle), corner0[1]+length*math.sin(angle))
            rect_corners = [(round(x, 2), round(y, 2)) for x, y in [corner0, corner1, corner2, corner3]]
            if tmp_y1 > tmp_y2 or tmp_x1 > tmp_x2:
                top_edge = interpolate_points(rect_corners[0], rect_corners[1], num_rect_w+1)
                bottom_edge = interpolate_points(rect_corners[3], rect_corners[2], num_rect_w+1)
                left_edge = interpolate_points(top_edge[0], bottom_edge[0], rect_l)
                right_edge = interpolate_points(top_edge[-2], bottom_edge[-2], rect_l)
            else:
                top_edge = interpolate_points(rect_corners[0], rect_corners[1], num_rect_w+1)
                bottom_edge = interpolate_points(rect_corners[3], rect_corners[2], num_rect_w+1)
                left_edge = interpolate_points(top_edge[1], bottom_edge[1], rect_l)
                right_edge = interpolate_points(top_edge[-1], bottom_edge[-1], rect_l)

            for i in range(rect_l):
                matrix[i] = interpolate_points(left_edge[i], right_edge[i], num_rect_w)
        else:
            interm_pos = [(x1+(x2-x1) * (i+1) / (num_poly+1), 
                        y1+(y2-y1) * (i+1) / (num_poly+1)) for i in range(num_poly)]
            matrix[0] = interm_pos

        # create connections_map
        connections_map = np.empty((rect_l, num_rect_w), dtype=object)
        for i in range(rect_l):
            if i%2 == 0:
                for j in range(num_rect_w):
                    connections_map[i][j] = []
                    if j < num_rect_w - 1:
                        connections_map[i][j].append((i, j+1))
                    if i < rect_l - 1:
                        connections_map[i][j].append((i+1, j))
            else:
                for j in range(num_rect_w-1, -1, -1):
                    connections_map[i][j] = []
                    if j > 0:
                        connections_map[i][j].append((i, j-1))
                    if i < rect_l - 1:
                        connections_map[i][j].append((i+1, j))
        
        # eliminate unused points
        idle_pts = rect_l * num_rect_w - num_poly
        first_row_idle = math.ceil(idle_pts/2)
        last_row_idle = idle_pts - first_row_idle

        first_node_idx = first_row_idle
        last_node_idx = last_row_idle if (rect_l - 1) % 2 else num_rect_w - 1 - last_row_idle
        
        matrix[0, :first_node_idx] = None
        connections_map[0, :first_node_idx] = None
        if (rect_l - 1) % 2 == 0:
            matrix[rect_l-1, last_node_idx+1:] = None
            connections_map[rect_l-1, last_node_idx+1:] = None
        else:
            matrix[rect_l-1, :last_node_idx] = None
            connections_map[rect_l-1, :last_node_idx] = None

        m_nan_cnt = np.count_nonzero(matrix == None)
        cm_nan_cnt = np.count_nonzero(connections_map == None)
        # logging.debug(f'matrix: {matrix}')
        # logging.debug(f'connections_map: {connections_map}')
        # logging.debug(f'first_node_idx: {first_node_idx}, last_node_idx: {last_node_idx}')
        assert m_nan_cnt == idle_pts and cm_nan_cnt == idle_pts, \
            "m_nan_cnt:{},cm_nan_cnt:{}, idle_pts:{}, rect_l:{}, rect_w:{}, num_poly:{}\nmatrix: {}\nconnections_map: {}".format(
                m_nan_cnt, cm_nan_cnt, idle_pts, 
                rect_l, num_rect_w, num_poly, 
                matrix, connections_map)

        if self.params.graph_debugging:
            # Visualization
            fig, ax = plt.subplots()
            if rect_l > 1:
                logging.info(f'rect_corners: {rect_corners}')
                # Plot the corners
                for corner in rect_corners:
                    ax.plot(corner[0], corner[1], 'ro')  
                # Plotting the rectangle edges
                rect_edges = np.array(rect_corners + [rect_corners[0]])
                ax.plot(rect_edges[:,0], rect_edges[:,1], 'r-')

            for row in matrix:
                for point in row:
                    if point != None:
                        ax.plot(point[0], point[1], 'bx')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.show()
        
        return matrix, connections_map, first_node_idx, last_node_idx
            
   
    def generate_unique_polygon_id(self, size=3):
        # return f"poly_{uuid.uuid4().hex[:size+1]}"       # increase digits if needed
        tmp_num = self.poly_cnt
        if tmp_num < 0:
            raise ValueError("Number must be non-negative")
        else:
            result = ''
            while tmp_num:
                tmp_num, rem = divmod(tmp_num, len(LETTER_STRING))
                result = LETTER_STRING[rem] + result
            result = result.rjust(size, LETTER_STRING[0])
            self.poly_cnt += 1
            return 'poly_'+ result


    def create_edges_def_data(self):   
        
        def get_closest_point(pos1, pos2, qubit_size, wireblk_size):
            def normalize(vector):
                norm = np.linalg.norm(vector)
                return vector / norm if norm else vector

            poly1 = create_polygon((qubit_size, qubit_size), pos1)
            poly2 = create_polygon((qubit_size, qubit_size), pos2)
            pt1 = (poly1.centroid.x, poly1.centroid.y)
            pt2 = (poly2.centroid.x, poly2.centroid.y)
            direct_v1 = np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]])
            direct_v2 = np.array([pt1[0] - pt2[0], pt1[1] - pt2[1]])
            
            # modified_pt1 = pt1 + normalize(direct_v1) * np.sqrt(2)/2 * (wireblk_size+qubit_size)
            # modified_pt2 = pt2 + normalize(direct_v2) * np.sqrt(2)/2 * (wireblk_size+qubit_size)
            modified_pt1 = pt1 + normalize(direct_v1) * 1/2 * (wireblk_size+qubit_size)
            modified_pt2 = pt2 + normalize(direct_v2) * 1/2 * (wireblk_size+qubit_size)
            return modified_pt1, modified_pt2

        new_pin_distribution = defaultdict(list)
        wireblk_def_data = dict()
        poly_to_edge = dict()
        poly_to_freq_map = dict()
        logging.info(f'CONNETION_FORMAT: {CONNETION_FORMAT}')
        for j, (edge, pins) in enumerate(self.db.qubit_pin_distribution.items()):
            edge_freq = self.db.edge_to_freq_map[edge]
            edge_wirelength = self.get_edge_wirelength(edge_freq)
            node_a, node_b = edge
            pin_a, pin_b = pins
            pos_a, pos_b = self.db.c_graph_pos_map[node_a], self.db.c_graph_pos_map[node_b]

            if self.partition:
                num_poly = math.ceil(edge_wirelength*self.padding_size/(self.partition_size**2))
                wireblk_size = (self.partition_size, self.partition_size)
                assert num_poly != 0
                num_pt_width = math.floor(self.qubit_size / self.partition_size)

                m_pos_a, m_pos_b = get_closest_point(pos_a, pos_b, self.qubit_size, self.partition_size)
                if CONNETION_FORMAT == 'retangle':
                    pos_matrix, connection_map, first_node_idx, last_node_idx = self.create_pos_matrix(
                        m_pos_a, m_pos_b,
                        self.qubit_size, 
                        self.partition_size,
                        num_pt_width, num_poly)
                elif CONNETION_FORMAT == 'line':
                    pos_matrix, connection_map, first_node_idx, last_node_idx = self.create_pos_matrix(
                        m_pos_a, m_pos_b,
                        self.qubit_size, 
                        self.partition_size,
                        num_poly, num_poly)
                else:
                    raise Exception(f"CONNETION_FORMAT: {CONNETION_FORMAT} is not acceptable for now")

                logging.debug('pos_a: {}, pos_b: {}, partition: {}, num_poly: {}, edge_wl: {:.2f}, wireblk_size: {}'.format(
                    pos_a, pos_b, self.partition, num_poly, edge_wirelength, wireblk_size))

                wireblk_def_data[edge] = []
                poly_id_map = np.empty(pos_matrix.shape, dtype=object)
                for r_idx, row in enumerate(pos_matrix):
                    for c_idx, point in enumerate(row):
                        if point != None:
                            poly_id = self.generate_unique_polygon_id()
                            poly_id_map[r_idx][c_idx] = poly_id
                            position = pos_matrix[r_idx][c_idx]                            
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
                        else:
                            poly_id_map[r_idx][c_idx] = None

                first_node = poly_id_map[0][first_node_idx]
                new_pin_distribution[(node_a, first_node)] = (pin_a, 'IN')
                for r_idx, row in enumerate(connection_map):
                    for c_idx, connection in enumerate(row):
                        if connection != None:
                            cur_node_id = poly_id_map[r_idx][c_idx]
                            for nr_idx, nc_idx in connection:
                                next_node_id = poly_id_map[nr_idx][nc_idx]
                                if cur_node_id != None and next_node_id != None:
                                    new_pin_distribution[(cur_node_id, next_node_id)] = ('OUT', 'IN')

                last_node = poly_id_map[-1][last_node_idx]
                new_pin_distribution[(last_node, node_b)] = ('OUT', pin_b)

                if "wireblk" not in self.coresite_size_dict.keys():
                    self.coresite_size_dict["wireblk"] = self.partition_size
                if "wireblk" not in self.wireblk_polygon_map.keys():
                    self.wireblk_polygon_map["wireblk"] = Polygon(
                        [(0, 0), (0, wireblk_size[1]), 
                         (wireblk_size[0], wireblk_size[1]), (wireblk_size[0], 0)])
                    
                # print(f'poly_id_map : {poly_id_map}')
                # print(f'new_pin_distribution : {new_pin_distribution}')

                # assert 0
                # interm_pos = [(pos_a[0]+(pos_b[0]-pos_a[0]) * (i+1) / (num_poly+1),
                #             pos_a[1]+(pos_b[1]-pos_a[1]) * (i+1) / (num_poly+1))
                #             for i in range(num_poly)]
                # last_node = node_a
                # last_pin = pin_a
                # wireblk_def_data[edge] = []
                # for position in interm_pos:
                #     poly_id = generate_polygon_id()
                #     half_size = (wireblk_size[0]/2, wireblk_size[1]/2)
                #     die_polygon = Polygon([
                #         (position[0] - half_size[0], position[1] - half_size[1]),
                #         (position[0] - half_size[0], position[1] + half_size[1]),
                #         (position[0] + half_size[0], position[1] + half_size[1]),
                #         (position[0] + half_size[0], position[1] - half_size[1]),
                #     ])
                #     wireblk_def_data[edge].append((poly_id, die_polygon))
                #     poly_to_edge[poly_id] = edge
                #     poly_to_freq_map[poly_id] = edge_freq
                #     new_pin_distribution[(last_node, poly_id)] = (last_pin, 'IN')
                #     last_node = poly_id
                #     last_pin = 'OUT'
                # new_pin_distribution[(last_node, node_b)] = (last_pin, pin_b)
                # coordinates = [(0, 0), 
                #                (0, wireblk_size[1]), 
                #                (wireblk_size[0], wireblk_size[1]), 
                #                (wireblk_size[0], 0)]
                # if "wireblk" not in self.coresite_size_dict.keys():
                #     self.coresite_size_dict["wireblk"] = self.partition_size
                # if "wireblk" not in self.wireblk_polygon_map.keys():
                #     self.wireblk_polygon_map["wireblk"] = Polygon(coordinates)

            else:
                wireblk_size = (round(self.partition_size*edge_wirelength/(self.qubit_size*2), 2), 2*(self.qubit_size*2))
                logging.debug(f'partition: {self.partition}, edge_wl : {edge_wirelength:.2f}, wireblk_size: {wireblk_size}')
                midpoint = ((pos_a[0] + pos_b[0]) / 2, (pos_a[1] + pos_b[1]) / 2)
                half_width, half_height = wireblk_size[0] / 2, wireblk_size[1] / 2
                edge_polygon = Polygon([
                    (midpoint[0] - half_width, midpoint[1] - half_height),
                    (midpoint[0] - half_width, midpoint[1] + half_height),
                    (midpoint[0] + half_width, midpoint[1] + half_height),
                    (midpoint[0] + half_width, midpoint[1] - half_height),
                ])
                poly_id = self.generate_unique_polygon_id()
                wireblk_def_data[edge] = [(poly_id, edge_polygon)]
                poly_to_edge[poly_id] = edge
                poly_to_freq_map[poly_id] = edge_freq
                new_pin_distribution[(node_a, poly_id)] = (pin_a, "IN")
                new_pin_distribution[(poly_id, node_b)] = ("OUT", pin_b)
                self.coresite_size_dict[edge] = 2*self.qubit_size
                self.wireblk_polygon_map[edge] = Polygon(
                    [(0, 0), (0, wireblk_size[1]), 
                     (wireblk_size[0], wireblk_size[1]), (wireblk_size[0], 0)])
                
        return wireblk_def_data, new_pin_distribution, poly_to_edge, poly_to_freq_map

