from shapely.geometry import Polygon, Point
from shapely.affinity import translate
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from collections import defaultdict, Counter, deque
import matplotlib.pyplot as plt
from scipy.ndimage import label
import networkx as nx
import numpy as np
import itertools
import logging
import math
import copy
from utils import (parse_def_file, parse_lef_file, 
                   generate_qubit_polygons, generate_wireblk_polygons)



class FreqCollisionChecker:
    def __init__(self, 
                 def_file,
                 params, 
                 db,
                 color = ['#72A0C1', '#0072CE','#00539B','#003366']
                 ) -> None:
        
        self.params = params
        qf_range=params.q_freq_range
        rf_range=params.res_freq_range
        self.q_th = params.q_th
        self.r_th = params.res_th

        self.factor = self.params.scale_factor
        self.partition = params.partition
        self.def_file_path = def_file
        self.lef_file_path = params.file_paths["lef"]

        self.def_data = parse_def_file(def_file_path=self.def_file_path)
        self.lef_data = parse_lef_file(lef_file_path=self.lef_file_path, factor=self.factor)
        self.qubit_polygons = generate_qubit_polygons(lef_data=self.lef_data, def_data=self.def_data)
        self.wireblk_polygons = generate_wireblk_polygons(lef_data=self.lef_data, 
                                                          def_data=self.def_data, 
                                                          partition=self.partition)
    
        self.qubit_to_freq_map = db.qubit_to_freq_map
        self.edge_to_freq_map = db.edge_to_freq_map
        self.wireblk_to_freq_map = db.poly_to_freq_map
        self.wireblk_to_edge_map = db.poly_to_edge

        self.all_polygons = {**self.qubit_polygons, **self.wireblk_polygons}
        self.component_to_freq_map = {**self.qubit_to_freq_map, **self.wireblk_to_freq_map}

        self.freq_list = sorted(list(set(self.component_to_freq_map.values())))
        self.edge_freq_set = set(self.edge_to_freq_map.values())
        self.ef_for_optimization = [f'{round(i, 2)}Ghz' for i in np.arange(rf_range[0], rf_range[1], round(self.r_th/2, 2))]
        self.edge_freq_set.update(self.ef_for_optimization)
        self.qf_for_optimization = [f'{round(i, 2)}Ghz' for i in np.arange(qf_range[0], qf_range[1], round(self.q_th/2, 2))]
        self.qubit_freq_set = set(self.component_to_freq_map.values())-self.edge_freq_set
        self.qubit_freq_set.update(self.qf_for_optimization)

        self.colors = color
        self.qcm = mcolors.LinearSegmentedColormap.from_list('q_f_colormap', self.colors, N=len(self.qubit_freq_set))
        self.qubit_f_cm = {freq: self.qcm(i / (len(self.qubit_freq_set) - 1)) 
                           for i, freq in enumerate(sorted(list(self.qubit_freq_set)))}
        self.q_f_norm = mcolors.Normalize(vmin=qf_range[0], vmax=qf_range[1])

        grey_colors = ['#dddddd', '#888888', '#333333'] # light grey, medium grey, dark_grey
        self.rcm = mcolors.LinearSegmentedColormap.from_list('res_f_colormap', grey_colors, N=len(self.edge_freq_set))
        self.res_f_cm = {freq: self.rcm(i / (len(self.edge_freq_set) - 1)) 
                           for i, freq in enumerate(sorted(list(self.edge_freq_set)))}
        self.res_f_norm = mcolors.Normalize(vmin=rf_range[0], vmax=rf_range[1])
        self.edge_cm = mcolors.LinearSegmentedColormap.from_list('edge_colormap', self.colors, N=len(self.edge_to_freq_map.keys()))
        self.edge_color_map = {edge: self.edge_cm(i) for i, edge in enumerate(self.edge_to_freq_map.keys())}

        self.total_collision_length = 0
        self.font = 6
        self.font_color = 'black'
        self.collision_color = 'black'
        self.edge_neighbor_freq_length = None
        # org
        self.edges_to_poly_map = defaultdict(list)
        for poly_name in self.wireblk_polygons.keys():
            self.edges_to_poly_map[self.wireblk_to_edge_map[poly_name]].append(poly_name)
        # print(f'self.edges_to_poly_map : {self.edges_to_poly_map}')
        self.fix_edge_to_blk = {}
        self.n_polys_in_edge = len(next(iter(self.edges_to_poly_map.values())))

        self.wireblk_neighbor_map = defaultdict(set)
        for comp1, comp2 in itertools.combinations(self.wireblk_polygons, 2):
            if self.check_adjacency(comp1, comp2):
                self.wireblk_neighbor_map[comp1].add(comp2)
                self.wireblk_neighbor_map[comp2].add(comp1) 

        self.qubit_neighbor_map = defaultdict(set)
        for comp1, comp2 in itertools.combinations(self.qubit_polygons, 2):
            if self.check_adjacency(comp1, comp2):
                self.qubit_neighbor_map[comp1].add(comp2)
                self.qubit_neighbor_map[comp2].add(comp1)

        self.edge_neighbor_map = None
        self.has_overlap = False


    def convert_freq_to_float(self, freq_str):
        return float(freq_str.lower().replace('ghz', '').strip())

    def check_freq_difference(self, freq_str_1, freq_str_2):
        return abs(self.convert_freq_to_float(freq_str_1)- self.convert_freq_to_float(freq_str_2))
    
    def get_qubit_collision_boundry(self, qubit1, qubit2, th=0.1, verbal=False):
        poly1 = self.all_polygons[qubit1]
        poly2 = self.all_polygons[qubit2]
        if poly1.touches(poly2) or poly1.intersects(poly2):
            f1 = self.convert_freq_to_float(self.component_to_freq_map[qubit1])
            f2 = self.convert_freq_to_float(self.component_to_freq_map[qubit2])
            if abs(f1 - f2) < th:
                intersection = poly1.intersection(poly2)
                if intersection.geom_type == 'LineString':
                    if verbal:
                        print(f'{qubit1:<3} | {qubit2:<3} has frequency collision @ ({f1}, {f2})')
                    return intersection, False
                elif intersection.geom_type == 'Polygon':
                    return None, True
                elif intersection.geom_type == 'Point':
                    return None, False
                else:
                    raise Exception(f"Intersection type :{intersection.geom_type} is not accepted")
        return None, False

    def get_wireblk_collision_boundry(self, blk1, blk2, th=0.1, verbal=False):
        poly1 = self.all_polygons[blk1]
        poly2 = self.all_polygons[blk2]
        if (poly1.touches(poly2) or poly1.intersects(poly2)):
            intersection = poly1.intersection(poly2)
            if intersection.geom_type == 'LineString':
                f1 = self.convert_freq_to_float(self.component_to_freq_map[blk1])
                f2 = self.convert_freq_to_float(self.component_to_freq_map[blk2])
                if abs(f1 - f2) < th:
                    if verbal:
                        print(f'{blk1:<3} {poly1.centroid} | {blk2:<3} {poly1.centroid} collide @ ({f1}, {f2})')
                    return intersection, True, False
                else:
                    return intersection, False, False
            elif intersection.geom_type == 'Polygon':
                    return None, False, True
            elif intersection.geom_type == 'Point':
                return None, False, False
            else:
                raise Exception(f"Intersection type :{intersection.geom_type} is not accepted")
        return None, False, False

    def check_adjacency(self, comp1, comp2):
        poly1 = self.all_polygons[comp1]
        poly2 = self.all_polygons[comp2]
        if (poly1.touches(poly2) or poly1.intersects(poly2)):
            intersection = poly1.intersection(poly2)
            if intersection.geom_type == 'LineString' or intersection.geom_type == 'Polygon':
                return True
        return False


    def plot_collisions(self, collision_width=5, suffix="", verbal=False):
        self.total_collision_length = 0
        fig, ax = plt.subplots()
        # component vitualization
        for qubit_name, qubit_polygon in self.qubit_polygons.items():
            x, y = qubit_polygon.exterior.xy
            face_color = self.qubit_f_cm[self.component_to_freq_map[qubit_name]]
            ax.fill(x, y, facecolor=face_color, label=qubit_name)
            # centroid = qubit_polygon.centroid
            # ax.text(centroid.x, centroid.y, qubit_name, ha='center', va='center', 
            #         fontsize=self.font, color=self.font_color)

        for wireblk_name, wireblk_polygon in self.wireblk_polygons.items():
            x, y = wireblk_polygon.exterior.xy
            face_color = self.res_f_cm[self.component_to_freq_map[wireblk_name]]
            ax.fill(x, y, facecolor=face_color, label=wireblk_name)
            # centroid = wireblk_polygon.centroid
            # ax.text(centroid.x, centroid.y, wireblk_name, ha='center', 
            #   va='center', fontsize=8, color='black')

        # collision check
        qubit_collisions = []
        for qubit1, qubit2 in itertools.combinations(self.qubit_polygons, 2):
            collided_boundry, overlap = self.get_qubit_collision_boundry(qubit1, qubit2, verbal=verbal)
            if overlap: self.has_overlap = True 
            if collided_boundry:
                x, y = collided_boundry.xy
                self.total_collision_length += collided_boundry.length
                ax.plot(x, y, color=self.collision_color, linewidth=collision_width)
                qubit_collisions.append((qubit1, qubit2))

        edge_collisions = []
        edge_neighbor_freq_length = defaultdict(lambda: defaultdict(float))
        for comp1, comp2 in itertools.combinations(self.wireblk_polygons, 2):
            edge1 = self.wireblk_to_edge_map[comp1]
            edge2 = self.wireblk_to_edge_map[comp2]
            freq1 = self.component_to_freq_map[comp1]
            freq2 = self.component_to_freq_map[comp2]
            if edge1 != edge2:
                boundry, collision, overlap = self.get_wireblk_collision_boundry(comp1, comp2, verbal=verbal)
                if overlap: self.has_overlap = True 
                if boundry != None:
                    edge_neighbor_freq_length[edge1][freq2] += boundry.length
                    edge_neighbor_freq_length[edge2][freq1] += boundry.length
                    if collision:
                        x, y = boundry.xy
                        self.total_collision_length += boundry.length
                        ax.plot(x, y, color=self.collision_color, linewidth=collision_width)
                        edge_collisions.append((edge1, edge2))
        self.edge_neighbor_freq_length = edge_neighbor_freq_length

        # complexity so high only for qubit has same frequency as resonators
        # for comp1, comp2 in itertools.combinations(self.all_polygons, 2):
        #     if self.wireblk_edge_map[comp1] != self.wireblk_edge_map[comp2]:
        #         shared_edge = self.get_collision_boundry(comp1, comp2)
        #         if shared_edge:
        #             x, y = shared_edge.xy
        #             self.total_collision_length += shared_edge.length
        #             ax.plot(x, y, color=collision_color, linewidth=collision_width)
        
        flat_qubit_collisions = [item for pair in qubit_collisions for item in pair]
        flat_edge_collisions = [item for pairs in edge_collisions for pair in pairs for item in pair]
        qubit_collision_counts = Counter(flat_qubit_collisions)
        edge_collision_counts = Counter(flat_edge_collisions)
        count_sum = qubit_collision_counts + edge_collision_counts

        if verbal:
            print(f"qubit_collisions : {qubit_collisions}\nedge_collisions : {edge_collisions}")
            print(f'Caused by Qubit collision ({len(qubit_collision_counts)}): {qubit_collision_counts}')
            print(f'Caused by edge collision({len(edge_collision_counts)}): {edge_collision_counts}')
            print(f'Impacted Q ({len(count_sum)}/{len(self.qubit_polygons.keys())}): {count_sum}')
        
        # Create ScalarMappable objects for both the qubits and resonators
        q_sm = plt.cm.ScalarMappable(cmap=self.qcm, norm=self.q_f_norm)
        q_sm.set_array([])
        r_sm = plt.cm.ScalarMappable(cmap=self.rcm, norm=self.res_f_norm)
        r_sm.set_array([])
        cbar_q_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7]) 
        cbar_r_ax = fig.add_axes([1.05, 0.15, 0.03, 0.7])
        cbar_q = plt.colorbar(q_sm, cax=cbar_q_ax, fraction=0.046, pad=0.04)
        cbar_q.set_label('Qubit Frequencies', rotation=270, labelpad=15)
        cbar_r = fig.colorbar(r_sm, cax=cbar_r_ax)
        cbar_r.set_label('Resonator Frequencies', rotation=270, labelpad=15)

        # Adjust the color bar location if necessary
        cbar_q.ax.yaxis.set_label_position('left')
        cbar_r.ax.yaxis.set_label_position('left')

        ax.set_aspect('equal')
        # plt.xlabel('X-axis (microns)')
        # plt.ylabel('Y-axis (microns)')

        # legend_elements = [Patch(facecolor=self.qubit_f_cm[freq], label=freq) for freq in self.freq_list]
        # legend_elements.append(Patch(facecolor=self.collision_color, label='collision'))
        # plt.legend(handles=legend_elements, title="Frequencies", loc='upper left', bbox_to_anchor=(1, 1))
        plt.subplots_adjust(right=1.5)
        plt.tight_layout()
        fig_path = f'{self.params.debugging_dir}/{self.params.topology}_collision_check{suffix}.png'
        plt.savefig(fig_path)
        logging.info(f'save figure {fig_path}')
        return len(count_sum)


    def plot_layout(self, qubit_color="#ADD8E6", wire_color="#0000FF"):
        fig, ax = plt.subplots()
        # qubits
        for qubit_name, qubit_polygon in self.qubit_polygons.items():
            x, y = qubit_polygon.exterior.xy
            ax.fill(x, y, facecolor=qubit_color, label=qubit_name)
            centroid = qubit_polygon.centroid
            ax.text(centroid.x, centroid.y, qubit_name, ha='center', va='center', 
                    fontsize=self.font, color=self.font_color)
            print(qubit_name, centroid, qubit_polygon.exterior.xy)
        
        # wire
        for wireblk_name, wireblk_polygon in self.wireblk_polygons.items():
            x, y = wireblk_polygon.exterior.xy
            ax.fill(x, y, facecolor=wire_color, label=wireblk_name)
            centroid = wireblk_polygon.centroid
            print(wireblk_name, centroid, wireblk_polygon.exterior.xy)
            # ax.text(centroid.x, centroid.y, wireblk_name, ha='center', va='center', 
            # fontsize=self.font, color='black')

        ax.set_aspect('equal')
        plt.xlabel('X-axis (microns)')
        plt.ylabel('Y-axis (microns)')
        plt.title('QUBIT Placement Visualization')
        legend_elements = [Patch(facecolor=c, label=n) for n, c in zip(['qubit', 'res'], 
                                                                       [qubit_color, wire_color])]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.subplots_adjust(right=0.75)
        plt.tight_layout()
        fig_path = f'{self.params.debugging_dir}/{self.params.topology}_layout.png'
        plt.savefig(fig_path)
        logging.info(f'save figure {fig_path}')


    def plot_edge_layout(self, qubit_color="#ADD8E6", wire_color="#0000FF", edges=None):
        fig, ax = plt.subplots()
        # qubits
        for qubit_name, qubit_polygon in self.qubit_polygons.items():
            x, y = qubit_polygon.exterior.xy
            ax.fill(x, y, facecolor=qubit_color, label=qubit_name)
            centroid = qubit_polygon.centroid
            ax.text(centroid.x, centroid.y, qubit_name, ha='center', va='center', 
                    fontsize=self.font, color=self.font_color)
        # wire
        if edges:
            for edge_name, wireblk_list in edges.items():
                for wireblk_name in wireblk_list:
                    wireblk_polygon = self.wireblk_polygons[wireblk_name]
                    x, y = wireblk_polygon.exterior.xy
                    ax.fill(x, y, facecolor=wire_color, label=wireblk_name)
        else:
            for wireblk_name, wireblk_polygon in self.wireblk_polygons.items():
                x, y = wireblk_polygon.exterior.xy
                ax.fill(x, y, facecolor=wire_color, label=wireblk_name)
                # centroid = wireblk_polygon.centroid
                # ax.text(centroid.x, centroid.y, wireblk_name, ha='center', va='center', 
                # fontsize=self.font, color='black')
        
        ax.set_aspect('equal')
        plt.xlabel('X-axis (microns)')
        plt.ylabel('Y-axis (microns)')
        plt.title('edge Placement Visualization')
        legend_elements = [Patch(facecolor=c, label=n) for n, c in zip(['qubit', 'res'], 
                                                                       [qubit_color, wire_color])]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.subplots_adjust(right=0.75)
        plt.tight_layout()
        fig_path = f'{self.params.debugging_dir}/{self.params.topology}_layout.png'
        plt.savefig(fig_path)
        logging.info(f'save figure {fig_path}')


    def print_grid(self):
        assert isinstance(self.grid, np.ndarray)
        print("--------")
        print("Grid Representation:")
        for row in self.grid:
            print(' '.join(f'{element:3}' for element in row))
            # print('-' * len(array_str))
        print("--------")
    

    def plot_matrix_layout(self):
        """
             - matrix is from left top to right bottom. 
             - Poly/metal is from left bottom to right top
             - position representation: [y, x] => y-axe/row, x-axe/column

            the centroid of for wireblk is abbstracted to a point in matrix
             __ __
            |  .  |
            |__ __|

            Qubit is abbstracted as below
             __ __ __ __
            |  .  |  .  |
            |__ __|__ __|
            |  .  |  .  |
            |__ __|__ __|
            maxtrix[y][x] -> poly position 
                left bottom : (x*grid_size, y*grid_size)
                left upper  : (x*grid_size, (y+1)*grid_size)
                right bottom : ((x+1)*grid_size, y*grid_size)
                right upper  : ((x+1)*grid_size, (y+1)*grid_size)
        """

        self.offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        n_qubits = len(self.qubit_polygons.keys())
        idx_to_components, components_to_idx = dict(), dict()

        for i, key in enumerate(list(self.qubit_polygons.keys()) + list(self.edge_to_freq_map.keys())):
            idx = i+1   # 0 is for empty space, the component idx starts from 1
            idx_to_components[idx] = key
            components_to_idx[key] = idx

        grid_size = int(self.factor * self.params.partition_size)
        n_x_grids = math.ceil(self.params.substrate_area[0] / grid_size) + 1
        n_y_grids = math.ceil(self.params.substrate_area[1] / grid_size) + 1
        # poly position, grid
        self.grid = np.zeros((n_x_grids, n_y_grids), dtype=int)
        # print(type(self.grid))
        self.poly_2_pos = dict()
        self.pos_2_poly = np.full((n_x_grids, n_y_grids), '', dtype=object)

        for j in range(n_y_grids):
            for i in range(n_x_grids):
                cell = Point(i*grid_size+grid_size/2, j*grid_size+grid_size/2)
                for poly_name, poly in self.all_polygons.items():
                    if 'poly' in poly_name:
                        component_name = self.wireblk_to_edge_map[poly_name] # wireblk -> edge
                    else:
                        component_name = poly_name # qubit -> qubit
                    idx = components_to_idx[component_name]
                    if cell.intersects(poly):
                        assert self.grid[j][i] == 0
                        self.grid[j][i] = idx
                        self.poly_2_pos[poly_name] = {'row': j, 'col':i} # j: y-axe/row, i: x-axe/column
                        self.pos_2_poly[j][i] = poly_name

        print(grid_size, n_x_grids, n_y_grids)
        print(f'components_to_idx : {components_to_idx}')
        print(f'idx_to_components : {idx_to_components}')
        print(self.qubit_polygons['Q1'].centroid, self.qubit_polygons['Q1'].exterior)
        self.print_grid()

    
        def find_disjoint_sections(matrix):
            sections = {}
            for wire in np.unique(matrix):
                if wire <= n_qubits:
                    continue
                labeled, num_features = label(matrix == wire)
                # print(num_features)
                if num_features == 1:
                    continue
                sections[wire] = [np.argwhere(labeled == i + 1) for i in range(num_features)]
            return sections
        
        def find_largest_section(sections):
            move_minor, bridge, rest = {}, {}, {}
            for wire, sects in sections.items():
                num_wireblks = len(self.edges_to_poly_map[idx_to_components[wire]])
                largest_segments_idx = max(range(len(sects)), key=lambda x: len(sects[x]))
                if len(sects[largest_segments_idx])/num_wireblks > 0.5:
                    other_sections = [section for i, section in enumerate(sects) if i != largest_segments_idx]
                    # print(np.vstack(other_sections))
                    # print()
                    move_minor[wire] = {"max" : sects[largest_segments_idx], 
                                        "rest": np.vstack(other_sections)}
                print(f"Wire: {wire}, wireblk len: {num_wireblks}, largest_idx: {largest_segments_idx}")  # Debugging print
                # print(f'Sections: {sects}')
                # for i, sec in enumerate(sects):
                #     print(i, len(sec))
            return move_minor
        
        def find_adjacent_spaces(segment, least_num):
            # adjacent_spaces = set()
            # for y, x in segment:    # y-axe/row, x-axe/column
            #     for dx, dy in self.offsets:
            #         adj_x, adj_y = x + dx, y + dy
            #         if (0 <= adj_x < n_x_grids and 0 <= adj_y < n_y_grids) and self.grid[adj_y][adj_x] == 0:
            #             adjacent_spaces.add((adj_y, adj_x))     # y-axe/row, x-axe/column
            # return list(adjacent_spaces)

            segment = [tuple(row) for row in segment]
            visited = set(segment)
            ava_space = set()
            queue = deque(segment)
            while queue:
                y, x = queue.popleft()
                for dy, dx in self.offsets:
                    adj_y, adj_x = y + dy, x + dx
                    if (0 <= adj_x < n_x_grids and 0 <= adj_y < n_y_grids) \
                    and self.grid[adj_y][adj_x] == 0 and (adj_y, adj_x) not in visited:
                        visited.add((adj_y, adj_x))
                        ava_space.add((adj_y, adj_x))
                        queue.append((adj_y, adj_x))
                        if len(ava_space) >= least_num:
                            return list(ava_space)
            return []

        def count_touches(space, segment):
            count = 0
            for dx, dy in self.offsets:
                for seg_y, seg_x in segment:
                    if space[0]+dy == seg_y and space[1]+dx == seg_x:
                        count +=1
            return count
            # return sum((space[0]+dy, space[1]+dx) in segment for dx, dy in self.offsets)

        def move_to_available_space(rest_segments, largest_segment):
            available_spaces = find_adjacent_spaces(largest_segment, len(rest_segments))
            new_positions = []
            if len(available_spaces) > 0:
                available_spaces.sort(key=lambda space: count_touches(space, largest_segment), reverse=True)
                print(f'available_spaces: {available_spaces}')
                for i, space in enumerate(available_spaces):
                    if i < len(rest_segments):
                        new_positions.append(space)
            return new_positions, len(available_spaces) >= len(rest_segments)
    
        def change_pos(origin_pos, latest_pos):
            # print(origin_pos, latest_pos)
            assert len(origin_pos) == len(latest_pos)
            for (row, col), (new_row, new_col) in zip(origin_pos, latest_pos):
                # find poly
                poly_name = self.pos_2_poly[row][col]
                idx = self.grid[row][col]
                # print(row, col, '/', new_row, new_col, poly_name, idx)

                assert poly_name != '', f'poly_name <{poly_name}>'
                # clear origin position
                self.grid[row][col] = 0
                self.pos_2_poly[row][col] = ''
                # assign new position
                self.grid[new_row][new_col] = idx
                self.pos_2_poly[new_row][new_col] = poly_name
                self.poly_2_pos[poly_name]['row'] = new_row
                self.poly_2_pos[poly_name]['col'] = new_col

                # also change poly
                # print(self.all_polygons[poly_name].centroid, row, col, new_row, new_col)
                # print(Point(col*grid_size+grid_size/2, row*grid_size+grid_size/2))
                # print(Point(new_col*grid_size+grid_size/2, new_row*grid_size+grid_size/2))
                delta_x = (new_col*grid_size+grid_size/2) - (col*grid_size+grid_size/2)
                delta_y = (new_row*grid_size+grid_size/2) - (row*grid_size+grid_size/2)
                new_polygon = translate(self.all_polygons[poly_name], xoff=delta_x, yoff=delta_y)
                self.all_polygons[poly_name] = new_polygon
                self.wireblk_polygons[poly_name] = new_polygon
                # print(self.all_polygons[poly_name].centroid, self.wireblk_polygons[poly_name].centroid)

        def remove_pos(origin_pos, wire):
            for (row, col) in origin_pos:
                poly_name = self.pos_2_poly[row][col]
                # print(row, col, '/', poly_name)

                assert poly_name != '', f'poly_name <{poly_name}>'
                # clear origin position
                self.grid[row][col] = 0
                self.pos_2_poly[row][col] = ''
                _ = self.poly_2_pos.pop(poly_name, None)

                # also change poly
                self.all_polygons.pop(poly_name, None)
                self.wireblk_polygons.pop(poly_name, None)
                edge_name = idx_to_components[wire]
                self.edges_to_poly_map[edge_name].remove(poly_name)

        
        disjoint_sections = find_disjoint_sections(self.grid)
        for edge, sections in disjoint_sections.items():
            sorted_disjoint_sections = sorted(sections, key=lambda x: len(x), reverse=True)
            disjoint_sections[edge] = sorted_disjoint_sections
        print(disjoint_sections.keys())
        # print(disjoint_sections)

        disjoint_sections_copy = copy.deepcopy(disjoint_sections)
        for wire, sections in disjoint_sections_copy.items():
            print(f"wire : {wire}")
            for i, cur_section in enumerate(sections):
                rest_of_sections = np.vstack(sections[:i] + sections[i+1:])
                new_positions, fittable = move_to_available_space(rest_of_sections, cur_section)
                print(f"{i}: Ava space is fittable: {fittable}")
                if fittable:
                    change_pos(rest_of_sections, new_positions)
                    removed_value = disjoint_sections.pop(wire, None)
                    if removed_value is None:
                        assert 0,  f"{wire} was not found in the dictionary"
                    break
        self.print_grid()

        print(disjoint_sections.keys())
        disjoint_sections_copy = copy.deepcopy(disjoint_sections)
        for wire, sections in disjoint_sections_copy.items():
            print(f"wire : {wire}")
            total_length = sum(len(lst) for lst in sections)
            accumulated_len = 0
            remove_flag = False
            for i, cur_section in enumerate(sections):
                if not remove_flag:
                    accumulated_len += len(cur_section)
                    if accumulated_len/total_length > 0.85:
                        remove_flag = True
                else:
                    print(f'remove {i}: {cur_section}')
                    remove_pos(cur_section, wire)
            if remove_flag:
                removed_value = disjoint_sections.pop(wire, None)
                if removed_value is None:
                    assert 0,  f"{wire} was not found in the dictionary"
        self.print_grid()

        # disjoint_sections = find_disjoint_sections(self.grid)
        # largest_sections = find_largest_section(disjoint_sections)
        # for wire, segments in largest_sections.items():
        #     print(f"wire : {wire}")
        #     new_positions, fittable = move_to_available_space(segments['rest'], segments['max'])
        #     print(f"Ava space is fittable: {fittable}, new positions: {new_positions}")
        #     if fittable:
        #         change_pos(segments['rest'], new_positions)
        #         removed_value = disjoint_sections.pop(wire, None)
        #         if removed_value is None:
        #             assert 0,  f"{wire} was not found in the dictionary"
        # print(disjoint_sections.keys())
        # self.print_grid()
        


    def check_integration(self, verbal=True):
        def create_touch_graph(polygons):
            G = nx.Graph()
            for i, poly_name in enumerate(polygons):
                G.add_node(poly_name)
            for (i, poly1_name), (j, poly2_name) in itertools.combinations(enumerate(polygons), 2):
                poly1 = self.all_polygons[poly1_name]
                poly2 = self.all_polygons[poly2_name]
                if poly1.touches(poly2):
                    G.add_edge(poly1_name, poly2_name)
            return G

        integration_cnt = 0
        edge_subgraph = defaultdict(list)
        for edge, polys_list in self.edges_to_poly_map.items():
            G = create_touch_graph(polys_list)
            if nx.is_connected(G):
                integration_cnt += 1
            else:
                connected_components = nx.connected_components(G)
                subgraphs = [G.subgraph(c).copy() for c in connected_components]
                # Print the subgraphs for this edge
                if verbal:
                    print(f"Edge {edge} - Subgraphs:")
                for i, sg in enumerate(subgraphs, 1):
                    node_list = list(sg.nodes())
                    if verbal:
                        print(f"  Subgraph {i}: Nodes ({len(node_list)}) - {node_list}")
                    edge_subgraph[edge].append(node_list)
        if verbal:
            print(f'{integration_cnt}/{len(self.edges_to_poly_map.keys())} edges are connected, issue edges: {list(edge_subgraph.keys())}')
        return edge_subgraph



    def get_total_area(self):
        return sum(polygon.area for polygon in self.all_polygons.values())


    def get_min_bounding_rect(self):
        print(f'Overlap : {self.has_overlap}')
        polygons_union = list(self.all_polygons.values())[0]
        for polygon in list(self.all_polygons.values())[1:]:
            polygons_union = polygons_union.union(polygon)
        return polygons_union.envelope


    def build_potential_freq_collisions_map(self, nodes_order, verbal=False):
        # find potential frequency collisions
        potential_freq_collisions = dict()
        potential_freq_collisions.update({idx: [] for idx in nodes_order.values()})

        qubit_list = list(self.qubit_to_freq_map.keys())
        for q1, q2 in itertools.combinations(qubit_list, 2):
            if self.check_freq_difference(self.qubit_to_freq_map[q1], self.qubit_to_freq_map[q2]) <= self.q_th:
                q1_idx, q2_idx = nodes_order[q1], nodes_order[q2]
                potential_freq_collisions[q1_idx].append(q2_idx)
                potential_freq_collisions[q2_idx].append(q1_idx)
                if verbal:
                    print(f'{q1}({self.qubit_to_freq_map[q1]}) - {q2}({self.qubit_to_freq_map[q2]}) < {self.q_th}')
                
        self.wireblk_to_idx_map = dict()
        for edge in self.edges_to_poly_map:
            self.wireblk_to_idx_map[edge] = [nodes_order.get(poly_name) for poly_name in self.edges_to_poly_map[edge]]
        
        edge_list = list(self.edge_to_freq_map.keys())
        for e1, e2 in itertools.combinations(edge_list, 2):
            if self.check_freq_difference(self.edge_to_freq_map[e1], self.edge_to_freq_map[e2]) <= self.r_th:
                e1_wireblks = self.wireblk_to_idx_map[e1]
                e2_wireblks = self.wireblk_to_idx_map[e2]
                for w1 in e1_wireblks:
                    potential_freq_collisions[w1].extend(e2_wireblks)
                for w2 in e2_wireblks:
                    potential_freq_collisions[w2].extend(e1_wireblks)
                if verbal:
                    txt_1 = self.edge_to_freq_map[e1]
                    txt_2 = self.edges_to_poly_map[e1]
                    txt_3 = self.edge_to_freq_map[e2]
                    txt_4 = self.edges_to_poly_map[e2]
                    print(f'{e1}: ({txt_1}) {txt_2} | {e2}: ({txt_3}) {txt_4} < {self.r_th}')
        
        # High complexity iterate every wireblks
        # wireblk_list = list(self.wireblk_to_freq_map.keys())        
        # for w1, w2 in itertools.combinations(wireblk_list, 2):
        #     if self.wireblk_to_edge_map[w1] != self.wireblk_to_edge_map[w2]:
        #         if self.check_freq_difference(self.wireblk_to_freq_map[w1], self.wireblk_to_freq_map[w2]) <= self.r_th:
        #             w1_idx, w2_idx = nodes_order[w1], nodes_order[w2]
        #             # w1_idx, w2_idx = w1, w2
        #             potential_freq_collisions[w1_idx].append(w2_idx)
        #             potential_freq_collisions[w2_idx].append(w1_idx)
        #             if verbal:
        #                 print(f'{w1}: ({self.wireblk_to_freq_map[w1]}) {self.wireblk_to_edge_map[w1]}
                    #  | {w2}: ({self.wireblk_to_freq_map[w2]}) {self.wireblk_to_edge_map[w2]} < {self.r_th}')
        return potential_freq_collisions
