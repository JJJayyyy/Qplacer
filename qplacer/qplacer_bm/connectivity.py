import math
import copy
import logging
import numpy as np
import networkx as nx
from itertools import product



class ConnectivityGraphBuilder:
    def __init__(self, qubits, topology='grid', debugging=False):
        self.qubits = qubits
        self.topology = topology
        self.debugging = debugging
        

    def get_connectivity_graph(self, qubit_name='Q', scale=3):
        if self.topology == 'grid':
            side = int(np.sqrt(self.qubits))
            c_graph, node_pos = self.grid_2d_graph(side, side)
        elif self.topology == 'oxtagon':
            squares = self.qubits//8
            c_graph, node_pos = self.oxtagon_graph(cols=min(squares, 5), rows=max(squares//5, 1))
        elif self.topology == 'ibm_falcon':
            c_graph, node_pos = self.hexagonal_lattice_graph(1, 2, device=self.topology)
        elif self.topology == 'ibm_hummingbird':
            c_graph, node_pos = self.hexagonal_lattice_graph(4, 2, device=self.topology)
        elif self.topology == 'ibm_eagle':
            c_graph, node_pos = self.hexagonal_lattice_graph(6, 3, device=self.topology)
        elif self.topology == 'heavy_hexagonal':
            side = int(np.sqrt((self.qubits+2)/2))-1
            c_graph, node_pos = self.hexagonal_lattice_graph(max(1, side//2), side)
        elif self.topology == 'hexagonal':
            side = int(np.sqrt((self.qubits+2)/2))-1
            c_graph, node_pos = self.hexagonal_lattice_graph(side, side, heavy=False)
        elif self.topology == 'xtree':
            if self.qubits > 5:
                levels = math.ceil(((self.qubits-5)/4) ** (1/3))
            elif self.qubits == 5:
                levels = 1
            else:
                levels = 0
            c_graph, node_pos = self.xtree_graph(levels)
        else:
            logging.warning(f"Topology {self.topology} not supported; use grid graph instead.")
            side = int(np.sqrt(self.qubits))
            c_graph, node_pos = self.grid_2d_graph(side, side)

        node_pos = self.convert_to_grid(node_pos, self.topology)
        for node, p in node_pos.items():
            c_graph.nodes[node]['pos'] = p

        mapping = dict()
        qubit_pos_map = dict()
        for idx, node in enumerate(c_graph.nodes()):
            # new_name = f"{qubit_name}{idx+1}"
            new_name = f"{qubit_name}{idx}"
            mapping[node] = new_name
            qubit_pos_map[new_name] = (node_pos[node][0]*scale, node_pos[node][1]*scale)
        c_graph = nx.relabel_nodes(c_graph, mapping)
        assert c_graph.number_of_nodes() == self.qubits, f'# nodes : {c_graph.number_of_nodes()}, {self.qubits}'
        return c_graph, qubit_pos_map



    def grid_2d_graph(self, m, n):
        G = nx.Graph()
        node_labels = {idx: idx+1 for idx, (i, j) in enumerate(product(range(m), range(n)))}
        G.add_nodes_from(node_labels.values())
        pos_to_label = {(i, j): i * n + j + 1 for i, j in product(range(m), range(n))}

        for i in range(m):
            for j in range(n):
                node_label = pos_to_label[(i, j)]
                if j < n - 1:
                    G.add_edge(node_label, pos_to_label[(i, j + 1)])
                if i < m - 1:
                    G.add_edge(node_label, pos_to_label[(i + 1, j)])

        pos = {node_label: (j, -i) for (i, j), node_label in pos_to_label.items()}
        return G, pos



    def oxtagon_graph(self, cols=4, rows=2, compact=True):
        def insert_node_in_each_edge(graph):
            original_edges = list(graph.edges())
            new_node_id = max(graph.nodes(), default=-1) + 1 
            
            for edge in original_edges:
                graph.remove_edge(*edge)
                graph.add_edge(edge[0], new_node_id)
                graph.add_edge(new_node_id, edge[1])
                new_node_id += 1
            return graph

        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])  # Create a square
        """
            Q == Q
          //      \\
          Q        Q        Q - Q - Q - Q
         ||        ||   ->  |           |
          Q        Q        Q - Q - Q - Q
          \\      //
            Q == Q
        """
        if compact:
            sq_pos = {1: (0, 1), 2: (1, 0), 3: (3, 0), 4: (2, 1), 
                    5: (0, 0), 6: (1, 1), 7: (2, 0), 8: (3, 1)}
        else:
            sq_pos = {1: (0, 2), 2: (1, 0), 3: (3, 1), 4: (2, 3), 
                        5: (0, 1), 6: (1, 3), 7: (2, 0), 8: (3, 2)}
        
        pos = copy.deepcopy(sq_pos)
        num_nodes_sq = len(pos)
        G = insert_node_in_each_edge(G)
        
        pos_shift = 4 
        for col in range(1, cols):
            G2 = nx.Graph()
            G2.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
            G2 = insert_node_in_each_edge(G2)
            G2 = nx.relabel_nodes(G2, lambda x: x + max(G.nodes()))
            new_positions = {max(G.nodes()) + node: (x+pos_shift*col, y) for node, (x, y) in sq_pos.items()}
            pos.update(new_positions)

            G = nx.union(G, G2)
            # print(len(pos)*col, len(pos)*col + 4)
            G.add_edge(num_nodes_sq*col  , num_nodes_sq*col+1)
            G.add_edge(num_nodes_sq*col-5, num_nodes_sq*col+5)

        # vertical 
        if rows > 1:
            if compact:
                pos_shift = 2
            for row in range(1, rows):
                new_positions = {max(G.nodes()) + node: (x, y-pos_shift*row) for node, (x, y) in pos.items()}
                pos.update(new_positions)

                G2 = copy.deepcopy(G)
                G2 = nx.relabel_nodes(G2, lambda x: x + max(G.nodes()))
                G = nx.union(G, G2)
                for col in range(cols):
                    G.add_edge(num_nodes_sq*col + 2, (row*cols+col)*num_nodes_sq + 6)
                    G.add_edge(num_nodes_sq*col + 7, (row*cols+col)*num_nodes_sq + 4)
        return G, pos

        

    def hexagonal_lattice_graph(self, m, n, device=None, heavy=True):
        def insert_node_in_each_edge(graph, pos):
            original_edges = list(graph.edges())
            new_node_id = max(graph.nodes()) + 1
            edge = original_edges[0]
            for edge in original_edges:
                midpoint = ((pos[edge[0]][0] + pos[edge[1]][0]) / 2, (pos[edge[0]][1] + pos[edge[1]][1]) / 2)
                graph.remove_edge(*edge)
                pos[new_node_id] = midpoint
                graph.add_edge(edge[0], new_node_id)
                graph.add_edge(new_node_id, edge[1])
                new_node_id += 1
            return graph, pos
        
        def device_specific_nodes(graph, device, pos, unit_len=0.5):
            new_node_id = max(graph.nodes()) + 1
            device_node_dict = {
                'ibm_falcon': {1:(-unit_len, 0), 2:(0, unit_len), 4:(0, unit_len), 
                            8:(0, -unit_len), 10:(0, -unit_len), 11:(unit_len, 0)},
                'ibm_hummingbird': {5:(unit_len, 0), 26:(-unit_len, 0)}, 
                'ibm_eagle': {50:(-unit_len, 0)}, 
            }
            device_remove_node_dict = {
                'ibm_eagle': [8], 
            }
            if device in device_node_dict.keys():
                for node, node_pos in device_node_dict[device].items(): 
                    node_pos = (pos[node][0]+node_pos[0], pos[node][1]+node_pos[1])
                    pos[new_node_id] = node_pos
                    graph.add_edge(node, new_node_id)
                    new_node_id += 1
            
            if device in device_remove_node_dict.keys():
                for remove_node in device_remove_node_dict[device]: 
                    graph.remove_node(remove_node)
                    pos.pop(remove_node)
            return graph, pos
        
        node_in_row = 2*(n+1)
        G, pos = self.grid_2d_graph(m+1, node_in_row)

        for row in range(m):
            for col in range((n+1)):
                if row % 2 == 0:
                    edge = (row*node_in_row + 2*(col+1), row*node_in_row + 2*(col+1+n+1))
                else:
                    edge = (row*node_in_row + 2*(col)+1, row*node_in_row + 2*(col+n+1)+1)
                G.remove_edge(*edge)
        if (n+1) % 2 == 1:    # first row end edge
            G.remove_node(node_in_row)  # edge = (node_in_row-1, node_in_row)
            pos.pop(node_in_row, None)
        if (m+1) % 2 == 0:     # last row end edge
            G.remove_node((m+1)*node_in_row) # edge = ((m+1)*node_in_row-1, (m+1)*node_in_row)
            pos.pop((m+1)*node_in_row, None)
        else:   # last row begin edge
            G.remove_node(m*node_in_row+1)  # edge = (m*node_in_row+1, m*node_in_row+2)
            pos.pop(m*node_in_row+1, None)
        if heavy:
            G, pos = insert_node_in_each_edge(G, pos)
        if device != None:
            G, pos = device_specific_nodes(G, device, pos)
        return G, pos



    def xtree_graph(self, levels):
        def add_branches(graph, pos, parent, level, max_level, parent_direction=0.0):
            if level >= max_level:
                return
            branch_angle = np.pi / 2 if level == 0 else np.pi/(4*level)
            # radius = max_level+2-level
            radius = 1
            for i in range(4 if level == 0 else 3):
                child = len(graph)
                graph.add_node(child)
                graph.add_edge(parent, child)
                if level == 0:
                    angle = i * branch_angle
                else:
                    angle = parent_direction - branch_angle + i * branch_angle
                pos[child] = (pos[parent][0] + np.cos(angle) * radius,
                            pos[parent][1] + np.sin(angle) * radius)
                
                # Recursive call to add branches to the child
                add_branches(graph, pos, child, level+1, max_level, angle)

        G = nx.Graph()
        G.add_node(0)
        pos = {0: (0, 0)}
        add_branches(G, pos, parent=0, level=0, max_level=levels)
        return G, pos



    def convert_to_grid(self, pos, topology='grid'):
        def bias_pos(pos):
            min_x = min(pos.values(), key=lambda x: x[0])[0]
            min_y = min(pos.values(), key=lambda x: x[1])[1]
            return {node: (x - min_x, y - min_y) for node, (x, y) in pos.items()}
        
        def scale_pos(pos, factor=2):
            return {node: (x * factor, y * factor) for node, (x, y) in pos.items()}
        
        if topology == 'ibm_falcon':
            pos[23] = (pos[12][0], pos[13][1])
            pos[24] = (pos[2][0], pos[13][1])
            pos[25] = (pos[15][0], pos[13][1])
            pos[26] = (pos[14][0], pos[13][1])
            pos[27] = (pos[4][0], pos[13][1])
            pos[28] = (pos[17][0], pos[13][1])
            pos = scale_pos(pos)
        elif topology in ['ibm_hummingbird', 'heavy_hexagonal', 'ibm_eagle']:
            pos = scale_pos(pos)
        pos = bias_pos(pos)
        return pos