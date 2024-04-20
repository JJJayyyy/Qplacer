import logging
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.optimize import minimize
from itertools import combinations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module='networkx')


class FrequencyAssigner:
    def __init__(self, G, optimizer='SLSQP', seed=None):
        self.G = G
        self.opt = optimizer
        # adj_matrix_sparse = nx.adjacency_matrix(self.G)
        # adj_matrix_dense = adj_matrix_sparse.todense()

        adj_matrix_sparse = nx.adjacency_matrix(self.G).tocoo()
        if sp.issparse(adj_matrix_sparse):
            adj_matrix_dense = adj_matrix_sparse.todense()
        else:
            adj_matrix_dense = adj_matrix_sparse

        self.adjacency_matrix = np.array(adj_matrix_dense)
        self.optimized_q_freqs = []
        if seed != None:
            np.random.seed(seed)


    def assign_qubit_frequencies(self, q_freq_range=(4.8, 5.2), q_th=0.1, debugging=False):
        def q_freq_assign_obj(freqs):
            penalty = 0
            for i in range(len(freqs)):
                for j in range(i+1, len(freqs)):
                    if self.adjacency_matrix[i, j] == 1:
                        freq_diff = np.abs(freqs[i] - freqs[j])
                        if freq_diff < q_th:
                            penalty += 5 / (freq_diff + 1e-1)
            return penalty

        init_q_freqs = np.random.uniform(q_freq_range[0], q_freq_range[1], len(self.adjacency_matrix))
        q_freq_bounds = [(q_freq_range[0], q_freq_range[1]) for _ in range(len(self.adjacency_matrix))]
        q_result = minimize(q_freq_assign_obj, init_q_freqs, method=self.opt, bounds=q_freq_bounds)
        self.optimized_q_freqs = q_result.x
        # node_to_freq_map = {f'Q{idx+1}' : f'{round(f, 2)}Ghz' for idx, f in enumerate(self.optimized_q_freqs)}
        node_to_freq_map = {f'Q{idx}' : f'{round(f, 2)}Ghz' for idx, f in enumerate(self.optimized_q_freqs)}
        logging.debug(f"Optimized Qubit Frequencies: {node_to_freq_map}")
        return node_to_freq_map
        

    def assign_resonator_frequencies(self, res_freq_range=(6, 7), res_th=0.1, optimize=True, debugging=False):
        if len(self.optimized_q_freqs)==0 and optimize==True:
            raise Exception(f"Qubit frequency need to be assigned first, self.optimized_q_freqs = {self.optimized_q_freqs}")   
        else:
            edges = [(j, i) for i in range(len(self.adjacency_matrix)) 
                    for j in range(i) if self.adjacency_matrix[i, j] == 1]

            def res_freq_assign_obj(freqs):
                penalty = 0
                for (i, j), (k, l) in combinations(edges, 2):
                    freq_diff = np.abs(freqs[edges.index((i, j))] - freqs[edges.index((k, l))])
                    if freq_diff < res_th:
                        penalty += 5 / (freq_diff + 1e-1)
                # Additional penalty for resonator-qubit frequency differences
                if optimize:
                    for edge, freq in zip(edges, freqs):
                        for qubit in edge:
                            freq_diff = np.abs(freq - self.optimized_q_freqs[qubit])
                            penalty += freq_diff 
                return penalty

            num_edges = len(edges)
            init_res_freqs = np.random.uniform(res_freq_range[0], res_freq_range[1], num_edges)
            q_freq_bounds = [(res_freq_range[0], res_freq_range[1]) for _ in range(num_edges)]
            res_result = minimize(res_freq_assign_obj, init_res_freqs, method='SLSQP', bounds=q_freq_bounds)
            optimized_res_freqs = res_result.x
            # edge_to_freq_map = {(f'Q{edge[0]+1}', f'Q{edge[1]+1}'): f'{round(f, 2)}Ghz' for edge, f in zip(edges, optimized_res_freqs)}
            edge_to_freq_map = {(f'Q{edge[0]}', f'Q{edge[1]}'): f'{round(f, 2)}Ghz' for edge, f in zip(edges, optimized_res_freqs)}
            for edge, freq in edge_to_freq_map.items():
                logging.debug(f"Edge {edge}: Frequency {freq} GHz")

            return edge_to_freq_map