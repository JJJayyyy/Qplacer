import sys
import os
import re
import math
import time
import numpy as np
import torch
import logging
import Params
import operators.dreamplace.ops.place_io.place_io as place_io


datatypes = {
        'float32' : np.float32,
        'float64' : np.float64
        }

class PlaceDB (object):
    """
    @brief placement database
    """
    def __init__(self):
        """
        initialization
        To avoid the usage of list, I flatten everything.
        """
        self.rawdb = None # raw placement database, a C++ object
        self.pydb = None # python placement database interface

        self.num_physical_nodes = 0 # number of real nodes, including movable nodes, terminals, and terminal_NIs
        self.num_terminals = 0 # number of terminals, essentially fixed macros
        self.num_terminal_NIs = 0 # number of terminal_NIs that can be overlapped, essentially IO pins
        self.node_name2id_map = {} # node name to id map, cell name
        self.node_names = None # 1D array, cell name
        self.node_x = None # 1D array, cell position x
        self.node_y = None # 1D array, cell position y
        self.node_orient = None # 1D array, cell orientation
        self.node_size_x = None # 1D array, cell width
        self.node_size_y = None # 1D array, cell height

        self.node2orig_node_map = None # some fixed cells may have non-rectangular shapes; we flatten them and create new nodes
                                        # this map maps the current multiple node ids into the original one

        self.pin_direct = None # 1D array, pin direction IO
        self.pin_offset_x = None # 1D array, pin offset x to its node
        self.pin_offset_y = None # 1D array, pin offset y to its node

        self.net_name2id_map = {} # net name to id map
        self.net_names = None # net name
        self.net_weights = None # weights for each net

        self.net2pin_map = None # array of 1D array, each row stores pin id
        self.flat_net2pin_map = None # flatten version of net2pin_map
        self.flat_net2pin_start_map = None # starting index of each net in flat_net2pin_map

        self.node2pin_map = None # array of 1D array, contains pin id of each node
        self.flat_node2pin_map = None # flatten version of node2pin_map
        self.flat_node2pin_start_map = None # starting index of each node in flat_node2pin_map

        self.pin2node_map = None # 1D array, contain parent node id of each pin
        self.pin2net_map = None # 1D array, contain parent net id of each pin

        self.rows = None # NumRows x 4 array, stores xl, yl, xh, yh of each row

        self.xl = None
        self.yl = None
        self.xh = None
        self.yh = None

        self.row_height = None
        self.site_width = None

        self.bin_size_x = None
        self.bin_size_y = None
        self.num_bins_x = None
        self.num_bins_y = None

        self.num_movable_pins = None

        self.total_movable_node_area = None # total movable cell area
        self.total_fixed_node_area = None # total fixed cell area
        self.total_space_area = None # total placeable space area excluding fixed cells

        # enable filler cells
        # the Idea from e-place and RePlace
        self.total_filler_node_area = None
        self.num_filler_nodes = None
        self.dtype = None

    def scale_pl(self, shift_factor, scale_factor):
        """
        @brief scale placement solution only
        @param shift_factor shift factor to make the origin of the layout to (0, 0)
        @param scale_factor scale factor
        """
        self.node_x -= shift_factor[0]
        self.node_x *= scale_factor
        self.node_y -= shift_factor[1]
        self.node_y *= scale_factor

    def unscale_pl(self, shift_factor, scale_factor): 
        """
        @brief unscale placement solution only
        @param shift_factor shift factor to make the origin of the layout to (0, 0)
        @param scale_factor scale factor
        """
        unscale_factor = 1.0 / scale_factor
        if shift_factor[0] == 0 and shift_factor[1] == 0 and unscale_factor == 1.0:
            node_x = self.node_x
            node_y = self.node_y
        else:
            node_x = self.node_x * unscale_factor + shift_factor[0]
            node_y = self.node_y * unscale_factor + shift_factor[1]

        return node_x, node_y

    def scale(self, shift_factor, scale_factor):
        """
        @brief shift and scale coordinates
        @param shift_factor shift factor to make the origin of the layout to (0, 0)
        @param scale_factor scale factor
        """
        logging.info("shift coordinate system by (%g, %g), scale coordinate system by %g" 
                % (shift_factor[0], shift_factor[1], scale_factor))
        self.scale_pl(shift_factor, scale_factor)
        self.node_size_x *= scale_factor
        self.node_size_y *= scale_factor
        self.pin_offset_x *= scale_factor
        self.pin_offset_y *= scale_factor
        self.xl -= shift_factor[0]
        self.xl *= scale_factor
        self.yl -= shift_factor[1]
        self.yl *= scale_factor
        self.xh -= shift_factor[0]
        self.xh *= scale_factor
        self.yh -= shift_factor[1]
        self.yh *= scale_factor
        self.row_height *= scale_factor
        self.site_width *= scale_factor

        # shift factor for rectangle 
        box_shift_factor = np.array([shift_factor, shift_factor]).reshape(1, -1)
        self.rows -= box_shift_factor
        self.rows *= scale_factor
        self.total_space_area *= scale_factor * scale_factor # this is area

        if len(self.flat_region_boxes): 
            self.flat_region_boxes -= box_shift_factor
            self.flat_region_boxes *= scale_factor
        # may have performance issue

    def sort(self):
        """
        @brief Sort net by degree.
        Sort pin array such that pins belonging to the same net is abutting each other
        """
        logging.info("sort nets by degree and pins by net")

        # sort nets by degree
        net_degrees = np.array([len(pins) for pins in self.net2pin_map])
        net_order = net_degrees.argsort() # indexed by new net_id, content is old net_id
        self.net_names = self.net_names[net_order]
        self.net2pin_map = self.net2pin_map[net_order]
        for net_id, net_name in enumerate(self.net_names):
            self.net_name2id_map[net_name] = net_id
        for new_net_id in range(len(net_order)):
            for pin_id in self.net2pin_map[new_net_id]:
                self.pin2net_map[pin_id] = new_net_id
        ## check
        #for net_id in range(len(self.net2pin_map)):
        #    for j in range(len(self.net2pin_map[net_id])):
        #        assert self.pin2net_map[self.net2pin_map[net_id][j]] == net_id

        # sort pins such that pins belonging to the same net is abutting each other
        pin_order = self.pin2net_map.argsort() # indexed new pin_id, content is old pin_id
        self.pin2net_map = self.pin2net_map[pin_order]
        self.pin2node_map = self.pin2node_map[pin_order]
        self.pin_direct = self.pin_direct[pin_order]
        self.pin_offset_x = self.pin_offset_x[pin_order]
        self.pin_offset_y = self.pin_offset_y[pin_order]
        old2new_pin_id_map = np.zeros(len(pin_order), dtype=np.int32)
        for new_pin_id in range(len(pin_order)):
            old2new_pin_id_map[pin_order[new_pin_id]] = new_pin_id
        for i in range(len(self.net2pin_map)):
            for j in range(len(self.net2pin_map[i])):
                self.net2pin_map[i][j] = old2new_pin_id_map[self.net2pin_map[i][j]]
        for i in range(len(self.node2pin_map)):
            for j in range(len(self.node2pin_map[i])):
                self.node2pin_map[i][j] = old2new_pin_id_map[self.node2pin_map[i][j]]
        ## check
        #for net_id in range(len(self.net2pin_map)):
        #    for j in range(len(self.net2pin_map[net_id])):
        #        assert self.pin2net_map[self.net2pin_map[net_id][j]] == net_id
        #for node_id in range(len(self.node2pin_map)):
        #    for j in range(len(self.node2pin_map[node_id])):
        #        assert self.pin2node_map[self.node2pin_map[node_id][j]] == node_id

    @property
    def num_movable_nodes(self):
        """
        @return number of movable nodes
        """
        return self.num_physical_nodes - self.num_terminals - self.num_terminal_NIs

    @property
    def num_nodes(self):
        """
        @return number of movable nodes, terminals, terminal_NIs, and fillers
        """
        return self.num_physical_nodes + self.num_filler_nodes

    @property
    def num_nets(self):
        """
        @return number of nets
        """
        return len(self.net2pin_map)

    @property
    def num_pins(self):
        """
        @return number of pins
        """
        return len(self.pin2net_map)

    @property
    def width(self):
        """
        @return width of layout
        """
        return self.xh-self.xl

    @property
    def height(self):
        """
        @return height of layout
        """
        return self.yh-self.yl

    @property
    def area(self):
        """
        @return area of layout
        """
        return self.width*self.height

    def bin_xl(self, id_x):
        """
        @param id_x horizontal index
        @return bin xl
        """
        return self.xl+id_x*self.bin_size_x

    def bin_xh(self, id_x):
        """
        @param id_x horizontal index
        @return bin xh
        """
        return min(self.bin_xl(id_x)+self.bin_size_x, self.xh)

    def bin_yl(self, id_y):
        """
        @param id_y vertical index
        @return bin yl
        """
        return self.yl+id_y*self.bin_size_y

    def bin_yh(self, id_y):
        """
        @param id_y vertical index
        @return bin yh
        """
        return min(self.bin_yl(id_y)+self.bin_size_y, self.yh)

    def num_bins(self, l, h, bin_size):
        """
        @brief compute number of bins
        @param l lower bound
        @param h upper bound
        @param bin_size bin size
        @return number of bins
        """
        return int(np.ceil((h-l)/bin_size))

    def bin_centers(self, l, h, bin_size):
        """
        @brief compute bin centers
        @param l lower bound
        @param h upper bound
        @param bin_size bin size
        @return array of bin centers
        """
        num_bins = self.num_bins(l, h, bin_size)
        centers = np.zeros(num_bins, dtype=self.dtype)
        for id_x in range(num_bins):
            bin_l = l+id_x*bin_size
            bin_h = min(bin_l+bin_size, h)
            centers[id_x] = (bin_l+bin_h)/2
        return centers

    def net_hpwl(self, x, y, net_id):
        """
        @brief compute HPWL of a net
        @param x horizontal cell locations
        @param y vertical cell locations
        @return hpwl of a net
        """
        pins = self.net2pin_map[net_id]
        nodes = self.pin2node_map[pins]
        hpwl_x = np.amax(x[nodes]+self.pin_offset_x[pins]) - np.amin(x[nodes]+self.pin_offset_x[pins])
        hpwl_y = np.amax(y[nodes]+self.pin_offset_y[pins]) - np.amin(y[nodes]+self.pin_offset_y[pins])

        return (hpwl_x+hpwl_y)*self.net_weights[net_id]

    def hpwl(self, x, y):
        """
        @brief compute total HPWL
        @param x horizontal cell locations
        @param y vertical cell locations
        @return hpwl of all nets
        """
        wl = 0
        for net_id in range(len(self.net2pin_map)):
            wl += self.net_hpwl(x, y, net_id)
        return wl

    def sum_pin_weights(self, weights=None):
        """
        @brief sum pin weights inside a physical node
        @param weights the torch tensor to store node weight data
        """
        if weights is None:
            weights = torch.zeros(
                self.num_nodes,
                dtype=self.net_weights.dtype, device="cpu")
        self.pydb.sum_pin_weights(
            torch.tensor(self.net_weights),
            weights)
        return weights

    def overlap(self, xl1, yl1, xh1, yh1, xl2, yl2, xh2, yh2):
        """
        @brief compute overlap between two boxes
        @return overlap area between two rectangles
        """
        return max(min(xh1, xh2)-max(xl1, xl2), 0.0) * max(min(yh1, yh2)-max(yl1, yl2), 0.0)

    def density_map(self, x, y):
        """
        @brief this density map evaluates the overlap between cell and bins
        @param x horizontal cell locations
        @param y vertical cell locations
        @return density map
        """
        bin_index_xl = np.maximum(np.floor(x/self.bin_size_x).astype(np.int32), 0)
        bin_index_xh = np.minimum(np.ceil((x+self.node_size_x)/self.bin_size_x).astype(np.int32), self.num_bins_x-1)
        bin_index_yl = np.maximum(np.floor(y/self.bin_size_y).astype(np.int32), 0)
        bin_index_yh = np.minimum(np.ceil((y+self.node_size_y)/self.bin_size_y).astype(np.int32), self.num_bins_y-1)

        density_map = np.zeros([self.num_bins_x, self.num_bins_y])

        for node_id in range(self.num_physical_nodes):
            for ix in range(bin_index_xl[node_id], bin_index_xh[node_id]+1):
                for iy in range(bin_index_yl[node_id], bin_index_yh[node_id]+1):
                    density_map[ix, iy] += self.overlap(
                            self.bin_xl(ix), self.bin_yl(iy), self.bin_xh(ix), self.bin_yh(iy),
                            x[node_id], y[node_id], x[node_id]+self.node_size_x[node_id], y[node_id]+self.node_size_y[node_id]
                            )

        for ix in range(self.num_bins_x):
            for iy in range(self.num_bins_y):
                density_map[ix, iy] /= (self.bin_xh(ix)-self.bin_xl(ix))*(self.bin_yh(iy)-self.bin_yl(iy))

        return density_map

    def density_overflow(self, x, y, target_density):
        """
        @brief if density of a bin is larger than target_density, consider as overflow bin
        @param x horizontal cell locations
        @param y vertical cell locations
        @param target_density target density
        @return density overflow cost
        """
        density_map = self.density_map(x, y)
        return np.sum(np.square(np.maximum(density_map-target_density, 0.0)))

    def print_node(self, node_id):
        """
        @brief print node information
        @param node_id cell index
        """
        logging.debug("node %s(%d), size (%g, %g), pos (%g, %g)" % (self.node_names[node_id], node_id, self.node_size_x[node_id], self.node_size_y[node_id], self.node_x[node_id], self.node_y[node_id]))
        pins = "pins "
        for pin_id in self.node2pin_map[node_id]:
            pins += "%s(%s, %d) " % (self.node_names[self.pin2node_map[pin_id]], self.net_names[self.pin2net_map[pin_id]], pin_id)
        logging.debug(pins)

    def print_net(self, net_id):
        """
        @brief print net information
        @param net_id net index
        """
        logging.debug("net %s(%d)" % (self.net_names[net_id], net_id))
        pins = "pins "
        for pin_id in self.net2pin_map[net_id]:
            pins += "%s(%s, %d) " % (self.node_names[self.pin2node_map[pin_id]], self.net_names[self.pin2net_map[pin_id]], pin_id)
        logging.debug(pins)

    def print_row(self, row_id):
        """
        @brief print row information
        @param row_id row index
        """
        logging.debug("row %d %s" % (row_id, self.rows[row_id]))

    def read(self, params):
        """
        @brief read using c++
        @param params parameters
        """
        self.dtype = datatypes[params.dtype]
        self.rawdb = place_io.PlaceIOFunction.read(params)
        self.initialize_from_rawdb(params)

    def initialize_from_rawdb(self, params):
        """
        @brief initialize data members from raw database
        @param params parameters
        """
        pydb = place_io.PlaceIOFunction.pydb(self.rawdb)
        self.pydb = pydb
        self.device = torch.device("cuda" if params.gpu else "cpu")

        self.num_physical_nodes = pydb.num_nodes
        self.num_terminals = pydb.num_terminals
        self.num_terminal_NIs = pydb.num_terminal_NIs
        self.node_name2id_map = pydb.node_name2id_map
        self.node_names = np.array(pydb.node_names, dtype=np.string_)
        
        self.node_x = np.array(pydb.node_x, dtype=self.dtype)
        self.node_y = np.array(pydb.node_y, dtype=self.dtype)
        self.node_orient = np.array(pydb.node_orient, dtype=np.string_)
        self.node_size_x = np.array(pydb.node_size_x, dtype=self.dtype)
        self.node_size_y = np.array(pydb.node_size_y, dtype=self.dtype)
        self.node2orig_node_map = np.array(pydb.node2orig_node_map, dtype=np.int32)
        self.pin_direct = np.array(pydb.pin_direct, dtype=np.string_)
        self.pin_offset_x = np.array(pydb.pin_offset_x, dtype=self.dtype)
        self.pin_offset_y = np.array(pydb.pin_offset_y, dtype=self.dtype)
        self.pin_names = np.array(pydb.pin_names, dtype=np.string_)
        self.net_name2id_map = pydb.net_name2id_map
        self.pin_name2id_map = pydb.pin_name2id_map
        self.net_names = np.array(pydb.net_names, dtype=np.string_)
        self.net2pin_map = pydb.net2pin_map
        self.flat_net2pin_map = np.array(pydb.flat_net2pin_map, dtype=np.int32)
        self.flat_net2pin_start_map = np.array(pydb.flat_net2pin_start_map, dtype=np.int32)

        if params.frequency_assign:
            self.net_weights = np.array(params.net_weight_list, dtype=self.dtype)
            self.node_in_group = params.wireblk_in_group
        else:
            self.net_weights = np.array(pydb.net_weights, dtype=self.dtype)
            self.node_in_group = []
        # self.net_weight_deltas = np.array(pydb.net_weight_deltas, dtype=self.dtype)
        # self.net_criticality = np.array(pydb.net_criticality, dtype=self.dtype)
        # self.net_criticality_deltas = np.array(pydb.net_criticality_deltas, dtype=self.dtype)
        self.node2pin_map = pydb.node2pin_map
        self.flat_node2pin_map = np.array(pydb.flat_node2pin_map, dtype=np.int32)
        self.flat_node2pin_start_map = np.array(pydb.flat_node2pin_start_map, dtype=np.int32)
        self.pin2node_map = np.array(pydb.pin2node_map, dtype=np.int32)
        self.pin2net_map = np.array(pydb.pin2net_map, dtype=np.int32)
        self.rows = np.array(pydb.rows, dtype=self.dtype)
        self.flat_region_boxes = np.array(pydb.flat_region_boxes, dtype=self.dtype)
        self.flat_region_boxes_start = np.array(pydb.flat_region_boxes_start, dtype=np.int32)
        self.node2fence_region_map = np.array(pydb.node2fence_region_map, dtype=np.int32)

        self.xl = float(pydb.xl)
        self.yl = float(pydb.yl)
        self.xh = float(pydb.xh)
        self.yh = float(pydb.yh)
        self.row_height = float(pydb.row_height)
        self.site_width = float(pydb.site_width)
        self.num_movable_pins = pydb.num_movable_pins
        self.total_space_area = float(pydb.total_space_area)

        # convert node2pin_map to array of array
        for i in range(len(self.node2pin_map)):
            self.node2pin_map[i] = np.array(self.node2pin_map[i], dtype=np.int32)
        self.node2pin_map = np.array(self.node2pin_map, dtype=object)

        # convert net2pin_map to array of array
        for i in range(len(self.net2pin_map)):
            self.net2pin_map[i] = np.array(self.net2pin_map[i], dtype=np.int32)
        self.net2pin_map = np.array(self.net2pin_map, dtype=object)


    def initialize_num_bins(self, params):
        """
        @brief initialize number of bins with a heuristic method, which many not be optimal.
        The heuristic is adapted form RePlAce, 2x2 to 4096x4096. 
        """
        # derive bin dimensions by keeping the aspect ratio 
        # this bin setting is not for global placement, only for other steps 
        # global placement has its bin settings defined in global_place_stages
        if params.num_bins_x <= 1 or params.num_bins_y <= 1: 
            total_bin_area = self.area
            avg_movable_area = self.total_movable_node_area / self.num_movable_nodes
            ideal_bin_area = avg_movable_area / params.target_density
            ideal_num_bins = total_bin_area / ideal_bin_area
            if (ideal_num_bins < 4): # smallest number of bins 
                ideal_num_bins = 4
            aspect_ratio = (self.yh - self.yl) / (self.xh - self.xl)
            y_over_x = True 
            if aspect_ratio < 1: 
                aspect_ratio = 1.0 / aspect_ratio
                y_over_x = False 
            aspect_ratio = int(math.pow(2, round(math.log2(aspect_ratio))))
            num_bins_1d = 2 # min(num_bins_x, num_bins_y)
            while num_bins_1d <= 4096:
                found_num_bins = num_bins_1d * num_bins_1d * aspect_ratio
                if (found_num_bins > ideal_num_bins / 4 and found_num_bins <= ideal_num_bins): 
                    break 
                num_bins_1d *= 2
            if y_over_x:
                self.num_bins_x = num_bins_1d
                self.num_bins_y = num_bins_1d * aspect_ratio
            else:
                self.num_bins_x = num_bins_1d * aspect_ratio
                self.num_bins_y = num_bins_1d
            params.num_bins_x = self.num_bins_x
            params.num_bins_y = self.num_bins_y
        else:
            self.num_bins_x = params.num_bins_x
            self.num_bins_y = params.num_bins_y

    def __call__(self, params):
        """
        @brief top API to read placement files
        @param params parameters
        """
        tt = time.time()
        self.read(params)
        self.initialize(params)
        logging.info("reading benchmark takes %g seconds" % (time.time()-tt))


    def initialize(self, params):
        """
        @brief initialize data members after reading
        @param params parameters
        """
        # shift and scale
        params.shift_factor[0] = self.xl
        params.shift_factor[1] = self.yl
        logging.info("set shift_factor = (%g, %g), as original row bbox = (%g, %g, %g, %g)" 
                % (params.shift_factor[0], params.shift_factor[1], self.xl, self.yl, self.xh, self.yh))
        if params.scale_factor == 0.0 or self.site_width != 1.0:
            params.scale_factor = 1.0 / self.site_width
        logging.info("set scale_factor = %g, as site_width = %g" % (params.scale_factor, self.site_width))
        self.scale(params.shift_factor, params.scale_factor)

        content = """
================================= Benchmark Statistics =================================
#nodes = %d, #terminals = %d, # terminal_NIs = %d, #movable = %d, #nets = %d
die area = (%g, %g, %g, %g) %g
row height = %g, site width = %g
""" % (self.num_physical_nodes, self.num_terminals, self.num_terminal_NIs, 
       self.num_movable_nodes, len(self.net_names), self.xl, self.yl, self.xh, self.yh, 
       self.area, self.row_height, self.site_width)

        # set num_movable_pins
        if self.num_movable_pins is None:
            self.num_movable_pins = 0
            for node_id in self.pin2node_map:
                if node_id < self.num_movable_nodes:
                    self.num_movable_pins += 1
        content += "#pins = %d, #movable_pins = %d\n" % (self.num_pins, self.num_movable_pins)
        # set total cell area
        self.total_movable_node_area = float(np.sum(self.node_size_x[:self.num_movable_nodes]*self.node_size_y[:self.num_movable_nodes]))
        # total fixed node area should exclude the area outside the layout and the area of terminal_NIs
        self.total_fixed_node_area = float(np.sum(
                np.maximum(
                    np.minimum(self.node_x[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs] + self.node_size_x[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs], self.xh)
                    - np.maximum(self.node_x[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs], self.xl),
                    0.0) * np.maximum(
                        np.minimum(self.node_y[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs] + self.node_size_y[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs], self.yh)
                        - np.maximum(self.node_y[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs], self.yl),
                        0.0)
                ))
        content += "total_movable_node_area = %g, total_fixed_node_area = %g, total_space_area = %g\n" % (self.total_movable_node_area, self.total_fixed_node_area, self.total_space_area)

        target_density = min(self.total_movable_node_area / self.total_space_area, 1.0)
        if target_density > params.target_density:
            logging.warning("target_density %g is smaller than utilization %g, ignored" % (params.target_density, target_density))
            params.target_density = target_density
        content += "utilization = %g, target_density = %g\n" % (self.total_movable_node_area / self.total_space_area, params.target_density)

        if params.enable_fillers:
            # the way to compute this is still tricky; we need to consider place_io together on how to
            # summarize the area of fixed cells, which may overlap with each other.
            node_size_order = np.argsort(self.node_size_x[: self.num_movable_nodes])
            range_lb = int(self.num_movable_nodes*0.05)
            range_ub = int(self.num_movable_nodes*0.95)
            if range_lb >= range_ub: # when there are too few cells, i.e., <= 1
                filler_size_x = 0
            else:
                filler_size_x = np.mean(self.node_size_x[node_size_order[range_lb:range_ub]])
            filler_size_y = self.row_height
            placeable_area = max(self.area - self.total_fixed_node_area, self.total_space_area)
            content += "use placeable_area = %g to compute fillers\n" % (placeable_area)
            self.total_filler_node_area = max(placeable_area * params.target_density - self.total_movable_node_area, 0.0)
            filler_area = filler_size_x * filler_size_y
            if filler_area == 0: 
                self.num_filler_nodes = 0
            else:
                self.num_filler_nodes = int(round(self.total_filler_node_area / filler_area))
                self.node_size_x = np.concatenate([self.node_size_x, np.full(self.num_filler_nodes, 
                                                                             fill_value=filler_size_x, 
                                                                             dtype=self.node_size_x.dtype)])
                self.node_size_y = np.concatenate([self.node_size_y, np.full(self.num_filler_nodes, 
                                                                             fill_value=filler_size_y, 
                                                                             dtype=self.node_size_y.dtype)])
            content += "total_filler_node_area = %g, #fillers = %d, filler sizes = %gx%g\n" % (
                self.total_filler_node_area, self.num_filler_nodes, filler_size_x, filler_size_y)
        else:
            self.total_filler_node_area = 0
            self.num_filler_nodes = 0
            filler_size_x, filler_size_y = 0, 0
            content += "total_filler_node_area = %g, #fillers = %d, filler sizes = %gx%g\n" % (
                self.total_filler_node_area, self.num_filler_nodes, filler_size_x, filler_size_y)

        # set number of bins 
        # derive bin dimensions by keeping the aspect ratio 
        self.initialize_num_bins(params)
        # set bin size 
        self.bin_size_x = (self.xh - self.xl) / self.num_bins_x
        self.bin_size_y = (self.yh - self.yl) / self.num_bins_y
        content += "num_bins = %dx%d, bin sizes = %gx%g\n" % (self.num_bins_x, self.num_bins_y, 
                                                              self.bin_size_x / self.row_height, 
                                                              self.bin_size_y / self.row_height)
        logging.info(content)


    def write(self, params, filename, sol_file_format=None):
        """
        @brief write placement solution
        @param filename output file name
        @param sol_file_format solution file format, DEF
        """
        tt = time.time()
        logging.info("writing to %s" % (filename))
        if sol_file_format is None:
            if filename.endswith(".def"):
                sol_file_format = place_io.SolutionFileFormat.DEF
        # unscale locations
        node_x, node_y = self.unscale_pl(params.shift_factor, params.scale_factor)
        place_io.PlaceIOFunction.write(self.rawdb, filename, sol_file_format, node_x, node_y)
        logging.info("write %s takes %.3f seconds" % (str(sol_file_format), time.time()-tt))

    def apply(self, params, node_x, node_y):
        """
        @brief apply placement solution and update database
        """
        # assign solution
        self.node_x[:self.num_movable_nodes] = node_x[:self.num_movable_nodes]
        self.node_y[:self.num_movable_nodes] = node_y[:self.num_movable_nodes]

        # unscale locations
        node_x, node_y = self.unscale_pl(params.shift_factor, params.scale_factor)

        # update raw database
        place_io.PlaceIOFunction.apply(self.rawdb, node_x, node_y)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("One input parameters in json format in required")

    params = Params.Params()
    params.load(sys.argv[sys.argv[1]])
    logging.info("parameters = %s" % (params))

    db = PlaceDB()
    db(params)