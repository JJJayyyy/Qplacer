##
# @file   freq_electric_potential.py
# @author Junyao Zhang
# @date   Oct 2023
# @brief  electric potential according to particle repulse force
#

import time
import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F
import logging

import operators.qplacement.ops.frequency_repulsion.frequency_repulsion_cpp as frequency_repulsion_cpp


logger = logging.getLogger(__name__)


class FrequencyRepulsionFunction(Function):
    """
    @brief compute electric repulsion force
    """
    @staticmethod
    def forward(
        ctx,
        pos,
        node_size_x, 
        node_size_y,
        potential_collision_map,
        num_movable_nodes,
        num_filler_nodes,
        qubit_dist_threhold_x, 
        qubit_dist_threhold_y,
        coupler_dist_threhold_x, 
        coupler_dist_threhold_y, 
    ):
        tt = time.time()
        ctx.pos = pos
        ctx.potential_collision_map = potential_collision_map
        ctx.num_movable_nodes = num_movable_nodes
        ctx.num_filler_nodes = num_filler_nodes
        ctx.qubit_dist_threhold_x = qubit_dist_threhold_x
        ctx.qubit_dist_threhold_y = qubit_dist_threhold_y
        ctx.coupler_dist_threhold_x = coupler_dist_threhold_x
        ctx.coupler_dist_threhold_y = coupler_dist_threhold_y


        if pos.is_cuda:
            torch.cuda.synchronize()

        energy = torch.zeros_like(ctx.pos)
        epsilon = 1e-2
        # force_ratio = 0.01
        force_ratio = 0.5
        energy = frequency_repulsion_cpp.frequency_repulsion(
            ctx.pos,
            node_size_x,
            node_size_y,
            ctx.potential_collision_map,
            ctx.qubit_dist_threhold_x,
            ctx.qubit_dist_threhold_y,
            force_ratio,
            epsilon,
            ctx.num_movable_nodes,
        )

        abs_sum_energy = torch.sum(torch.abs(energy))
        ctx.energy = energy
        ctx.sum_energy = abs_sum_energy

        # print(f'repulsion energy {energy.shape}: {energy}')
        # print(f'repulsion energy abs sum: {abs_sum_energy}')
        logger.debug("frequency repulsion forward %.3f ms" % ((time.time()-tt)*1000))
        return abs_sum_energy


    @staticmethod
    def backward(ctx, grad_pos):
        tt = time.time()
        output = ctx.energy * grad_pos
        if grad_pos.is_cuda:
            torch.cuda.synchronize()
        logger.debug("frequency force backward %.3f ms" % ((time.time() - tt) * 1000))
        return output, None, None, None, None, None, None, None, None, None






class FrequencyRepulsion(nn.Module):
    """
    @brief Compute frequency electric repulsion
    """
    def __init__(
        self,
        node_size_x,
        node_size_y,
        num_movable_nodes,
        num_filler_nodes,
        potential_collision_map,
        qubit_dist_threhold_x,
        qubit_dist_threhold_y,
        coupler_dist_threhold_x=None,
        coupler_dist_threhold_y=None,
        placedb=None,
        ):
        """
        @brief initialization
        """
        super(FrequencyRepulsion, self).__init__()
        self.placedb = placedb
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.num_movable_nodes = num_movable_nodes
        self.num_filler_nodes = num_filler_nodes
        self.potential_collision_map = potential_collision_map
        self.qubit_dist_threhold_x = qubit_dist_threhold_x
        self.qubit_dist_threhold_y = qubit_dist_threhold_y
        self.coupler_dist_threhold_x = coupler_dist_threhold_x
        self.coupler_dist_threhold_y = coupler_dist_threhold_y


    def forward(self, pos):
        return FrequencyRepulsionFunction.apply(
            pos,
            self.node_size_x, self.node_size_y, 
            self.potential_collision_map,
            self.num_movable_nodes, self.num_filler_nodes,
            self.qubit_dist_threhold_x, self.qubit_dist_threhold_y,
            self.coupler_dist_threhold_x, self.coupler_dist_threhold_y, 
            )


