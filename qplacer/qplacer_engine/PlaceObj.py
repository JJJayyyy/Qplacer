import numpy as np
import logging
import torch
import torch.nn as nn

import operators.dreamplace.ops.weighted_average_wirelength.weighted_average_wirelength as weighted_average_wirelength
import operators.dreamplace.ops.logsumexp_wirelength.logsumexp_wirelength as logsumexp_wirelength
import operators.dreamplace.ops.electric_potential.electric_overflow as electric_overflow
import operators.dreamplace.ops.electric_potential.electric_potential as electric_potential
import operators.qplacement.ops.frequency_repulsion.frequency_repulsion as frequency_replusion


class PreconditionOp:
    """
    Preconditioning engine is critical for convergence.
    Need to be carefully designed.
    """
    def __init__(self, placedb, data_collections, op_collections):
        self.placedb = placedb
        self.data_collections = data_collections
        self.op_collections = op_collections
        self.iteration = 0
        self.alpha = 1.0
        self.best_overflow = None
        self.overflows = []

    def set_overflow(self, overflow):
        self.overflows.append(overflow)
        if self.best_overflow is None:
            self.best_overflow = overflow
        elif self.best_overflow.mean() > overflow.mean():
            self.best_overflow = overflow

    def __call__(self, grad, density_weight, update_mask=None):
        """
        Introduce alpha parameter to avoid divergence.
        It is tricky for this parameter to increase.
        """
        with torch.no_grad():
            # The preconditioning step in python is time-consuming, as in each gradient
            # pass, the total net weight should be re-calculated.
            sum_pin_weights_in_nodes = self.op_collections.pws_op(self.data_collections.net_weights)
            if density_weight.size(0) == 1:
                precond = (sum_pin_weights_in_nodes + self.alpha*density_weight*self.data_collections.node_areas)
            else:
                ### only precondition the non fence region
                node_areas = self.data_collections.node_areas.clone()
                mask = self.data_collections.node2fence_region_map[: self.placedb.num_movable_nodes] >= len(
                    self.placedb.regions
                )
                node_areas[: self.placedb.num_movable_nodes].masked_scatter_(
                    mask, node_areas[: self.placedb.num_movable_nodes][mask] * density_weight[-1]
                )
                filler_beg, filler_end = self.placedb.filler_start_map[-2:]
                node_areas[
                    self.placedb.num_nodes
                    - self.placedb.num_filler_nodes
                    + filler_beg : self.placedb.num_nodes
                    - self.placedb.num_filler_nodes
                    + filler_end
                ] *= density_weight[-1]
                precond = sum_pin_weights_in_nodes + self.alpha * node_areas

            precond.clamp_(min=1.0)
            grad[0 : self.placedb.num_nodes].div_(precond)
            grad[self.placedb.num_nodes : self.placedb.num_nodes * 2].div_(precond)
            self.iteration += 1

        return grad


class PlaceObj(nn.Module):
    """
    @brief Define placement objective:
        wirelength + density_weight * density penalty
    It includes various ops related to global placement as well.
    """
    def __init__(self, density_weight, params, placedb, data_collections,
                 op_collections, global_place_params):
        """
        @brief initialize ops for placement
        @param density_weight density weight in the objective
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param op_collections a collection of all ops
        @param global_place_params global placement parameters for current global placement stage
        """
        super(PlaceObj, self).__init__()
        ### quadratic penalty
        self.density_quad_coeff = 2000
        self.init_force = None
        ### increase density penalty if slow convergence
        self.density_factor = 1
        ### non fence region will use first-order density penalty by default
        self.quad_penalty = False
        ### update mask controls whether stop gradient/updating, 1 represents allow grad/update
        self.update_mask = None

        self.params = params
        self.placedb = placedb
        self.data_collections = data_collections
        self.op_collections = op_collections
        self.global_place_params = global_place_params
        self.gpu = params.gpu
        self.data_collections = data_collections
        self.op_collections = op_collections
        self.density_weight = torch.tensor(
            [density_weight],
            dtype=self.data_collections.pos[0].dtype,
            device=self.data_collections.pos[0].device)
        
        ### Note: even for multi-electric fields, they use the same gamma
        if "num_bins_x" in global_place_params and global_place_params["num_bins_x"] > 1:
            num_bins_x = global_place_params["num_bins_x"]  
        else:
            num_bins_x = placedb.num_bins_x

        if "num_bins_y" in global_place_params and global_place_params["num_bins_y"] > 1:
            num_bins_y = global_place_params["num_bins_y"] 
        else:
            num_bins_y = placedb.num_bins_y

        name = "Global placement: %dx%d bins by default" % (num_bins_x, num_bins_y)
        logging.info(name)
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        self.bin_size_y = (placedb.yh - placedb.yl) / num_bins_y
        self.gamma = torch.tensor(10 * self.base_gamma(params, placedb),
                                  dtype=self.data_collections.pos[0].dtype,
                                  device=self.data_collections.pos[0].device)

        # compute weighted average wirelength from position
        name = "%dx%d bins" % (num_bins_x, num_bins_y)
        self.name = name
        if global_place_params["wirelength"] == "weighted_average":
            self.op_collections.wirelength_op, self.op_collections.update_gamma_op = self.build_weighted_average_wl(
                params, placedb, self.data_collections, self.op_collections.pin_pos_op)
        elif global_place_params["wirelength"] == "logsumexp":
            self.op_collections.wirelength_op, self.op_collections.update_gamma_op = self.build_logsumexp_wl(
                params, placedb, self.data_collections, self.op_collections.pin_pos_op)
        else:
            assert 0, "unknown wirelength model %s" % (global_place_params["wirelength"])

        self.op_collections.density_overflow_op = self.build_electric_overflow(
            params, placedb, self.data_collections, self.num_bins_x, self.num_bins_y)
        self.op_collections.density_op = self.build_electric_potential(
            params, placedb, self.data_collections, self.num_bins_x, self.num_bins_y, name=name)
        
        # frequency penalty
        if params.frequency_assign:
            assert hasattr(params, 'potential_collision_map')
            self.potential_collision_map = params.potential_collision_map
            self.op_collections.frequency_repulsion_op = self.build_frequency_repulsion(
                params, placedb, self.data_collections) 
            self.frequency_repulsion_weight = torch.tensor(
                [density_weight], dtype=self.data_collections.pos[0].dtype, device=self.data_collections.pos[0].device)
            self.op_collections.update_frequency_repulsion_weight_op = self.build_update_frequency_repulsion_weight(params)

        self.op_collections.update_density_weight_op = self.build_update_density_weight(params, placedb)
        self.op_collections.precondition_op = self.build_precondition(params, placedb, self.data_collections, self.op_collections)
        self.op_collections.noise_op = self.build_noise(params, placedb, self.data_collections)
        self.Lgamma_iteration = global_place_params["iteration"]
        if 'Llambda_density_weight_iteration' in global_place_params:
            self.Llambda_density_weight_iteration = global_place_params['Llambda_density_weight_iteration']
        else:
            self.Llambda_density_weight_iteration = 1
        if 'Lsub_iteration' in global_place_params:
            self.Lsub_iteration = global_place_params['Lsub_iteration']
        else:
            self.Lsub_iteration = 1



    def obj_fn(self, pos):
        """
        @brief Compute objective.
            wirelength + density_weight * density penalty + repulsion_weight * repulsion penalty
        @param pos locations of cells
        @return objective value
        """
        self.wirelength = self.op_collections.wirelength_op(pos)
        self.density = self.op_collections.density_op(pos)
        if self.init_force is None:   ### record initial density
            self.init_force = self.density.data.clone()
            ### density weight subgradient preconditioner
            self.density_weight_grad_precond = self.init_force.masked_scatter(self.init_force > 0, 1 /self.init_force[self.init_force > 0])
            self.quad_penalty_coeff = self.density_quad_coeff / 2 * self.density_weight_grad_precond
        
        result = torch.add(self.wirelength, self.density, 
                           alpha=(self.density_factor * self.density_weight).item())     
        # print(f"self.wirelength: {self.wirelength}")
        # print('density_weight: {}, density: {}, alpha * density: {}'.format(
        #     self.density_weight,
        #     self.density,
        #     self.density*(self.density_factor * self.density_weight).item()
        #     ))
        # print(f"result : {result:.4f}")
        if self.params.frequency_assign:
            self.frequency_force = self.op_collections.frequency_repulsion_op(pos)
            result = torch.add(result, self.frequency_force, 
                               alpha=(self.density_factor * self.frequency_repulsion_weight).item())
            # print('f_force_weight: {}, f_force: {}, alpha * f_force: {}'.format(
            # self.frequency_repulsion_weight,
            # self.frequency_force,
            # self.frequency_force * (self.density_factor * self.frequency_repulsion_weight).item()
            # ))
            # print(f"freq result : {result:.4f}")
        # assert 0
        return result


    def obj_and_grad_fn(self, pos):
        """
        @brief compute objective and gradient.
        @param pos locations of cells
        @return objective value
        """
        #self.check_gradient(pos)
        if pos.grad is not None:
            pos.grad.zero_()
        obj = self.obj_fn(pos)  # forward is called for nesterov
        obj.backward()
        self.op_collections.precondition_op(pos.grad, self.density_weight, self.update_mask)
        return obj, pos.grad


    def estimate_initial_learning_rate(self, x_k, lr):
        """
        @brief Estimate initial learning rate by moving a small step.
        Computed as | x_k - x_k_1 |_2 / | g_k - g_k_1 |_2.
        @param x_k current solution
        @param lr small step
        """
        obj_k, g_k = self.obj_and_grad_fn(x_k)
        x_k_1 = torch.autograd.Variable(x_k - lr * g_k, requires_grad=True)
        obj_k_1, g_k_1 = self.obj_and_grad_fn(x_k_1)

        return (x_k - x_k_1).norm(p=2) / (g_k - g_k_1).norm(p=2)

    def build_weighted_average_wl(self, params, placedb, data_collections, pin_pos_op):
        """
        @brief build the op to compute weighted average wirelength
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param pin_pos_op the op to compute pin locations according to cell locations
        """
        # use WeightedAverageWirelength atomic
        # logging.debug(f'flat_net2pin_map ({len(data_collections.flat_net2pin_map)}): {data_collections.flat_net2pin_map}')
        # logging.debug(f'flat_net2pin_start_map ({len(data_collections.flat_net2pin_start_map)}): {data_collections.flat_net2pin_start_map}')
        # logging.debug(f'pin2net_map ({len(data_collections.pin2net_map)}): {data_collections.pin2net_map}')
        logging.debug(f'net_weights ({len(data_collections.net_weights)}): {data_collections.net_weights}')
        wirelength_for_pin_op = weighted_average_wirelength.WeightedAverageWirelength(
            flat_netpin=data_collections.flat_net2pin_map,
            netpin_start=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            net_weights=data_collections.net_weights,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            pin_mask=data_collections.pin_mask_ignore_fixed_macros,
            gamma=self.gamma,
            algorithm='merged')
        # wirelength for position
        def build_wirelength_op(pos):
            return wirelength_for_pin_op(pin_pos_op(pos))

        # update gamma
        base_gamma = self.base_gamma(params, placedb)
        def build_update_gamma_op(iteration, overflow):
            self.update_gamma(iteration, overflow, base_gamma)
            #logging.debug("update gamma to %g" % (wirelength_for_pin_op.gamma.data))

        return build_wirelength_op, build_update_gamma_op

    def build_logsumexp_wl(self, params, placedb, data_collections, pin_pos_op):
        """
        @brief build the op to compute log-sum-exp wirelength
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param pin_pos_op the op to compute pin locations according to cell locations
        """
        wirelength_for_pin_op = logsumexp_wirelength.LogSumExpWirelength(
            flat_netpin=data_collections.flat_net2pin_map,
            netpin_start=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            net_weights=data_collections.net_weights,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            pin_mask=data_collections.pin_mask_ignore_fixed_macros,
            gamma=self.gamma,
            algorithm='merged')

        # wirelength for position
        def build_wirelength_op(pos):
            return wirelength_for_pin_op(pin_pos_op(pos))

        # update gamma
        base_gamma = self.base_gamma(params, placedb)

        def build_update_gamma_op(iteration, overflow):
            self.update_gamma(iteration, overflow, base_gamma)
            #logging.debug("update gamma to %g" % (wirelength_for_pin_op.gamma.data))

        return build_wirelength_op, build_update_gamma_op

    def build_electric_overflow(self, params, placedb, data_collections, num_bins_x, num_bins_y):
        """
        @brief compute electric density overflow
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param num_bins_x number of bins in horizontal direction
        @param num_bins_y number of bins in vertical direction
        """
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y

        return electric_overflow.ElectricOverflow(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            bin_center_x=data_collections.bin_center_x_padded(placedb, 0, num_bins_x),
            bin_center_y=data_collections.bin_center_y_padded(placedb, 0, num_bins_y),
            target_density=data_collections.target_density,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminals=placedb.num_terminals,
            num_filler_nodes=0,
            padding=0,
            deterministic_flag=params.deterministic_flag,
            sorted_node_map=data_collections.sorted_node_map,
            movable_macro_mask=data_collections.movable_macro_mask)

    def build_electric_potential(self, params, placedb, data_collections, num_bins_x, num_bins_y, name):
        """
        @brief e-place electrostatic potential
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param num_bins_x number of bins in horizontal direction
        @param num_bins_y number of bins in vertical direction
        @param name string for printing
        @param fence_regions a [n_subregions, 4] tensor for fence regions potential penalty
        """
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y

        max_num_bins_x = np.ceil(
            (np.amax(placedb.node_size_x[0:placedb.num_movable_nodes]) + 2*bin_size_x)/bin_size_x)
        max_num_bins_y = np.ceil(
            (np.amax(placedb.node_size_y[0:placedb.num_movable_nodes]) + 2*bin_size_y)/bin_size_y)
        max_num_bins = max(int(max_num_bins_x), int(max_num_bins_y))
        logging.info("%s #bins %dx%d, bin sizes %gx%g, max_num_bins = %d, padding = %d"
            % (name, num_bins_x, num_bins_y,
               bin_size_x / placedb.row_height,
               bin_size_y / placedb.row_height, max_num_bins, 0))
        if num_bins_x < max_num_bins:
            logging.warning("num_bins_x (%d) < max_num_bins (%d)" % (num_bins_x, max_num_bins))
        if num_bins_y < max_num_bins:
            logging.warning("num_bins_y (%d) < max_num_bins (%d)" % (num_bins_y, max_num_bins))

        target_density = data_collections.target_density.item()
        return electric_potential.ElectricPotential(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            bin_center_x=data_collections.bin_center_x_padded(placedb, 0, num_bins_x),
            bin_center_y=data_collections.bin_center_y_padded(placedb, 0, num_bins_y),
            target_density=target_density,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminals=placedb.num_terminals,
            num_filler_nodes=placedb.num_filler_nodes,
            padding=0,
            deterministic_flag=params.deterministic_flag,
            sorted_node_map=data_collections.sorted_node_map,
            movable_macro_mask=data_collections.movable_macro_mask,
            fast_mode=params.RePlAce_skip_energy_flag,
            region_id=None,
            fence_regions=None,
            node2fence_region_map=data_collections.node2fence_region_map,
            placedb=placedb)

    def initialize_density_weight(self, params, placedb):
        """
        @brief compute initial density weight
        @param params parameters
        @param placedb placement database
        """
        wirelength = self.op_collections.wirelength_op(self.data_collections.pos[0])
        if self.data_collections.pos[0].grad is not None:
            self.data_collections.pos[0].grad.zero_()
        wirelength.backward()
        wirelength_grad_norm = self.data_collections.pos[0].grad.norm(p=1)
        self.data_collections.pos[0].grad.zero_()

        density = self.op_collections.density_op(self.data_collections.pos[0])
        ### record initial density
        self.init_force = density.data.clone()
        density.backward()
        density_grad_norm = self.data_collections.pos[0].grad.norm(p=1)

        grad_norm_ratio = wirelength_grad_norm / density_grad_norm
        self.density_weight = torch.tensor(
            [params.density_weight * grad_norm_ratio],
            dtype=self.data_collections.pos[0].dtype,
            device=self.data_collections.pos[0].device)

        return self.density_weight

    def build_update_density_weight(self, params, placedb, algo="hpwl"):
        """
        @brief update density weight
        @param params parameters
        @param placedb placement database
        """
        ### params for hpwl mode from RePlAce
        ref_hpwl = params.RePlAce_ref_hpwl
        LOWER_PCOF = params.RePlAce_LOWER_PCOF
        UPPER_PCOF = params.RePlAce_UPPER_PCOF
        ### params for overflow mode from elfPlace
        assert algo in {"hpwl", "overflow"}, logging.error("density weight update not supports hpwl mode or overflow mode")

        def update_density_weight_op_hpwl(cur_metric, prev_metric, iteration):
            ### based on hpwl
            with torch.no_grad():
                delta_hpwl = cur_metric.hpwl - prev_metric.hpwl
                if delta_hpwl < 0:
                    mu = UPPER_PCOF * np.maximum(np.power(0.9999, float(iteration)), 0.98)
                else:
                    mu = UPPER_PCOF * torch.pow(UPPER_PCOF, -delta_hpwl / ref_hpwl).clamp(min=LOWER_PCOF, max=UPPER_PCOF)
                self.density_weight *= mu

        def update_density_weight_op_overflow(cur_metric, prev_metric, iteration):
            assert self.quad_penalty == True, logging.error("density weight update based on overflow only works for quadratic density penalty")
            ### based on overflow
            ### stop updating if a region has lower overflow than stop overflow
            with torch.no_grad():
                density_norm = cur_metric.density * self.density_weight_grad_precond
                density_weight_grad = density_norm + self.density_quad_coeff / 2 * density_norm ** 2
                density_weight_grad /= density_weight_grad.norm(p=2)

                self.density_weight_u += self.density_weight_step_size * density_weight_grad
                density_weight_s = 1 + self.density_quad_coeff * density_norm

                density_weight_new = (self.density_weight_u * density_weight_s).clamp(max=10)

                ### conditional update if this region's overflow is higher than stop overflow
                if(self.update_mask is None):
                    self.update_mask = cur_metric.overflow >= self.params.stop_overflow
                else:
                    ### restart updating is not allowed
                    self.update_mask &= cur_metric.overflow >= self.params.stop_overflow
                self.density_weight.masked_scatter_(self.update_mask, density_weight_new[self.update_mask])

                ### update density weight step size
                rate = torch.log(self.density_quad_coeff * density_norm.norm(p=2)).clamp(min=0)
                rate = rate / (1 + rate)
                rate = rate * (self.density_weight_step_size_inc_high - self.density_weight_step_size_inc_low) + self.density_weight_step_size_inc_low
                self.density_weight_step_size *= rate

        update_density_weight_op = {"hpwl":update_density_weight_op_hpwl,
                                    "overflow": update_density_weight_op_overflow}[algo]

        return update_density_weight_op

    def base_gamma(self, params, placedb):
        """
        @brief compute base gamma
        @param params parameters
        @param placedb placement database
        """
        return params.gamma * (self.bin_size_x + self.bin_size_y)

    def update_gamma(self, iteration, overflow, base_gamma):
        """
        @brief update gamma in wirelength model
        @param iteration optimization step
        @param overflow evaluated in current step
        @param base_gamma base gamma
        """
        ### overflow can have multiple values for fence regions, use their weighted average based on movable node number
        if overflow.numel() == 1:
            overflow_avg = overflow
        else:
            overflow_avg = overflow
        coef = torch.pow(10, (overflow_avg - 0.1) * 20 / 9 - 1)
        self.gamma.data.fill_((base_gamma * coef).item())
        return True

    def build_noise(self, params, placedb, data_collections):
        """
        @brief add noise to cell locations
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        """
        node_size = torch.cat(
            [data_collections.node_size_x, data_collections.node_size_y],
            dim=0).to(data_collections.pos[0].device)

        def noise_op(pos, noise_ratio):
            with torch.no_grad():
                noise = torch.rand_like(pos)
                noise.sub_(0.5).mul_(node_size).mul_(noise_ratio)
                # no noise to fixed cells
                noise[placedb.num_movable_nodes:placedb.num_nodes - placedb.num_filler_nodes].zero_()
                noise[placedb.num_nodes+placedb.num_movable_nodes : 2*placedb.num_nodes-placedb.num_filler_nodes].zero_()
                return pos.add_(noise)

        return noise_op

    def build_precondition(self, params, placedb, data_collections, op_collections):
        """
        @brief preconditioning to gradient
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param op_collections a collection of all ops
        """
        return PreconditionOp(placedb, data_collections, op_collections)

    # frequency op
    def initialize_frequency_repulsion_weight(self, params):
        """
        @brief compute initial frequency repulsion weight
        @param params parameters
        """
        wirelength = self.op_collections.wirelength_op(self.data_collections.pos[0])
        if self.data_collections.pos[0].grad is not None:
            self.data_collections.pos[0].grad.zero_()
        wirelength.backward()
        wirelength_grad_norm = self.data_collections.pos[0].grad.norm(p=1)
        self.data_collections.pos[0].grad.zero_()

        frequency_repulsion = self.op_collections.frequency_repulsion_op(self.data_collections.pos[0])
        ### record initial frequency repulsion
        self.init_force = frequency_repulsion.data.clone()
        frequency_repulsion.backward()
        repulsion_grad_norm = self.data_collections.pos[0].grad.norm(p=1)
        grad_norm_ratio = wirelength_grad_norm / repulsion_grad_norm

        self.frequency_repulsion_weight = torch.tensor(
            [params.frequency_density_weight * grad_norm_ratio * 1],
            dtype=self.data_collections.pos[0].dtype,
            device=self.data_collections.pos[0].device)
        return self.frequency_repulsion_weight


    def build_update_frequency_repulsion_weight(self, params):
        """
        @brief update density weight
        @param params parameters
        """
        ### params for hpwl mode from RePlAce
        ref_hpwl = params.RePlAce_ref_hpwl
        LOWER_PCOF = params.RePlAce_LOWER_PCOF
        UPPER_PCOF = params.RePlAce_UPPER_PCOF

        def update_frequency_repulsion_weight_op_hpwl(cur_metric, prev_metric, iteration):
            with torch.no_grad():
                delta_hpwl = cur_metric.hpwl - prev_metric.hpwl
                if delta_hpwl < 0:
                    mu = UPPER_PCOF * np.maximum(np.power(0.9999, float(iteration)), 0.98)
                else:
                    mu = UPPER_PCOF * torch.pow(
                        UPPER_PCOF, -delta_hpwl / ref_hpwl).clamp(min=LOWER_PCOF, max=UPPER_PCOF)
                self.frequency_repulsion_weight *= mu

        update_frequency_density_repulsion_op = update_frequency_repulsion_weight_op_hpwl
        return update_frequency_density_repulsion_op
    

    def build_frequency_repulsion(self, params, placedb, data_collections):
        """
        frequency electrostatic repulsion

        Args:
        - params: parameters
        - placedb: placement database
        - data_collections: a collection of data and variables required for constructing ops
        """
        node_size_x = torch.tensor(data_collections.node_size_x, 
                                   device=data_collections.pos[0].device, 
                                   dtype=torch.float32)
        node_size_y = torch.tensor(data_collections.node_size_y, 
                                   device=data_collections.pos[0].device, 
                                   dtype=torch.float32)

        params.potential_collision_map = {int(key): value for key, value in params.potential_collision_map.items()}
        return frequency_replusion.FrequencyRepulsion(
            node_size_x=node_size_x,
            node_size_y=node_size_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_filler_nodes=placedb.num_filler_nodes,
            potential_collision_map=params.potential_collision_map,
            qubit_dist_threhold_x=params.qubit_dist_threhold,
            qubit_dist_threhold_y=params.qubit_dist_threhold,
            coupler_dist_threhold_x=params.coupler_dist_threhold,
            coupler_dist_threhold_y=params.coupler_dist_threhold,
            placedb=placedb, 
            )