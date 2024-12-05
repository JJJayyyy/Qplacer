#include "utility/src/torch.h"
#include "utility/src/utils.h"
#include <unordered_map>
#include <vector>


QPLACER_BEGIN_NAMESPACE

template <typename T>
void computeRepulsionForceLauncher(
    const T* pos_tensor, const T* node_size_x_tensor, const T* node_size_y_tensor,
    const std::unordered_map<int, std::vector<int>>& potential_collision_map,
    T epsilon, T qubit_dist_threshold_x, T qubit_dist_threshold_y, T force_ratio,
    int num_nodes, T* energy_tensor) {

    for (const auto& pair : potential_collision_map) {
        int node = pair.first;
        const auto& potential_colliders = pair.second;

        for (int collider : potential_colliders) {
            T dx = pos_tensor[collider] - pos_tensor[node];
            T dy = pos_tensor[collider + num_nodes] - pos_tensor[node + num_nodes];
            T distance_x = std::max(std::abs(dx) - (node_size_x_tensor[collider] + node_size_x_tensor[node]), epsilon);
            T distance_y = std::max(std::abs(dy) - (node_size_y_tensor[collider] + node_size_y_tensor[node]), epsilon);
            
            if (distance_x < qubit_dist_threshold_x && distance_y < qubit_dist_threshold_y) {
                T force_x = force_ratio / (distance_x * distance_x);
                T force_y = force_ratio / (distance_y * distance_y);

                energy_tensor[node] += (dx > 0) ? force_x : -force_x;
                energy_tensor[node + num_nodes] += (dy > 0) ? force_y : -force_y;
            }
        }
    }
}

at::Tensor frequency_repulsion(
    at::Tensor pos, at::Tensor node_size_x, at::Tensor node_size_y,
    const std::unordered_map<int, std::vector<int>>& potential_collision_map,
    double qubit_dist_threshold_x, double qubit_dist_threshold_y, 
    double force_ratio, double epsilon, int num_nodes) {

    CHECK_FLAT_CPU(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    at::Tensor energy = at::zeros_like(pos);

    QPLACER_DISPATCH_FLOATING_TYPES(pos, "computeRepulsionForceLauncher", [&] {
        QPLACER_NAMESPACE::computeRepulsionForceLauncher<scalar_t>(
            QPLACER_TENSOR_DATA_PTR(pos, scalar_t),
            QPLACER_TENSOR_DATA_PTR(node_size_x, scalar_t),
            QPLACER_TENSOR_DATA_PTR(node_size_y, scalar_t),
            potential_collision_map,
            epsilon, qubit_dist_threshold_x, qubit_dist_threshold_y, force_ratio,
            num_nodes,
            QPLACER_TENSOR_DATA_PTR(energy, scalar_t));
    });

    return energy;
}

QPLACER_END_NAMESPACE


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("frequency_repulsion", &QPLACER_NAMESPACE::frequency_repulsion,
        "Frequency Electric Force");
}
