#include "utility/src/torch.h"
#include "utility/src/utils.h"
// database dependency
#include "utility/src/legalization_db.h"
#include "utility/src/make_placedb.h"
// local dependency
#include "qubit_legalize/src/hannan_legalize.h"
#include "qubit_legalize/src/lp_legalize.h"

QPLACER_BEGIN_NAMESPACE

/// @brief The Qubit legalization follows the way of floorplanning,
/// because Qubit have quite different sizes.
/// @return true if legal
template <typename T>
bool qubitLegalizationLauncher(LegalizationDB<T> db);

/// @brief legalize layout with greedy legalization.
/// Only movable nodes will be moved. Fixed nodes and filler nodes are fixed.
///
/// @param init_pos initial locations of nodes, including movable nodes, fixed
/// nodes, and filler nodes, [0, num_movable_nodes) are movable nodes,
/// [num_movable_nodes, num_nodes-num_filler_nodes) are fixed nodes,
/// [num_nodes-num_filler_nodes, num_nodes) are filler nodes
/// @param node_size_x width of nodes, including movable nodes, fixed nodes, and
/// filler nodes, [0, num_movable_nodes) are movable nodes, [num_movable_nodes,
/// num_nodes-num_filler_nodes) are fixed nodes, [num_nodes-num_filler_nodes,
/// num_nodes) are filler nodes
/// @param node_size_y height of nodes, including movable nodes, fixed nodes,
/// and filler nodes, same as node_size_x
/// @param xl left edge of bounding box of layout area
/// @param yl bottom edge of bounding box of layout area
/// @param xh right edge of bounding box of layout area
/// @param yh top edge of bounding box of layout area
/// @param site_width width of a placement site
/// @param row_height height of a placement row
/// @param num_bins_x number of bins in horizontal direction
/// @param num_bins_y number of bins in vertical direction
/// @param num_nodes total number of nodes, including movable nodes, fixed
/// nodes, and filler nodes; fixed nodes are in the range of [num_movable_nodes,
/// num_nodes-num_filler_nodes)
/// @param num_movable_nodes number of movable nodes, movable nodes are in the
/// range of [0, num_movable_nodes)
/// @param number of filler nodes, filler nodes are in the range of
/// [num_nodes-num_filler_nodes, num_nodes)
at::Tensor qubit_legalization_forward(
    at::Tensor init_pos, at::Tensor pos, at::Tensor node_size_x,
    at::Tensor node_size_y, at::Tensor node_weights,
    at::Tensor flat_region_boxes, at::Tensor flat_region_boxes_start,
    at::Tensor node2fence_region_map, double xl, double yl, double xh,
    double yh, double site_width, double row_height, int num_bins_x,
    int num_bins_y, int num_movable_nodes, int num_terminal_NIs,
    int num_filler_nodes) {
	CHECK_FLAT_CPU(init_pos);
	CHECK_EVEN(init_pos);
	CHECK_CONTIGUOUS(init_pos);

	auto pos_copy = pos.clone();

	CPUTimer::hr_clock_rep timer_start, timer_stop;
	timer_start = CPUTimer::getGlobaltime();
	// Call the cuda kernel launcher
	QPLACER_DISPATCH_FLOATING_TYPES(pos, "qubitLegalizationLauncher", [&] {
		auto db = make_placedb<scalar_t>(
			init_pos, pos_copy, node_size_x, node_size_y, node_weights,
			flat_region_boxes, flat_region_boxes_start, node2fence_region_map,
			xl, yl, xh, yh, site_width, row_height, num_bins_x, num_bins_y,
			num_movable_nodes, num_terminal_NIs, num_filler_nodes);
		qubitLegalizationLauncher<scalar_t>(db);
	});
	timer_stop = CPUTimer::getGlobaltime();
	qplacerPrint(kINFO, "Qubit legalization takes %g ms\n",
					(timer_stop - timer_start) * CPUTimer::getTimerPeriod());

	return pos_copy;
}



template <typename T>
bool check_qubit_legality(LegalizationDB<T> db, const std::vector<int>& qubits, bool fast_check) {
	// check legality between movable and fixed qubits
	// for debug only, so it is slow
	auto checkOverlap2Nodes = [&](int i, int node_id1, T xl1, T yl1, T width1,
									T height1, int j, int node_id2, T xl2, T yl2,
									T width2, T height2) {
		T xh1 = xl1 + width1;
		T yh1 = yl1 + height1;
		T xh2 = xl2 + width2;
		T yh2 = yl2 + height2;
	
		if (std::min(xh1, xh2) > std::max(xl1, xl2) && std::min(yh1, yh2) > std::max(yl1, yl2)) {
			qplacerPrint((fast_check) ? kWARN : kERROR,
							"qubit %d (%g, %g, %g, %g) var %d overlaps with qubit %d "
							"(%g, %g, %g, %g) var %d, fixed: %d\n",
							node_id1, xl1, yl1, xh1, yh1, i, node_id2, xl2, yl2, xh2,
							yh2, j, (int)(node_id2 >= db.num_movable_nodes));
		return true;
		}
		return false;
  	};

	bool legal = true;
	for (unsigned int i = 0, ie = qubits.size(); i < ie; ++i) {
		int node_id1 = qubits[i];
		T xl1 = db.x[node_id1];
		T yl1 = db.y[node_id1];
		T width1 = db.node_size_x[node_id1];
		T height1 = db.node_size_y[node_id1];
		// constraints with other qubits
    	for (unsigned int j = i + 1; j < ie; ++j) {
			int node_id2 = qubits[j];
			T xl2 = db.x[node_id2];
			T yl2 = db.y[node_id2];
			T width2 = db.node_size_x[node_id2];
			T height2 = db.node_size_y[node_id2];

			bool overlap = checkOverlap2Nodes(i, node_id1, xl1, yl1, width1, height1,
												j, node_id2, xl2, yl2, width2, height2);
			if (overlap) {
				legal = false;
				if (fast_check) {return legal;}
			}
		}
		// constraints with fixed qubits
		// when considering fixed qubits, there is no guarantee to find legal
		// solution with current ad-hoc constraint graphs
		for (int j = db.num_movable_nodes; j < db.num_nodes; ++j) {
			int node_id2 = j;
			T xl2 = db.init_x[node_id2];
			T yl2 = db.init_y[node_id2];
			T width2 = db.node_size_x[node_id2];
			T height2 = db.node_size_y[node_id2];

			bool overlap = checkOverlap2Nodes(i, node_id1, xl1, yl1, width1, height1,
												j, node_id2, xl2, yl2, width2, height2);
				if (overlap) {
					legal = false;
					if (fast_check) {return legal;}
				}
		}
  }
  if (legal) {
    qplacerPrint(kDEBUG, "Qubit legality check [PASSED]\n");
  } else {
    qplacerPrint(kERROR, "Qubit legality check [FAILED]\n");
  }

  return legal;
}



template <typename T>
bool check_qubit_legality_touch(LegalizationDB<T> db, const std::vector<int>& qubits,
                          bool fast_check) {
  // check legality between movable and fixed qubits
  auto checkOverlap2Nodes = [&](int i, int node_id1, T xl1, T yl1, T width1,
                                T height1, int j, int node_id2, T xl2, T yl2,
                                T width2, T height2) {
    T xh1 = xl1 + width1;
    T yh1 = yl1 + height1;
    T xh2 = xl2 + width2;
    T yh2 = yl2 + height2;
    
    if (std::min(xh1, xh2) > std::max(xl1, xl2) - db.site_width &&
        std::min(yh1, yh2) > std::max(yl1, yl2) - db.site_width) { // touch return false

      qplacerPrint((fast_check) ? kWARN : kERROR,
                      "qubit %d (%g, %g, %g, %g) var %d touches/overlaps with qubit %d "
                      "(%g, %g, %g, %g) var %d, fixed: %d\n",
                      node_id1, xl1, yl1, xh1, yh1, i, node_id2, xl2, yl2, xh2,
                      yh2, j, (int)(node_id2 >= db.num_movable_nodes));
      return true;
    }
    return false;
  };

  bool legal = true;
  for (unsigned int i = 0, ie = qubits.size(); i < ie; ++i) {
    int node_id1 = qubits[i];
    T xl1 = db.x[node_id1];
    T yl1 = db.y[node_id1];
    T width1 = db.node_size_x[node_id1];
    T height1 = db.node_size_y[node_id1];
    // constraints with other qubits
    for (unsigned int j = i + 1; j < ie; ++j) {
      int node_id2 = qubits[j];
      T xl2 = db.x[node_id2];
      T yl2 = db.y[node_id2];
      T width2 = db.node_size_x[node_id2];
      T height2 = db.node_size_y[node_id2];

      bool overlap = checkOverlap2Nodes(i, node_id1, xl1, yl1, width1, height1,
                                        j, node_id2, xl2, yl2, width2, height2);
      if (overlap) {
        legal = false;
        if (fast_check) {
          return legal;
        }
      }
    }
    // constraints with fixed qubits
    // when considering fixed qubits, there is no guarantee to find legal
    // solution with current ad-hoc constraint graphs
    for (int j = db.num_movable_nodes; j < db.num_nodes; ++j) {
      int node_id2 = j;
      T xl2 = db.init_x[node_id2];
      T yl2 = db.init_y[node_id2];
      T width2 = db.node_size_x[node_id2];
      T height2 = db.node_size_y[node_id2];

      bool overlap = checkOverlap2Nodes(i, node_id1, xl1, yl1, width1, height1,
                                        j, node_id2, xl2, yl2, width2, height2);
      if (overlap) {
        legal = false;
        if (fast_check) {
          return legal;
        }
      }
    }
  }
  if (legal) {
    qplacerPrint(kDEBUG, "Qubit touch-legality check [PASSED]\n");
  } else {
    qplacerPrint(kERROR, "Qubit touch-legality check [FAILED]\n");
  }
  return legal;
}



template <typename T>
struct QubitLegalizeStats {
  T total_displace;
  T max_displace;
  T total_weighted_displace;  ///< displacement weighted by qubit area ratio to
                              ///< average qubit area
  T max_weighted_displace;
  // T average_qubit_area;
};

template <typename T>
QubitLegalizeStats<T> compute_displace(const LegalizationDB<T>& db,
                                       const std::vector<int>& qubits) {
  QubitLegalizeStats<T> stats;
  stats.total_displace = 0;
  stats.max_displace = 0;
  stats.total_weighted_displace = 0;
  stats.max_weighted_displace = 0;

  for (auto node_id : qubits) {
    T displace = std::abs(db.init_x[node_id] - db.x[node_id]) +
                 std::abs(db.init_y[node_id] - db.y[node_id]);
    stats.total_displace += displace;
    stats.max_displace = std::max(stats.max_displace, displace);

    displace *= db.node_weights[node_id];
    stats.total_weighted_displace += displace;
    stats.max_weighted_displace =
        std::max(stats.max_weighted_displace, displace);
  }
  return stats;
}

/// @brief Rough legalize some special qubits
/// 1. qubits that form small clusters overlapping with each other
/// 2. qubits blocked by big ones
/// All the other qubits are regarded as fixed.
/// @param small_clusters_flag controls whether to perform the legalization for
/// 1
/// @param blocked_qubits_flag controls whether to perform the legalization for
/// 2
template <typename T>
void roughLegalizeLauncher(const LegalizationDB<T>& db,
                           const std::vector<int>& qubits,
                           const std::vector<int>& fixed_qubits,
                           bool small_clusters_flag, bool blocked_qubits_flag) {

  std::vector<unsigned char> markers(db.num_nodes, false);
  std::vector<int> qubits_for_rough_legalize;
  std::vector<int> fixed_qubits_for_rough_legalize;

  // collect small clusters
  if (small_clusters_flag) {
    std::vector<std::vector<int> > clusters(qubits.size());
    T cluster_area_ratio = 2;
    T cluster_overlap_ratio = 0.5;
    unsigned int cluster_qubit_numbers_threshold = 2;
    for (unsigned int i = 0, ie = qubits.size(); i < ie; ++i) {
      int node_id1 = qubits[i];
      Box<T> box1(db.x[node_id1], db.y[node_id1],
                  db.x[node_id1] + db.node_size_x[node_id1],
                  db.y[node_id1] + db.node_size_y[node_id1]);
      T a1 = box1.area();
      clusters.at(i).push_back(node_id1);
      for (unsigned int j = i + 1; j < ie; ++j) {
        int node_id2 = qubits[j];
        Box<T> box2(db.x[node_id2], db.y[node_id2],
                    db.x[node_id2] + db.node_size_x[node_id2],
                    db.y[node_id2] + db.node_size_y[node_id2]);
        T a2 = box2.area();

        if (a1 >= a2 / cluster_area_ratio && a1 <= a2 * cluster_area_ratio) {
          T overlap = std::max((T)0, std::min(box1.xh, box2.xh) -
                                         std::max(box1.xl, box2.xl)) *
                      std::max((T)0, std::min(box1.yh, box2.yh) -
                                         std::max(box1.yl, box2.yl));
          if (overlap >= std::min(a1, a2) * cluster_overlap_ratio) {
            clusters.at(i).push_back(node_id2);
          }
        }
      }
    }
    for (unsigned int i = 0, ie = qubits.size(); i < ie; ++i) {
      if (clusters.at(i).size() >= cluster_qubit_numbers_threshold) {
        markers.at(qubits.at(i)) = true;
      }
    }
  }
  // collect small qubits blocked by large ones
  // If a small qubit is blocked by two big qubits, it is easier to move the
  // small one around. We detect such blocks by checking whether the qubit is
  // blocked from left, right, bottom, top 4 directions. Any qubit with (left,
  // right) or (bottom, top) blocked will be collected.
  if (blocked_qubits_flag) {
    T blocked_qubits_area_ratio =
        10;  // the area ratio of qubits to be regarded as large
    T blocked_qubits_direct_threshold = 0.9;  // determine the direction blocked
    for (unsigned int i = 0, ie = qubits.size(); i < ie; ++i) {
      int node_id1 = qubits[i];
      if (!markers[node_id1]) {
        Box<T> box1(db.x[node_id1], db.y[node_id1],
                    db.x[node_id1] + db.node_size_x[node_id1],
                    db.y[node_id1] + db.node_size_y[node_id1]);
        T a1 = box1.area();
        std::array<unsigned char, 4> intersect_directs;  // from L, R, B, T
                                                         // direction, the box
                                                         // is overlapped
        intersect_directs.fill(0);
        for (unsigned int j = 0; j < ie; ++j) {
          int node_id2 = qubits[j];
          if (i != j && !markers[node_id2]) {
            Box<T> box2(db.x[node_id2], db.y[node_id2],
                        db.x[node_id2] + db.node_size_x[node_id2],
                        db.y[node_id2] + db.node_size_y[node_id2]);
            T a2 = box2.area();

            if (a1 * blocked_qubits_area_ratio < a2) {
              Box<T> intersect_box(
                  std::max(box1.xl, box2.xl), std::max(box1.yl, box2.yl),
                  std::min(box1.xh, box2.xh), std::min(box1.yh, box2.yh));
              if (intersect_box.xl < intersect_box.xh &&
                  intersect_box.yl < intersect_box.yh) {
                if (intersect_box.height() >
                    box1.height() * blocked_qubits_direct_threshold) {
                  if (box2.xl <= box1.xl) {
                    intersect_directs[kXLOW] = 1;
                  }
                  if (box2.xh >= box1.xh) {
                    intersect_directs[kXHIGH] = 1;
                  }
                }
                if (intersect_box.width() >
                    box1.width() * blocked_qubits_direct_threshold) {
                  if (box2.yl <= box1.yl) {
                    intersect_directs[kYLOW] = 1;
                  }
                  if (box2.yh >= box1.yh) {
                    intersect_directs[kYHIGH] = 1;
                  }
                }

                if (node_id1 == 1096131 || node_id1 == 1096158) {
                  qplacerPrint(kDEBUG,
                                  "%d (%g, %g, %g, %g) overlap %d (%g, %g, %g, "
                                  "%g), (%g, %g, %g, %g), (%u, %u, %u, %u)\n",
                                  node_id1, box1.xl, box1.yl, box1.xh, box1.yh,
                                  node_id2, box2.xl, box2.yl, box2.xh, box2.yh,
                                  intersect_box.xl, intersect_box.yl,
                                  intersect_box.xh, intersect_box.yh,
                                  (unsigned)intersect_directs[0],
                                  (unsigned)intersect_directs[1],
                                  (unsigned)intersect_directs[2],
                                  (unsigned)intersect_directs[3]);
                }
              }
            }
            if ((intersect_directs[kXLOW] && intersect_directs[kXHIGH]) ||
                (intersect_directs[kYLOW] && intersect_directs[kYHIGH])) {
              markers[node_id1] = true;
              qplacerPrint(kDEBUG, "collect %d\n", node_id1);
              break;
            }
          }
        }
      }
    }
  }

  fixed_qubits_for_rough_legalize = fixed_qubits;
  for (auto node_id : qubits) {
    if (markers[node_id]) {
      qubits_for_rough_legalize.push_back(node_id);
    } else {
      fixed_qubits_for_rough_legalize.push_back(node_id);
    }
  }

  qplacerPrint(kINFO, "Rough legalize small clusters with %lu qubits\n",
                  qubits_for_rough_legalize.size());
#ifdef DEBUG
  qplacerPrint(kDEBUG, "qubits_for_rough_legalize[%lu]\n",
                  qubits_for_rough_legalize.size());
  for (auto node_id : qubits_for_rough_legalize) {
    qplacerPrint(kNONE, " %d", node_id);
  }
  qplacerPrint(kNONE, "\n");
#endif
  hannanLegalizeLauncher(db, qubits_for_rough_legalize,
                         fixed_qubits_for_rough_legalize, 1);
}


template <typename T>
bool qubitLegalizationLauncher(LegalizationDB<T> db) {
  // collect qubits
  std::vector<int> qubits;
  for (int i = 0; i < db.num_movable_nodes; ++i) {
    if (db.is_dummy_fixed(i)) {
      // in some extreme case, some qubits with 0 area should be ignored
      T area = db.node_size_x[i] * db.node_size_y[i];
      if (area > 0) {
        qubits.push_back(i);
      }
#ifdef DEBUG
      qplacerPrint(kDEBUG, "qubit %d %gx%g\n", i, db.node_size_x[i],
                      db.node_size_y[i]);
#endif
    }
  }
  qplacerPrint(
      kINFO,
      "Qubit legalization: regard %lu cells as dummy fixed (movable qubits)\n",
      qubits.size());

  // in case there is no movable qubits
  if (qubits.empty()) {
    return true;
  }

  // fixed qubits
  std::vector<int> fixed_qubits;
  fixed_qubits.reserve(db.num_nodes - db.num_movable_nodes);
  for (int i = db.num_movable_nodes; i < db.num_nodes; ++i) {
    // in some extreme case, some fixed qubits with 0 area should be ignored
    T area = db.node_size_x[i] * db.node_size_y[i];
    if (area > 0) {
      fixed_qubits.push_back(i);
    }
  }

  // store the best legalization solution found
  std::vector<T> best_x(qubits.size());
  std::vector<T> best_y(qubits.size());
  QubitLegalizeStats<T> best_displace;
  best_displace.total_displace = std::numeric_limits<T>::max();
  best_displace.max_displace = std::numeric_limits<T>::max();
  best_displace.total_weighted_displace = std::numeric_limits<T>::max();
  best_displace.max_weighted_displace = std::numeric_limits<T>::max();

  // update current best solution
  auto update_best = [&](bool legal, const QubitLegalizeStats<T>& displace) {
    if (legal && displace.total_displace < best_displace.total_displace) {
      for (unsigned int i = 0, ie = qubits.size(); i < ie; ++i) {
        int qubit_id = qubits[i];
        best_x[i] = db.x[qubit_id];
        best_y[i] = db.y[qubit_id];
      }
      best_displace = displace;
    }
  };

  // first round rough legalization with Hannan grid for clusters
  bool small_clusters_flag = true;
  bool blocked_qubits_flag = false;
  roughLegalizeLauncher(db, qubits, fixed_qubits, small_clusters_flag, blocked_qubits_flag);
  auto displace = compute_displace(db, qubits);
  qplacerPrint(
      kINFO, "Qubit displacement (rough) total %g, max %g, weighted total %g, max %g\n",
      displace.total_displace, displace.max_displace,
      displace.total_weighted_displace, displace.max_weighted_displace);

  bool touch_legal = false;
  bool legal = false;

  // start from 2 site_width iterative minimize spacing
  for (int num_spacing=2; num_spacing>=1; num_spacing--){
    lpLegalizeGraphLauncher(db, qubits, fixed_qubits, num_spacing);
    displace = compute_displace(db, qubits);
    qplacerPrint(
        kINFO, "Qubit (lpGraph) num_spacing %d, total %g, max %g, weighted total %g, max %g\n",
        num_spacing,
        displace.total_displace, displace.max_displace,
        displace.total_weighted_displace, displace.max_weighted_displace);
    legal = check_qubit_legality_touch(db, qubits, true);
    if (touch_legal){
      legal = true;
      break;
    } else if (num_spacing == 1 && !touch_legal){
      legal = check_qubit_legality(db, qubits, true);
    }
  }

  qplacerPrint(kINFO, "Align qubits to site and rows\n");
  // align the lower left corner to row and site
  for (unsigned int i = 0, ie = qubits.size(); i < ie; ++i) {
    int node_id = qubits[i];
    db.x[node_id] = db.align2site(db.x[node_id], db.node_size_x[node_id]);
    db.y[node_id] = db.align2row(db.y[node_id], db.node_size_y[node_id]);
  }

  legal = check_qubit_legality(db, qubits, false);
  return legal;
}

QPLACER_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &QPLACER_NAMESPACE::qubit_legalization_forward,
        "Qubit legalization forward");
}
