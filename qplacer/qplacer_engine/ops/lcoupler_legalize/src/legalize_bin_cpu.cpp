#include "lcoupler_legalize/src/function_cpu.h"

QPLACER_BEGIN_NAMESPACE


template <typename T>
void updateTouchedBlanksCPU(
    int blank_bin_id_y, 
    Blank<T> new_blank,
    std::vector<std::vector<Blank<T>>>& edge_touch_blanks,
    bool remove
    ) {
    std::vector<Blank<T> >& blanks = edge_touch_blanks.at(blank_bin_id_y);

    if (remove){
        if (blanks.size() > 0){
            T split_xh;
            int remove_bi = -1;
            int loc_bi = -1;
            for (unsigned int bi = 0; bi < blanks.size(); ++bi) {
                Blank<T>& blk = blanks.at(bi);
                if (blk.xl < new_blank.xl && new_blank.xh < blk.xh) {
                    split_xh = blk.xh;
                    blk.xh = new_blank.xl;
                    loc_bi = bi+1;
                    break;
                } else if (blk.xl == new_blank.xl && new_blank.xh == blk.xh) {
                    remove_bi = bi;
                    break; 
                } else if (new_blank.xl == blk.xl) {
                    blk.xl = new_blank.xh;
                    break;
                } else if (new_blank.xh == blk.xh) {
                    blk.xh = new_blank.xl;
                    break;
                }
            }
            if (remove_bi > -1){
                blanks.erase(blanks.begin()+remove_bi);
            }
            if (loc_bi > -1){
                Blank<T> split_blank;
                split_blank.xl = new_blank.xh;
                split_blank.xh = split_xh;
                split_blank.yl = new_blank.yl;
                split_blank.yh = new_blank.yh;
                blanks.insert(blanks.begin()+loc_bi, split_blank);
            }
        }
    } else {
        if (blanks.size() == 0){
            blanks.push_back(new_blank);
        } else {
            int pre_touch_bi = -1;
            int remove_bi = -1;
            int loc_bi = -1;
            for (unsigned int bi = 0; bi < blanks.size(); ++bi) {
                Blank<T>& blk = blanks.at(bi);
                if (blk.xl <= new_blank.xl && new_blank.xh <= blk.xh){
                    remove_bi = -1;
                    loc_bi = -1;
                    break;
                } else if (new_blank.xl == blk.xh) {
                    blk.xh = new_blank.xh;
                    pre_touch_bi = bi;
                    loc_bi=-1;  // reset loc_bi if latter blank modified
                } else if (new_blank.xh == blk.xl){
                    if (pre_touch_bi > -1){     // touch two blanks pre_blk-blk-post_blk -> preblk
                        blanks.at(pre_touch_bi).xh = blk.xh;
                        remove_bi = bi;         // erase post_touch bi
                    } else {
                        blk.xl = new_blank.xl;
                    }
                    loc_bi=-1;  // reset loc_bi if latter blank modified
                    break;      // post touch represents following block will not need to check
                } else if (new_blank.xh < blk.xl){
                    if (pre_touch_bi == -1 && loc_bi== -1){     // only do not touch any blocks
                        loc_bi = bi;
                    } else {
                        break;   // pre_blk-blk| |this blk. so not need to check following blks
                    }
                } else if (new_blank.xl > blk.xh){
                    loc_bi = bi+1;
                } else {
                    std::cout << "new_blank.xl: " << new_blank.xl << ", new_blank.xh: " << new_blank.xh 
                    << " | blk.xl: " << blk.xl << ", blk.xh: " << blk.xh << std::endl;
                    qplacerAssert(false);
                }
            }

            if (remove_bi > -1){
                blanks.erase(blanks.begin()+remove_bi);
            }
            if (loc_bi > -1){
                blanks.insert(blanks.begin()+loc_bi, new_blank);
            }
        }
    }
}


template <typename T>
T checkTouchedEdgeLengthCPU(
    int blank_num_bins_y,
    int best_blank_bin_id_y, 
    T target_xl,
    T width,
    const std::vector<std::vector<Blank<T>>>& edge_places
    ) {
    T touched_edge_length = 0;

    // find edge current/upper/lower 0, +1, -1
    for (int offset_y = 0; abs(offset_y) <= 1; offset_y = (offset_y > 0)? -offset_y : -(offset_y-1)){
        int blank_bin_id_y = best_blank_bin_id_y + offset_y;
        if (blank_bin_id_y < 0 || blank_bin_id_y >= blank_num_bins_y) { 
            continue; 
        }
        const std::vector<Blank<T> >& blanks = edge_places.at(blank_bin_id_y); // edges in this bin 
        if (blanks.size() == 0){
            continue;
        } else {
            for (unsigned int bi = 0; bi < blanks.size(); ++bi) {
                const Blank<T>& blk = blanks.at(bi);
                if (blk.xl <= target_xl && (target_xl+width) <= blk.xh && offset_y !=0){
                    touched_edge_length += width;
                }
                if (offset_y ==0){
                    if (target_xl == blk.xh) {
                       touched_edge_length += (blk.yh - blk.yl);
                    } else if (target_xl+width == blk.xl){
                        touched_edge_length += (blk.yh - blk.yl);
                    }
                }
            }
        }
    }
    return touched_edge_length;
}


template <typename T>
void checkBlanksCPU(
    const std::vector<std::vector<Blank<T> > >& bin_blanks, 
    const char* blanks_name, 
    bool printable) {
    if (printable) std::cout << blanks_name << ": " << std::endl;
    for (const auto& row : bin_blanks) {
        if (row.size() > 0) {
            int xl = -1;
            for (const auto& blank : row) {
                if (printable) std::cout << blank.toString() << " ";
                qplacerAssertMsg(xl <= int(blank.xl), "xl: %d, blank.xl: %d", xl, int(blank.xl));
                qplacerAssertMsg(blank.xl < blank.xh, "blank.xl: %g, blank.xh: %g", blank.xl, blank.xh);
                xl = blank.xh;
            }
            if (printable) std::cout << std::endl;
        }
    }
}


template <typename T>
void legalizeBinCPU(
        const T* init_x, const T* init_y, 
        const T* node_size_x, const T* node_size_y, 
        std::vector<std::vector<Blank<T> > >& bin_blanks, // blanks in each bin, sorted low -> high, left -> right 
        std::vector<std::vector<int> >& bin_cells, // unplaced cells in each bin 
        T* x, T* y, 
        int num_bins_x, int num_bins_y, int blank_num_bins_y, 
        T bin_size_x, T bin_size_y, T blank_bin_size_y, 
        T site_width, T row_height, 
        T xl, T yl, T xh, T yh,
        T alpha,        // a parameter to tune anchor initial locations and current locations 
        T beta,         // a parameter to tune space reserving 
        bool lr_flag,   // from left to right 
        int* num_unplaced_cells,
        const std::vector<std::vector<int>> node_in_group
        ) 
{

    for (int i = 0; i < num_bins_x*num_bins_y; i += 1) {
        int bin_id_x = i/num_bins_y; 
        int bin_id_y = i-bin_id_x*num_bins_y; 
        int blank_num_bins_per_bin = roundDiv(bin_size_y, blank_bin_size_y);
        int blank_bin_id_yl = bin_id_y*blank_num_bins_per_bin;
        int blank_bin_id_yh = std::min(blank_bin_id_yl+blank_num_bins_per_bin, blank_num_bins_y);

        // cells in this bin 
        std::vector<int>& cells = bin_cells.at(i);
        std::vector<int> edge_lengths;
        int total_length = 0;
        int edge_idx = 0;
        int total_bins = bin_cells.at(i).size();
        bool empty_edge_touch_blanks = true;
        int touch_blanks_initial_y = 0;
        int touch_blanks_end_y = blank_num_bins_y;
        std::vector<std::vector<Blank<T>>> edge_touch_blanks (num_bins_x*blank_num_bins_y);
        std::vector<std::vector<Blank<T>>> edge_places (num_bins_x*blank_num_bins_y); // edge place

        if (!lr_flag) {
            std::reverse(cells.begin(), cells.end());
            for (int edge_idx=0; edge_idx<node_in_group.size(); edge_idx++) {
                total_length += node_in_group[edge_idx].size();         
                edge_lengths.push_back(total_length);
            }   
        } else {
            for (int edge_idx=node_in_group.size()-1; edge_idx>=0; edge_idx--) {
                total_length += node_in_group[edge_idx].size();         
                edge_lengths.push_back(total_length);
            }   
        }

        int counter = 0;
        // ci -> cell idx
        for (int ci = bin_cells.at(i).size()-1; ci >= 0; --ci) {
            // legalize a new edge, clean the edge_touch_blanks 
            if (total_bins-1 - ci == edge_lengths[edge_idx]) {     // ci is descending 
                for (auto& inner_vector : edge_touch_blanks) {
                    inner_vector.clear();
                }
                
                for (auto& inner_vector : edge_places) {    // clean the edge place
                    inner_vector.clear();
                }

                edge_idx++;
                empty_edge_touch_blanks = true;
            }

            int node_id = cells.at(ci); 
            // align to site 
            T init_xl = floorDiv(((alpha*init_x[node_id]+(1-alpha)*x[node_id])-xl), site_width)*site_width+xl;
            T init_yl = (alpha*init_y[node_id]+(1-alpha)*y[node_id]);
            T width = ceilDiv(node_size_x[node_id], site_width)*site_width;
            T height = node_size_y[node_id];

            int num_node_rows = ceilDiv(height, row_height); // may take multiple rows, 1 for heghit = row_height
            int blank_index_offset[num_node_rows]; 
            std::fill(blank_index_offset, blank_index_offset+num_node_rows, 0);

            int blank_initial_bin_id_y = floorDiv((init_yl-yl), blank_bin_size_y);
            blank_initial_bin_id_y = std::min(blank_bin_id_yh-1, std::max(blank_bin_id_yl, blank_initial_bin_id_y));
            int blank_bin_id_dist_y = std::max(blank_initial_bin_id_y+1, blank_bin_id_yh-blank_initial_bin_id_y); 

            int best_blank_bin_id_y = -1;
            int best_blank_bi[num_node_rows]; 
            std::fill(best_blank_bi, best_blank_bi+num_node_rows, -1); 

            T best_cost = xh-xl+yh-yl; 
            T best_xl = -1; 
            T best_yl = -1; 

            checkBlanksCPU(bin_blanks, "check bin_blanks", false);

            if (!empty_edge_touch_blanks){
                int blank_bin_id_dist_y = std::max(std::abs(touch_blanks_initial_y - blank_initial_bin_id_y), 
                    std::abs(touch_blanks_end_y - blank_initial_bin_id_y));

                for (int bin_id_offset_y = 0; abs(bin_id_offset_y) <= blank_bin_id_dist_y; 
                bin_id_offset_y = (bin_id_offset_y > 0)? -bin_id_offset_y : -(bin_id_offset_y-1)){
                    int blank_bin_id_y = blank_initial_bin_id_y+bin_id_offset_y;
                    if (blank_bin_id_y < touch_blanks_initial_y || blank_bin_id_y > touch_blanks_end_y) {
                        continue; 
                    }
                    int blank_bin_id = blank_bin_id_y; 
                    const std::vector<Blank<T> >& blanks = edge_touch_blanks.at(blank_bin_id); // edge_touch_blanks in this bin 
                    int row_best_blank_bi[num_node_rows]; 
                    std::fill(row_best_blank_bi, row_best_blank_bi+num_node_rows, -1); 
                    T row_best_cost = xh-xl+yh-yl;
                    T row_best_xl = -1; 
                    T row_best_yl = -1; 
                    bool search_flag = true; 
                    T best_touched_edge_length = 0;

                    // Iterate blanks in a row
                    for (unsigned int bi = 0; search_flag && bi < blanks.size(); ++bi) {
                        const Blank<T>& blank = blanks[bi];

                        // for multi-row height cells, check blanks in upper rows ind blanks with maximum intersection 
                        blank_index_offset[0] = bi; 
                        std::fill(blank_index_offset+1, blank_index_offset+num_node_rows, -1); 
                        Interval<T> intersect_blank (blank.xl, blank.xh); 
                        T intersect_blank_width = intersect_blank.xh-intersect_blank.xl;
                        // If the blank is able to put bin cell
                        if (intersect_blank_width >= width) {
                            T target_xl = init_xl;  // compute displacement 
                            T target_yl = blank.yl; 
                            T beta = 4;             // allow tolerance to avoid more dead space 
                            T tolerance = std::min(beta*width, intersect_blank_width/beta); 
                            if (target_xl <= intersect_blank.xl + tolerance) {
                                target_xl = intersect_blank.xl; 
                            } else if (target_xl+width >= intersect_blank.xh - tolerance) {
                                target_xl = (intersect_blank.xh-width);
                            }
                            T cost = fabs(target_xl-init_xl)+fabs(target_yl-init_yl); 

                            T touched_edge_length = checkTouchedEdgeLengthCPU(blank_num_bins_y, blank_bin_id, target_xl, width, edge_places);

                            // update best_touched_edge_length and best cost 
                            if (touched_edge_length > best_touched_edge_length){
                                std::copy(blank_index_offset, blank_index_offset+num_node_rows, row_best_blank_bi); 
                                row_best_cost = cost; 
                                best_touched_edge_length = touched_edge_length;
                                row_best_xl = target_xl; 
                                row_best_yl = target_yl;
                            } else if (touched_edge_length == best_touched_edge_length){
                                if (cost < row_best_cost) {
                                    std::copy(blank_index_offset, blank_index_offset+num_node_rows, row_best_blank_bi); 
                                    row_best_cost = cost; 
                                    row_best_xl = target_xl; 
                                    row_best_yl = target_yl; 
                                }
                            }

                        }
                    }

                    if (row_best_cost < best_cost) {
                        // covert bi in edge_touch_blanks to bin_blanks
                        std::vector<Blank<T> >& blanks = bin_blanks.at(blank_bin_id_y); 
                        bool converted = false;
                        for (unsigned int bi = 0; bi < blanks.size(); ++bi) {
                            Blank<T>& blk = blanks.at(bi);
                            if (blk.xl <= row_best_xl && row_best_xl+width <= blk.xh) {
                                row_best_blank_bi[0] = bi;
                                converted = true;
                            }
                        }

                        qplacerAssert(converted);

                        best_blank_bin_id_y = blank_bin_id_y; 
                        std::copy(row_best_blank_bi, row_best_blank_bi+num_node_rows, best_blank_bi);
                        best_cost = row_best_cost; 
                        best_xl = row_best_xl; 
                        best_yl = row_best_yl; 
                    } 
                }
            } 
            else {
                /// bin_blanks is 2d vector <row number><chunks number><blank chunk>
                /// e.g, Blank(0.0, 20.0, 50.0, 21.0) 
                ///      Blank(0.0, 21.0, 24.0, 22.0) Blank(27.0, 21.0, 50.0, 22.0) 
                /// Iterate a rows 
                // Iterates over bin IDs in the y-direction. The loop starts from "blank_initial_bin_id_y" 
                // and explores bins above and below it. The iteration pattern is such that it checks the bin, 
                // then one bin up, one bin down, two bins up, two bins down.

                for (int bin_id_offset_y = 0; abs(bin_id_offset_y) < blank_bin_id_dist_y; 
                bin_id_offset_y = (bin_id_offset_y > 0)? -bin_id_offset_y : -(bin_id_offset_y-1)){
    
                    // std::cout << "bin_id_offset_y: " << bin_id_offset_y << std::endl;

                    int blank_bin_id_y = blank_initial_bin_id_y+bin_id_offset_y;
                    if (blank_bin_id_y < blank_bin_id_yl || blank_bin_id_y+num_node_rows > blank_bin_id_yh) {
                        continue; 
                    }
                    int blank_bin_id = bin_id_x*blank_num_bins_y+blank_bin_id_y; 
                    const std::vector<Blank<T> >& blanks = bin_blanks.at(blank_bin_id); // blanks in this bin 
                    int row_best_blank_bi[num_node_rows]; 
                    std::fill(row_best_blank_bi, row_best_blank_bi+num_node_rows, -1); 
                    T row_best_cost = xh-xl+yh-yl;
                    T row_best_xl = -1; 
                    T row_best_yl = -1; 
                    bool search_flag = true; 

                    // Iterate blanks in a row
                    for (unsigned int bi = 0; search_flag && bi < bin_blanks.at(blank_bin_id).size(); ++bi) {
                        const Blank<T>& blank = blanks[bi];

                        // for multi-row height cells, check blanks in upper rows ind blanks with maximum intersection 
                        blank_index_offset[0] = bi; 
                        std::fill(blank_index_offset+1, blank_index_offset+num_node_rows, -1); 
                        while (true) {
                            Interval<T> intersect_blank (blank.xl, blank.xh); 
                            // skip it if num_node_rows = 1, single cell per row
                            for (int row_offset = 1; row_offset < num_node_rows; ++row_offset) {
                                int next_blank_bin_id_y = blank_bin_id_y+row_offset; 
                                int next_blank_bin_id = bin_id_x*blank_num_bins_y+next_blank_bin_id_y; 
                                unsigned int next_bi = blank_index_offset[row_offset]+1; 
                                for (; next_bi < bin_blanks.at(next_blank_bin_id).size(); ++next_bi) {
                                    const Blank<T>& next_blank = bin_blanks.at(next_blank_bin_id)[next_bi];
                                    Interval<T> intersect_blank_tmp = intersect_blank; 
                                    intersect_blank_tmp.intersect(next_blank.xl, next_blank.xh);
                                    if (intersect_blank_tmp.xh-intersect_blank_tmp.xl >= width) {
                                        intersect_blank = intersect_blank_tmp; 
                                        blank_index_offset[row_offset] = next_bi; 
                                        break; 
                                    }
                                }
                                if (next_bi == bin_blanks.at(next_blank_bin_id).size()) {   // not found 
                                    intersect_blank.xl = intersect_blank.xh = 0; 
                                    break; 
                                }
                            }
                            
                            T intersect_blank_width = intersect_blank.xh-intersect_blank.xl;
                            // If the blank is able to put bin cell
                            if (intersect_blank_width >= width) {
                                // compute displacement 
                                T target_xl = init_xl; 
                                T target_yl = blank.yl; 
                                // allow tolerance to avoid more dead space 
                                T beta = 4; 
                                T tolerance = std::min(beta*width, intersect_blank_width/beta); 
                                if (target_xl <= intersect_blank.xl + tolerance) {
                                    target_xl = intersect_blank.xl; 
                                } else if (target_xl+width >= intersect_blank.xh - tolerance) {
                                    target_xl = (intersect_blank.xh-width);
                                }
                                T cost = fabs(target_xl-init_xl)+fabs(target_yl-init_yl); 
                                // std::cout << "target_xl: " << target_xl << ", target_yl: " << target_yl << " | Cost: " << cost << std::endl;
                                
                                // update best cost 
                                if (cost < row_best_cost) {
                                    std::copy(blank_index_offset, blank_index_offset+num_node_rows, row_best_blank_bi); 
                                    row_best_cost = cost; 
                                    row_best_xl = target_xl; 
                                    row_best_yl = target_yl; 
                                } else {  
                                    search_flag = false; // early exit since we iterate within rows from left to right
                                }
                            } else { // not found 
                                break; 
                            }
                            
                            if (num_node_rows < 2) { // for single-row height cells 
                                break; 
                            }
                        }
                    }

                    if (row_best_cost < best_cost) {
                        best_blank_bin_id_y = blank_bin_id_y; 
                        std::copy(row_best_blank_bi, row_best_blank_bi+num_node_rows, best_blank_bi);
                        best_cost = row_best_cost; 
                        best_xl = row_best_xl; 
                        best_yl = row_best_yl; 
                    } else if (best_cost+row_height < bin_id_offset_y*row_height) {
                        break;  // early exit since we iterate from close row to far-away row 
                    }
                }
            }
            
            // found blank  
            if (best_blank_bin_id_y >= 0) {
                counter++;
                x[node_id] = best_xl; 
                y[node_id] = best_yl; 
                // update cell position and blank 
                for (int row_offset = 0; row_offset < num_node_rows; ++row_offset) {
                    qplacerAssert(best_blank_bi[row_offset] >= 0); 
                    // blanks in this bin 
                    int best_blank_bin_id = bin_id_x*blank_num_bins_y+best_blank_bin_id_y+row_offset; 
                    std::vector<Blank<T> >& blanks = bin_blanks.at(best_blank_bin_id);
                    Blank<T>& blank = blanks.at(best_blank_bi[row_offset]); 
                    qplacerAssert(best_xl >= blank.xl && best_xl+width <= blank.xh);
                    qplacerAssert(best_yl+row_height*row_offset == blank.yl);

                    if (best_xl == blank.xl) {  // left side touch
                        // update blank 
                        blank.xl += width; 
                        if (floorDiv((blank.xl-xl), site_width)*site_width != blank.xl-xl) {
                            qplacerPrint(kDEBUG, "1. move node %d from %g to %g, blank (%g, %g)\n", 
                            node_id, x[node_id], blank.xl, blank.xl, blank.xh);
                        }
                        if (blank.xl >= blank.xh) {
                            bin_blanks.at(best_blank_bin_id).erase(bin_blanks.at(best_blank_bin_id).begin()+best_blank_bi[row_offset]);
                        }
                    }
                    else if (best_xl+width == blank.xh) { // right side touch
                        // update blank 
                        blank.xh -= width; 
                        if (floorDiv((blank.xh-xl), site_width)*site_width != blank.xh-xl) {
                            qplacerPrint(kDEBUG, "2. move node %d from %g to %g, blank (%g, %g)\n",
                             node_id, x[node_id], blank.xh-width, blank.xl, blank.xh);
                        }
                        if (blank.xl >= blank.xh) {
                            bin_blanks.at(best_blank_bin_id).erase(bin_blanks.at(best_blank_bin_id).begin()+best_blank_bi[row_offset]);
                        }
                    }
                    else {
                        // inside the blank: need to update current blank and insert one more blank 
                        Blank<T> new_blank; 
                        new_blank.xl = best_xl+width; 
                        new_blank.xh = blank.xh; 
                        new_blank.yl = blank.yl; 
                        new_blank.yh = blank.yh; 
                        blank.xh = best_xl; 
                        if (floorDiv((blank.xl-xl), site_width)*site_width != blank.xl-xl 
                            || floorDiv((blank.xh-xl), site_width)*site_width != blank.xh-xl
                            || floorDiv((new_blank.xl-xl), site_width)*site_width != new_blank.xl-xl
                            || floorDiv((new_blank.xh-xl), site_width)*site_width != new_blank.xh-xl)
                        {
                            qplacerPrint(kDEBUG, "3. move node %d from %g to %g, blank (%g, %g), new_blank (%g, %g)\n",
                             node_id, x[node_id], init_xl, blank.xl, blank.xh, new_blank.xl, new_blank.xh);
                        }
                        bin_blanks.at(best_blank_bin_id).insert(bin_blanks.at(best_blank_bin_id).begin()+best_blank_bi[row_offset]+1, new_blank);
                    }
                }
                bin_cells.at(i).erase(bin_cells.at(i).begin()+ci);  // remove from cells 

                // best blank
                int best_blank_bin_id = bin_id_x*blank_num_bins_y+best_blank_bin_id_y; 
                Blank<T> best_blank;                                     
                best_blank.xl = best_xl; 
                best_blank.xh = best_xl+width; 
                best_blank.yl = best_blank_bin_id; 
                best_blank.yh = best_blank_bin_id + row_height; 

                // add best blank to the edge_places
                updateTouchedBlanksCPU(best_blank_bin_id_y, best_blank, edge_places, false);
                checkBlanksCPU(edge_places, "check edge_places", false);

                // remove best blank if it exist in edge_touch_blanks
                updateTouchedBlanksCPU(best_blank_bin_id_y, best_blank, edge_touch_blanks, true);

                // find touched blanks 0, +1, -1
                for (int bin_id_offset_y = 0; abs(bin_id_offset_y) < 2; 
                bin_id_offset_y = (bin_id_offset_y > 0)? -bin_id_offset_y : -(bin_id_offset_y-1)){
                    int blank_bin_id_y = best_blank_bin_id_y+bin_id_offset_y;
                    if (blank_bin_id_y<blank_bin_id_yl || blank_bin_id_y+num_node_rows>blank_bin_id_yh) {continue;}
                    int blank_bin_id = bin_id_x*blank_num_bins_y+blank_bin_id_y; 
                    const std::vector<Blank<T> >& blanks = bin_blanks.at(blank_bin_id); // blanks in this bin 

                    // Iterate blanks in a row
                    for (unsigned int bi = 0; bi < bin_blanks.at(blank_bin_id).size(); ++bi) {
                        const Blank<T>& blank = blanks[bi];
                        Interval<T> intersect_blank (blank.xl, blank.xh); 
                        T intersect_blank_width = intersect_blank.xh-intersect_blank.xl;
                        // std::cout << "Checking blank " << bi << " with bounds xl,xh (" << blank.xl << "," << blank.xh << ") | "
                            // << blank.toString() << std::endl;

                        if (intersect_blank_width >= width) {   // If the touched blank is able to put bin cell
                            if (bin_id_offset_y == 0){
                                // left touched bin is next to blank.xh, right touch bin is next to blank.xl
                                if (best_xl == blank.xh) {  
                                    Blank<T> touched_blank; 
                                    touched_blank.xl = best_xl-width; 
                                    touched_blank.xh = best_xl; 
                                    touched_blank.yl = blank.yl; 
                                    touched_blank.yh = blank.yh; 
                                    updateTouchedBlanksCPU(blank_bin_id_y, touched_blank, edge_touch_blanks, false);
                                } else if (best_xl+width == blank.xl) {
                                    Blank<T> touched_blank; 
                                    touched_blank.xl = best_xl+width; 
                                    touched_blank.xh = best_xl+width+width; 
                                    touched_blank.yl = blank.yl; 
                                    touched_blank.yh = blank.yh; 
                                    updateTouchedBlanksCPU(blank_bin_id_y, touched_blank, edge_touch_blanks, false);
                                }
                            } else if (bin_id_offset_y == 1 || bin_id_offset_y == -1) {
                                // upper/lower touched bin is inside the blank
                                if (best_xl >= blank.xl && best_xl+width <= blank.xh) {  
                                    // Blank<T> touched_blank = (best_xl, best_xl+width, blank.yl, blank.yh);
                                    Blank<T> touched_blank; 
                                    touched_blank.xl = best_xl; 
                                    touched_blank.xh = best_xl+width; 
                                    touched_blank.yl = blank.yl; 
                                    touched_blank.yh = blank.yh;
                                    updateTouchedBlanksCPU(blank_bin_id_y, touched_blank, edge_touch_blanks, false);
                                }
                            }
                        } else { 
                            break; // not found 
                        }
                    }
                }

                checkBlanksCPU(edge_touch_blanks, "check edge_touch_blanks", false);

                // check whether there are touched blanks 
                bool first_y = true;
                empty_edge_touch_blanks = true;
                for (int row_idx = 0; row_idx < edge_touch_blanks.size(); row_idx++) {      
                    if (!edge_touch_blanks[row_idx].empty()) {
                        empty_edge_touch_blanks = false;
                        if (first_y){
                            touch_blanks_initial_y = row_idx;
                            first_y = false;
                        }
                        touch_blanks_end_y = row_idx;
                    }
                }
            }
        }
        std::cout << "counter: " << counter << ", bin_cells[0].size(): " << bin_cells[0].size() << std::endl; 
        *num_unplaced_cells += bin_cells.at(i).size();
    }
}





void instantiateLegalizeBinCPU(
    const float* init_x, const float* init_y, 
    const float* node_size_x, const float* node_size_y, 
    std::vector<std::vector<Blank<float> > >& bin_blanks, // blanks in each bin, sorted low -> high, left -> right 
    std::vector<std::vector<int> >& bin_cells, // unplaced cells in each bin 
    float* x, float* y, 
    int num_bins_x, int num_bins_y, int blank_num_bins_y, 
    float bin_size_x, float bin_size_y, float blank_bin_size_y, 
    float site_width, float row_height, 
    float xl, float yl, float xh, float yh,
    float alpha, // a parameter to tune anchor initial locations and current locations 
    float beta, // a parameter to tune space reserving 
    bool lr_flag, // from left to right 
    int* num_unplaced_cells,
    const std::vector<std::vector<int>> node_in_group
    ) {
    legalizeBinCPU(
        init_x, init_y, 
        node_size_x, node_size_y, 
        bin_blanks, // blanks in each bin, sorted from low to high, left to right 
        bin_cells, // unplaced cells in each bin 
        x, y, 
        num_bins_x, num_bins_y, blank_num_bins_y, 
        bin_size_x, bin_size_y, blank_bin_size_y, 
        site_width, row_height, 
        xl, yl, xh, yh,
        alpha, 
        beta, 
        lr_flag,  
        num_unplaced_cells, 
        node_in_group
    );
}

void instantiateLegalizeBinCPU(
    const double* init_x, const double* init_y, 
    const double* node_size_x, const double* node_size_y, 
    std::vector<std::vector<Blank<double> > >& bin_blanks, // blanks in each bin, sorted from low to high, left to right 
    std::vector<std::vector<int> >& bin_cells, // unplaced cells in each bin 
    double* x, double* y, 
    int num_bins_x, int num_bins_y, int blank_num_bins_y, 
    double bin_size_x, double bin_size_y, double blank_bin_size_y, 
    double site_width, double row_height, 
    double xl, double yl, double xh, double yh,
    double alpha, // a parameter to tune anchor initial locations and current locations 
    double beta, // a parameter to tune space reserving 
    bool lr_flag, // from left to right 
    int* num_unplaced_cells,
    const std::vector<std::vector<int>> node_in_group
    ) {
    legalizeBinCPU(
        init_x, init_y, 
        node_size_x, node_size_y, 
        bin_blanks, // blanks in each bin, sorted from low to high, left to right 
        bin_cells, // unplaced cells in each bin 
        x, y, 
        num_bins_x, num_bins_y, blank_num_bins_y, 
        bin_size_x, bin_size_y, blank_bin_size_y, 
        site_width, row_height, 
        xl, yl, xh, yh,
        alpha, 
        beta, 
        lr_flag, 
        num_unplaced_cells,
        node_in_group
    );
}

QPLACER_END_NAMESPACE
