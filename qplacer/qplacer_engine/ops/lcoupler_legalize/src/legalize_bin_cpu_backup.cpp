// #include "function_cpu.h"
#include "lcoupler_legalize/src/function_cpu.h"

DREAMPLACE_BEGIN_NAMESPACE


template <typename T>
void updateTouchedBlanksCPU(
    int blank_bin_id_y, 
    Blank<T>new_blank,
    std::vector<std::vector<Blank<T>>>& edge_touch_blanks,
    bool remove) {

    std::vector<Blank<T> >& blanks = edge_touch_blanks.at(blank_bin_id_y);

    if (remove){
        // std::cout << "remove: " << new_blank.toString() << std::endl;
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
                    // std::cout << "middle blk: " << blk.toString() << std::endl;
                    // std::cout << "remove remove_bi: " << remove_bi << ", loc_bi: " << loc_bi << std::endl;
                    break;
                } else if (blk.xl == new_blank.xl && new_blank.xh == blk.xh) {
                    remove_bi = bi;
                    // std::cout << "identical blk: " << blk.toString() << std::endl;
                    break; 
                } else if (new_blank.xl == blk.xl) {
                    blk.xl = new_blank.xh;
                    // std::cout << "low blk: " << blk.toString() << std::endl;
                    break;
                } else if (new_blank.xh == blk.xh) {
                    blk.xh = new_blank.xl;
                    // std::cout << "high blk: " << blk.toString() << std::endl;
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
        // std::cout << "Add: " << new_blank.toString() << std::endl;
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
                    // std::cout << "add change xh: " << blk.toString() << std::endl;
                    blk.xh = new_blank.xh;
                    pre_touch_bi = bi;
                    loc_bi=-1;  // reset loc_bi if latter blank modified
                } else if (new_blank.xh == blk.xl){
                    // std::cout << "add change xl: " << blk.toString() << std::endl;
                    if (pre_touch_bi > -1){     // touch two blanks pre_blk-blk-post_blk -> preblk
                        blanks.at(pre_touch_bi).xh = blk.xh;
                        remove_bi = bi;         // erase post_touch bi
                    } else {
                        blk.xl = new_blank.xl;
                    }
                    loc_bi=-1;  // reset loc_bi if latter blank modified
                } else if (new_blank.xl < blk.xl){
                    // std::cout << "add left: " << blk.toString() << std::endl;
                    loc_bi = bi;
                } else if (new_blank.xh > blk.xh){
                    // std::cout << "add right: " << blk.toString() << std::endl;
                    loc_bi = bi+1;
                } else {
                    std::cout << "new_blank.xl: " << new_blank.xl << ", new_blank.xh: " << new_blank.xh 
                    << " | blk.xl: " << blk.xl << ", blk.xh: " << blk.xh << std::endl;
                    dreamplaceAssert(false);
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
void checkBlanksCPU(
    const std::vector<std::vector<Blank<T> > >& bin_blanks, 
    const char* blanks_name, 
    bool printable) {
    if (printable) std::cout << blanks_name << ": " << std::endl;
    for (const auto& row : bin_blanks) {
        int xl = -1;
        for (const auto& blank : row) {
            dreamplaceAssert(xl <= blank.xl);
            xl = blank.xh;
            if (printable) std::cout << blank.toString() << " ";
        }
        if (printable) std::cout << std::endl;
    }

    // std::cout << "edge_touch_blanks" << std::endl;
    // for (const auto& row : edge_touch_blanks) {
    //     for (const auto& blank : row) std::cout << blank.toString() << " ";
    //     std::cout << std::endl;
    // }
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
    std::vector<std::vector<Blank<T>>> edge_touch_blanks (num_bins_x*blank_num_bins_y);

    for (int i = 0; i < num_bins_x*num_bins_y; i += 1) {
        int bin_id_x = i/num_bins_y; 
        int bin_id_y = i-bin_id_x*num_bins_y; 
        int blank_num_bins_per_bin = roundDiv(bin_size_y, blank_bin_size_y);
        int blank_bin_id_yl = bin_id_y*blank_num_bins_per_bin;
        int blank_bin_id_yh = std::min(blank_bin_id_yl+blank_num_bins_per_bin, blank_num_bins_y);

        // cells in this bin 
        std::vector<int>& cells = bin_cells.at(i);

        std::cout << "cells : "; for (const auto& c : cells) std::cout << c << " "; std::cout << std::endl;

        if (!lr_flag) std::reverse(cells.begin(), cells.end());
        // ci -> cell idx
        for (int ci = bin_cells.at(i).size()-1; ci >= 0; --ci) {
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

            std::cout << std::endl << "node_id: " << node_id << ", ci: " << ci << std::endl;
            std::cout << "init_xl: " << init_xl << ", init_yl: " << init_yl << std::endl;
            std::cout << "blank_bin_id_dist_y: " << blank_bin_id_dist_y << ", blank_initial_bin_id_y: " << blank_initial_bin_id_y << std::endl;
            // Iterates over bin IDs in the y-direction. The loop starts from "blank_initial_bin_id_y" 
            // and explores bins above and below it. The iteration pattern is such that it checks the bin, 
            // then one bin up, one bin down, two bins up, two bins down.

            checkBlanksCPU(bin_blanks, "check bin_blanks", true);
            
            /// bin_blanks is 2d vector <row number><chunks number><blank chunk>
            /// e.g, Blank(0.0, 20.0, 50.0, 21.0) 
            ///      Blank(0.0, 21.0, 24.0, 22.0) Blank(27.0, 21.0, 50.0, 22.0) 
            /// Iterate a row 
            for (int bin_id_offset_y = 0; abs(bin_id_offset_y) < blank_bin_id_dist_y; 
            bin_id_offset_y = (bin_id_offset_y > 0)? -bin_id_offset_y : -(bin_id_offset_y-1)){
                
                std::cout << "bin_id_offset_y: " << bin_id_offset_y << std::endl;
                
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

                // std::cout << "bin_blanks.at( " << blank_bin_id << " ).size() : " << bin_blanks.at(blank_bin_id).size() << std::endl;
                
                // Iterate blanks in a row
                for (unsigned int bi = 0; search_flag && bi < bin_blanks.at(blank_bin_id).size(); ++bi) {
                    const Blank<T>& blank = blanks[bi];
                    // std::cout << "Checking blank " << bi << " with xl, xh (" << blank.xl << ", " << blank.xh << ")"
                    // << "blank: " << blank.toString() << std::endl;

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

                            std::cout << "target_xl: " << target_xl << ", target_yl: " << target_yl << " | Cost: " << cost << std::endl;

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

            // found blank  
            if (best_blank_bin_id_y >= 0) {
                std::cout << "found: " << best_blank_bin_id_y << ", best_xl: " << best_xl << ", best_yl: " << best_yl << std::endl;
                x[node_id] = best_xl; 
                y[node_id] = best_yl; 
                // update cell position and blank 
                for (int row_offset = 0; row_offset < num_node_rows; ++row_offset) {
                    dreamplaceAssert(best_blank_bi[row_offset] >= 0); 
                    // blanks in this bin 
                    int best_blank_bin_id = bin_id_x*blank_num_bins_y+best_blank_bin_id_y+row_offset; 
                    std::vector<Blank<T> >& blanks = bin_blanks.at(best_blank_bin_id);
                    Blank<T>& blank = blanks.at(best_blank_bi[row_offset]); 
                    dreamplaceAssert(best_xl >= blank.xl && best_xl+width <= blank.xh);
                    dreamplaceAssert(best_yl+row_height*row_offset == blank.yl);
                    if (best_xl == blank.xl) {  // left side touch
                        // update blank 
                        blank.xl += width; 
                        if (floorDiv((blank.xl-xl), site_width)*site_width != blank.xl-xl) {
                            dreamplacePrint(kDEBUG, "1. move node %d from %g to %g, blank (%g, %g)\n", 
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
                            dreamplacePrint(kDEBUG, "2. move node %d from %g to %g, blank (%g, %g)\n",
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
                            dreamplacePrint(kDEBUG, "3. move node %d from %g to %g, blank (%g, %g), new_blank (%g, %g)\n",
                             node_id, x[node_id], init_xl, blank.xl, blank.xh, new_blank.xl, new_blank.xh);
                        }
                        bin_blanks.at(best_blank_bin_id).insert(bin_blanks.at(best_blank_bin_id).begin()+best_blank_bi[row_offset]+1, new_blank);
                    }
                }
                // remove from cells 
                bin_cells.at(i).erase(bin_cells.at(i).begin()+ci);



                // remove best blank if it exist in edge_touch_blanks
                std::cout << "------" << std::endl;
                int row_offset = 0;
                int best_blank_bin_id = bin_id_x*blank_num_bins_y+best_blank_bin_id_y+row_offset; 
                std::vector<Blank<T> >& blanks = bin_blanks.at(best_blank_bin_id);
                Blank<T> best_blank = blanks.at(best_blank_bi[row_offset]);
                best_blank.xl = best_xl; 
                best_blank.xh = best_xl+width; 
                updateTouchedBlanksCPU(best_blank_bin_id_y, best_blank, edge_touch_blanks, true);

                // find touched blanks 0, +1, -1
                for (int bin_id_offset_y = 0; abs(bin_id_offset_y) < 2; 
                bin_id_offset_y = (bin_id_offset_y > 0)? -bin_id_offset_y : -(bin_id_offset_y-1)){
                    int blank_bin_id_y = best_blank_bin_id_y+bin_id_offset_y;
                    if (blank_bin_id_y<blank_bin_id_yl || blank_bin_id_y+num_node_rows>blank_bin_id_yh) {continue;}
                    int blank_bin_id = bin_id_x*blank_num_bins_y+blank_bin_id_y; 
                    const std::vector<Blank<T> >& blanks = bin_blanks.at(blank_bin_id); // blanks in this bin 
                    // std::cout << "id_offset_y: " << bin_id_offset_y << ", touched row: " << blank_bin_id_y
                    //     << " | bin_blanks.at( " << blank_bin_id << " ).size() : " 
                    //     << bin_blanks.at(blank_bin_id).size() << std::endl;
                    // Iterate blanks in a row
                    for (unsigned int bi = 0; bi < bin_blanks.at(blank_bin_id).size(); ++bi) {
                        const Blank<T>& blank = blanks[bi];
                        Interval<T> intersect_blank (blank.xl, blank.xh); 
                        T intersect_blank_width = intersect_blank.xh-intersect_blank.xl;
                        // std::cout << "Checking blank " << bi << " with bounds xl,xh (" << blank.xl << "," << blank.xh << ") | "
                        //     << blank.toString() << std::endl;
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
                                } else if (best_xl+width == blank.xl){
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
                // std::cout << "edge_touch_blanks" << std::endl;
                // for (const auto& row : edge_touch_blanks) {
                //     for (const auto& blank : row) std::cout << blank.toString() << " ";
                //     std::cout << std::endl;
                // }
                checkBlanksCPU(edge_touch_blanks, "check edge_touch_blanks", true);
                std::cout << "------" << std::endl;
            }
        }
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
    int* num_unplaced_cells 
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
        num_unplaced_cells 
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
    int* num_unplaced_cells 
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
        num_unplaced_cells 
    );
}

DREAMPLACE_END_NAMESPACE
