from shapely.geometry import Polygon
from collections import defaultdict
import warnings
import logging
import pickle
import os
import re


def create_polygon(size, position):
    (width, height), (x, y) = size, position
    return Polygon([(x, y), (x + width, y), (x + width, y + height), (x, y + height)])


def parse_def_file(def_file_path):
    component_locations = {'QUBIT':{}, 'WIRE_BLK':{}}
    with open(def_file_path, 'r') as file:
        in_components_section = False
        previous_line = None
        for line in file:
            line = line.strip()
            if 'END COMPONENTS' in line:
                in_components_section = False
                break
            elif 'COMPONENTS' in line:
                in_components_section = True
                continue

            if in_components_section:
                if previous_line != None:
                    line = previous_line + ' ' + line
                    previous_line = None
                tokens = line.strip().split()

                if len(tokens) == 0:
                    continue
                elif len(tokens) < 4:
                    previous_line = line
                else:
                    if tokens[2] == 'QUBIT':
                        component_locations['QUBIT'][tokens[1]] = (int(tokens[6]), int(tokens[7]))
                    elif tokens[2] == 'WIRE_BLK':
                        component_locations['WIRE_BLK'][tokens[1]] = (int(tokens[6]), int(tokens[7]))
    return component_locations


def parse_lef_file(lef_file_path, factor):
    with open(lef_file_path, 'r') as file:
        lef_content = file.readlines()

    macros = {}
    macro_name = None
    for line in lef_content:
        tokens = line.strip().split()
        if len(tokens) == 0:
            continue
        if tokens[0] == 'MACRO':
            macro_name = tokens[1]
            macros[macro_name] = {'size': (0, 0), 'pins': {}}
        elif macro_name and tokens[0] == 'SIZE':
            width = float(tokens[1])
            height = float(tokens[3])
            macros[macro_name]['size'] = (int(width*factor), int(height*factor))
        elif macro_name and tokens[0] == 'PIN':
            pin_name = tokens[1]
            macros[macro_name]['pins'][pin_name] = []
        elif macro_name and tokens[0] == 'RECT':
            if 'pins' in macros[macro_name] and pin_name:
                rect = tuple(map(float, tokens[1:5]))
                macros[macro_name]['pins'][pin_name].append(rect)
    return macros


def get_edge_wirelength(freq):
    # L = v_0/2f    v_0 = 1.3e8
    if isinstance(freq, float):
        return 1.3e8*1000/(2*freq)
    elif isinstance(freq, str):
        return 65/float(freq.lower().replace('ghz', '').strip())
    else:
        raise Exception(f"Type {type(freq)} is currently not acceptable")



def modify_def_positions(org_def_file_path, def_file_path, new_positions):
    """
    Modifies the positions of components in a DEF file.

    :param def_file_path: Path to the DEF file.
    :param new_positions: Dictionary with component IDs as keys and new positions as values.
                          Example: {'Q0': (100, 200), 'poly_aaa': (300, 400)}
    """
    with open(org_def_file_path, 'r') as file:
        logging.info(f'Load org def file: {org_def_file_path}')
        lines = file.readlines()

    update_next_line = False
    updated_lines = []
    component_id = ''
    # - Q0 QUBIT
    #   + PLACED ( 0 600 ) N ;
    # ['+', 'PLACED', '(', '0', '400', ')', 'FS', ';']
    for line in lines:
        if update_next_line:
            if '+ PLACED' in line:
                x, y = new_positions[component_id]
                parts = line.split()
                parts[3] = str(x)
                parts[4] = str(y)
                updated_lines.append(' '.join(parts) + '\n')
            update_next_line = False
            continue

        if line.strip().endswith('QUBIT') or line.strip().endswith('WIRE_BLK'):
            component_id = line.split()[1]
            if component_id in new_positions:
                update_next_line = True
            else:
                continue
        updated_lines.append(line)

    with open(def_file_path, 'w') as file:
        logging.info(f'Save latest def file: {def_file_path}')
        file.writelines(updated_lines)

        
def convert_polygons_to_positions(poly_dict):
    """
    Converts a dictionary of polygons to a format suitable for new_positions.

    :param poly_dict: Dictionary with component IDs as keys and shapely.geometry.Polygon objects as values.
                      Example: {'poly_aaa': <Polygon object>, ...}
    :return: Dictionary with component IDs and their centroid coordinates.
    """
    def get_lower_left_corner(polygon):
        min_y = float('inf')
        lower_left = None
        for x, y in polygon.exterior.coords:
            if y < min_y or (y == min_y and x < (lower_left[0] if lower_left else float('inf'))):
                min_y = y
                lower_left = (int(x), int(y))
        assert lower_left != None
        return lower_left
    
    new_positions = {}
    for component_id, polygon in poly_dict.items():
        if isinstance(polygon, Polygon):
            # pos = polygon.centroid
            # new_positions[component_id] = (int(pos.x), int(pos.y))
            new_positions[component_id] = get_lower_left_corner(polygon)
    return new_positions


def get_placer_runtime(log_file_path):
    placement_time_pattern = r"\[INFO\s*\] QPlacer - placement takes ([\d.]+) seconds"
    iterations_pattern = r"\[INFO\s*\] QPlacer - iteration\s+(\d+),\s+wHPWL"

    placement_time, total_iterations = None, None
    with open(log_file_path, 'r') as file:
        for line in file:
            if re.search(placement_time_pattern, line):
                placement_time = re.search(placement_time_pattern, line).group(1)
            if re.search(iterations_pattern, line):
                total_iterations = re.search(iterations_pattern, line).group(1)
    return float(placement_time), int(total_iterations)


def generate_wireblk_polygons(lef_data, def_data, partition):
    wireblk_polygons = {}
    if partition:
        for wireblk_name, position in def_data['WIRE_BLK'].items():
            size = lef_data['WIRE_BLK']['size']
            wireblk_polygon = create_polygon(size, position)
            wireblk_polygons[wireblk_name] = wireblk_polygon
        return wireblk_polygons


def generate_qubit_polygons(lef_data, def_data):
    qubit_polygons = {}
    for qubit_name, position in def_data['QUBIT'].items():
        size = lef_data['QUBIT']['size']
        qubit_polygon = create_polygon(size, position)
        qubit_polygons[qubit_name] = qubit_polygon
    return qubit_polygons


def load_freq_db(topology, topology_setup_dir):
    freq_db_path = f'{topology_setup_dir}/freq_{topology}.pkl' 
    if os.path.isfile(freq_db_path):
        print("Frequency Database: {}".format(freq_db_path))
        with open(freq_db_path, 'rb') as file:
            freq_db = pickle.load(file)
            assert isinstance(freq_db.edge_to_freq_map, dict) 
            assert isinstance(freq_db.qubit_to_freq_map, dict)
            return freq_db
    else:
        return None


def load_testcase(topology, topology_setup_dir, suffix='wp_wf'):
    suffix_dir_path = os.path.join(topology_setup_dir, suffix)
    if os.path.isdir(suffix_dir_path):
        print("=================================")
        print(f'Topology: {topology}, Suffix: {suffix}')
        files = os.listdir(suffix_dir_path)
        testcase_name = f"{topology}_{suffix}"
        testcase = defaultdict()
        for cur_file in files:
            file_path = os.path.join(suffix_dir_path, cur_file)
            print(f'    File: {file_path}')
            if file_path.endswith('.log'):
               testcase["log"] = file_path
            elif file_path.endswith(f'{testcase_name}_params.pkl'):
                with open(file_path, 'rb') as file:
                    testcase['params'] = pickle.load(file)
            elif file_path.endswith(f'{testcase_name}_db.pkl'):
                with open(file_path, 'rb') as file:
                    testcase['db'] = pickle.load(file)
                
        assert all(value is not None for value in testcase.values()), \
            f"Missing pkl/json files for {topology}/{suffix}"

        testcase["lef"] = testcase['params'].file_paths["lef"]
        testcase["def"] = f"results/{testcase_name}/{testcase_name}.gp.def"
        testcase["lg_def"] = f"results/{testcase_name}/{testcase_name}.lg.def"
        testcase["post_def"] = f"test/{topology}/{suffix}/{testcase_name}.post.def"

        for f in ['lef', 'def']:
            print(f'{f.upper()} : {testcase[f]}', end="\t")
            if os.path.isfile(testcase[f]):
                print('')
            else:
                print(f'[MISSED]')
        return testcase
    else:
        warnings.warn(f"{os.path.basename(suffix_dir_path)} is not valid directory", UserWarning)
