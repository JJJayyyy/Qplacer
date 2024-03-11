from shapely.geometry import Polygon


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


