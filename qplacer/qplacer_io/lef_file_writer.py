from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.affinity import translate
import matplotlib.pyplot as plt
import logging


class LefFileWriter:
    def __init__(self, 
                 params,
                 ):
        self.microns = params.scale_factor
        self.partition = params.partition
        if self.partition:
            self.coresite_size = params.partition_size
        else:
            self.coresite_size = params.qubit_size
        self.edge_macro_map = dict()



    def __call__(self, file_path, db, debugging=False):
        qubit_lef_data = db.qubit_geo_data['Q1']
        wire_lef_data = db.wireblk_polygon_map
        lef_str = self.lef_header(size_x=self.coresite_size, size_y=self.coresite_size)
        lef_str += self.format_qubit_to_lef('QUBIT', qubit_lef_data, debugging=debugging)
        lef_str += self.format_wireblk_to_lef(wire_lef_data, debugging=debugging)
        lef_str += self.lef_footer()
        with open(file_path, 'w') as file:
            file.write(lef_str)
            logging.info(f"Content written to {file_path}")
        return self.edge_macro_map


    def format_qubit_to_lef(self, name, qubit_data, buffer_width=0.01, debugging=False):
        qubit_name = name
        org_qubit_geo: Polygon = qubit_data["geometry"]
        org_q_minx, org_q_miny, org_q_maxx, org_q_maxy = org_qubit_geo.bounds
        qubit_geo = translate(org_qubit_geo, xoff=-org_q_minx, yoff=-org_q_miny)    # translate the location
        q_minx, q_miny, q_maxx, q_maxy = qubit_geo.bounds

        # debugging setup
        if debugging:
            fig = plt.figure(figsize = (10, 5))
            fig.tight_layout()
            font = 10
            x, y = qubit_geo.exterior.xy
            plt.fill(x, y, alpha=0.5, fc='blue', label=qubit_name)
            for coord in qubit_geo.exterior.coords:
                h_align = 'center'
                v_align = 'top' if coord[1] == q_miny else 'bottom'
                plt.text(coord[0], coord[1], f'({coord[0]:.2f}, {coord[1]:.2f})', fontsize=font, ha=h_align, va=v_align)
        
        # Generate DEF cell
        lef_str  = f"MACRO {qubit_name}\n"
        lef_str +=  "    CLASS CORE ;\n"
        lef_str += f"    FOREIGN {qubit_name} 0.000000 0.000000 ;\n"
        lef_str +=  "    ORIGIN 0.000000 0.000000 ;\n"
        lef_str += f"    SIZE {(q_maxx - q_minx):.6f} BY {(q_maxy - q_miny):.6f} ;\n"
        lef_str +=  "    SYMMETRY X Y ;\n"
        # lef_str +=  "    SITE CoreSite ;\n"

        for pin_name, org_pin_geo in qubit_data['pins'].items():
            pin_geo = translate(org_pin_geo, xoff=-org_q_minx, yoff=-org_q_miny)
            pin_geo_poly: Polygon = pin_geo.buffer(buffer_width).intersection(qubit_geo)  # Clip to qubit bounds
            if isinstance(pin_geo_poly, MultiPolygon):
                unioned_poly = MultiPolygon(list(pin_geo_poly)).convex_hull # Convert multi-polygons into one bounding box
            else:
                unioned_poly = pin_geo_poly

            p_minx, p_miny, p_maxx, p_maxy = unioned_poly.bounds
                    
            if debugging:
                corners = [(p_minx, p_miny), (p_minx, p_maxy), (p_maxx, p_maxy), (p_maxx, p_miny), (p_minx, p_miny)]
                xs, ys = zip(*corners)
                plt.plot(xs, ys, 'o-', label=f'Pin Box: {pin_name}')
                for coord in corners[:-1]:  # discard the find item in corners which is only for enclose the box
                    # h_align = 'right' if coord[0] == p_minx else 'left' 
                    h_align = 'center'
                    v_align = 'top' if coord[1] == p_miny else 'bottom' 
                    plt.text(coord[0], coord[1], f'({coord[0]:.2f}, {coord[1]:.2f})', fontsize=font, ha=h_align, va=v_align)
                x, y = pin_geo_poly.exterior.xy
                plt.plot(x, y, label=f'Pin: {pin_name}')

            lef_str += f"    PIN {pin_name}\n"
            lef_str +=  "        DIRECTION INPUT ;\n"  # Assumes all pins are INPUT
            lef_str +=  "        USE SIGNAL ;\n"
            lef_str +=  "        PORT\n"
            lef_str +=  "        LAYER Metal1 ;\n"
            lef_str += f"        RECT {(p_minx - q_minx):.6f} {(p_miny - q_miny):.6f} {(p_maxx - q_minx):.6f} {(p_maxy - q_miny):.6f} ;\n"
            lef_str +=  "        END\n"
            lef_str += f"    END {pin_name}\n"
        lef_str += f"END {qubit_name}\n\n"

        if debugging:
            t_font = 12
            plt.xlabel('x-coordinate', fontsize=t_font, fontweight="bold")
            plt.ylabel('y-coordinate', fontsize=t_font, fontweight="bold")
            plt.xticks(fontsize=t_font, fontweight="bold")
            plt.yticks(fontsize=t_font, fontweight="bold")
            plt.title('Qubit and Pins Visualization', fontsize=t_font, fontweight="bold")
            legend_props = {'weight':'bold', 'size':t_font}
            plt.legend(title=f"xoff={-org_q_minx:.2f}, yoff={-org_q_miny:.2f}", prop=legend_props, loc='center')
            plt.grid(True)
            plt.show()
        
        return lef_str



    def format_wireblk_to_lef(self, wireblk_polygon_dict, name='WIRE_BLK', debugging=False):
        if self.partition:
            org_wireblk_geo = wireblk_polygon_dict["wireblk"]
            self.edge_macro_map["wireblk"] = name
            org_minx, org_miny, org_maxx, org_maxy = org_wireblk_geo.bounds
            wireblk_geo = translate(org_wireblk_geo, xoff=-org_minx, yoff=-org_miny)
            minx, miny, maxx, maxy = wireblk_geo.bounds

            if debugging:
                fig = plt.figure(figsize = (10, 5))
                fig.tight_layout()
                font = 8
                x, y = wireblk_geo.exterior.xy
                plt.fill(x, y, alpha=0.5, fc='blue', label=name)
                for coord in wireblk_geo.exterior.coords:
                    h_align = 'center'
                    v_align = 'top' if coord[1] == miny else 'bottom'
                    plt.text(coord[0], coord[1], f'({coord[0]:.2f}, {coord[1]:.2f})', fontsize=font, ha=h_align, va=v_align)

            width, height = (maxx - minx), (maxy - miny)
            p_width, p_height = width / 10, height / 10
            in_p_x, in_p_y = minx, miny + p_height
            out_p_x, out_p_y = (width - p_width), (height - p_height * 2)
            pin_geo = {'IN': Polygon([(in_p_x, in_p_y), 
                                    (in_p_x + p_width, in_p_y), 
                                    (in_p_x + p_width, in_p_y + p_height),
                                    (in_p_x, in_p_y + p_height), 
                                    ]),
                        'OUT': Polygon([(out_p_x, out_p_y), 
                                        (out_p_x + p_width, out_p_y), 
                                        (out_p_x + p_width, out_p_y + p_height),
                                        (out_p_x, out_p_y + p_height), 
                                    ]),
                }
            lef_str  = f"MACRO {name}\n"
            lef_str +=  "    CLASS CORE ;\n"
            lef_str += f"    FOREIGN {name} 0.000000 0.000000 ;\n"
            lef_str +=  "    ORIGIN 0.000000 0.000000 ;\n"
            lef_str += f"    SIZE {round(width, 3):.6f} BY {round(height, 3):.6f} ;\n"
            lef_str +=  "    SYMMETRY X Y ;\n"
            lef_str +=  "    SITE CoreSite ;\n"

            for pin_name, geo in pin_geo.items():
                p_minx, p_miny, p_maxx, p_maxy = geo.bounds
                if debugging:
                    x, y = geo.exterior.xy
                    plt.fill(x, y, alpha=0.5, fc='green', label=pin_name)
                    for coord in geo.exterior.coords[:-1]:
                        h_align = 'center'
                        v_align = 'top' if coord[1] == p_miny else 'bottom'
                        plt.text(coord[0], coord[1], f'({coord[0]:.2f}, {coord[1]:.2f})', fontsize=font, ha=h_align, va=v_align)

                lef_str += f"    PIN {pin_name}\n"
                lef_str += f"        DIRECTION {pin_name}PUT ;\n"
                lef_str +=  "        USE SIGNAL ;\n"
                lef_str +=  "        PORT\n"
                lef_str +=  "        LAYER Metal1 ;\n"
                lef_str += f"        RECT {(p_minx):.6f} {(p_miny):.6f} {(p_maxx):.6f} {(p_maxy):.6f} ;\n"
                lef_str +=  "        END\n"
                lef_str += f"    END {pin_name}\n"
            lef_str += f"END {name}\n\n"

            if debugging:
                t_font = 12
                plt.xlabel('x-coordinate', fontsize=t_font, fontweight="bold")
                plt.ylabel('y-coordinate', fontsize=t_font, fontweight="bold")
                plt.xticks(fontsize=t_font, fontweight="bold")
                plt.yticks(fontsize=t_font, fontweight="bold")
                plt.title('Qubit and Pins Visualization', fontsize=t_font, fontweight="bold")
                legend_props = {'weight':'bold', 'size':t_font}
                plt.legend(title=f"xoff={-org_minx:.2f}, yoff={-org_miny:.2f}", prop=legend_props, loc='center')
                plt.grid(True)
                plt.show()

            return lef_str
        
        else:
            lef_str = ""
            for edge, wireblk_geo in wireblk_polygon_dict.items():
                edge_name = '_'.join(edge)
                macro_name = f"{name}_{edge_name}"
                self.edge_macro_map[edge] = macro_name
                org_minx, org_miny, org_maxx, org_maxy = wireblk_geo.bounds
                wireblk_geo = translate(wireblk_geo, xoff=-org_minx, yoff=-org_miny)
                minx, miny, maxx, maxy = wireblk_geo.bounds

                if debugging:
                    fig = plt.figure(figsize = (10, 5))
                    fig.tight_layout()
                    font = 8
                    x, y = wireblk_geo.exterior.xy
                    plt.fill(x, y, alpha=0.5, fc='blue', label=macro_name)
                    for coord in wireblk_geo.exterior.coords:
                        h_align = 'center'
                        v_align = 'top' if coord[1] == miny else 'bottom'
                        plt.text(coord[0], coord[1], f'({coord[0]:.2f}, {coord[1]:.2f})', fontsize=font, ha=h_align, va=v_align)

                width, height = (maxx - minx), (maxy - miny)
                p_width, p_height = width / 10, height / 10
                in_p_x, in_p_y = minx, miny + p_height
                out_p_x, out_p_y = (width - p_width), (height - p_height * 2)
                pin_geo = {'IN': Polygon([(in_p_x, in_p_y), 
                                        (in_p_x + p_width, in_p_y), 
                                        (in_p_x + p_width, in_p_y + p_height),
                                        (in_p_x, in_p_y + p_height), 
                                        ]),
                            'OUT': Polygon([(out_p_x, out_p_y), 
                                            (out_p_x + p_width, out_p_y), 
                                            (out_p_x + p_width, out_p_y + p_height),
                                            (out_p_x, out_p_y + p_height), 
                                        ]),
                        }
                lef_str += f"MACRO {macro_name}\n"
                lef_str +=  "    CLASS CORE ;\n"
                lef_str += f"    FOREIGN {macro_name} 0.000000 0.000000 ;\n"
                lef_str +=  "    ORIGIN 0.000000 0.000000 ;\n"
                lef_str += f"    SIZE {round(width, 3):.6f} BY {round(height, 3):.6f} ;\n"
                lef_str +=  "    SYMMETRY X Y ;\n"
                lef_str +=  "    SITE CoreSite ;\n"

                for pin_name, geo in pin_geo.items():
                    p_minx, p_miny, p_maxx, p_maxy = geo.bounds
                    if debugging:
                        x, y = geo.exterior.xy
                        plt.fill(x, y, alpha=0.5, fc='green', label=pin_name)
                        for coord in geo.exterior.coords[:-1]:
                            h_align = 'center'
                            v_align = 'top' if coord[1] == p_miny else 'bottom'
                            plt.text(coord[0], coord[1], f'({coord[0]:.2f}, {coord[1]:.2f})', fontsize=font, ha=h_align, va=v_align)

                    lef_str += f"    PIN {pin_name}\n"
                    lef_str += f"        DIRECTION {pin_name}PUT ;\n"
                    lef_str +=  "        USE SIGNAL ;\n"
                    lef_str +=  "        PORT\n"
                    lef_str +=  "        LAYER Metal1 ;\n"
                    lef_str += f"        RECT {(p_minx):.6f} {(p_miny):.6f} {(p_maxx):.6f} {(p_maxy):.6f} ;\n"
                    lef_str +=  "        END\n"
                    lef_str += f"    END {pin_name}\n"
                lef_str += f"END {macro_name}\n\n"

                if debugging:
                    t_font = 12
                    plt.xlabel('x-coordinate', fontsize=t_font, fontweight="bold")
                    plt.ylabel('y-coordinate', fontsize=t_font, fontweight="bold")
                    plt.xticks(fontsize=t_font, fontweight="bold")
                    plt.yticks(fontsize=t_font, fontweight="bold")
                    plt.title('Qubit and Pins Visualization', fontsize=t_font, fontweight="bold")
                    legend_props = {'weight':'bold', 'size':t_font}
                    plt.legend(title=f"xoff={-org_minx:.2f}, yoff={-org_miny:.2f}", prop=legend_props, loc='center')
                    plt.grid(True)
                    plt.show()
                    
            return lef_str



    def lef_header(self, size_x=0.2, size_y=0.2):
        lef_str  =  'VERSION 5.8 ;\n\nBUSBITCHARS "[]" ;\n\nDIVIDERCHAR "/" ;\n\n'
        lef_str += f"UNITS\n    DATABASE MICRONS {self.microns} ;\nEND UNITS\n\n"
        lef_str +=  "MANUFACTURINGGRID 0.000500 ;\n\nCLEARANCEMEASURE EUCLIDEAN ;\nUSEMINSPACING OBS ON ;\n\n"
        lef_str += f"SITE CoreSite\n    CLASS CORE ;\n    SIZE {size_x:.3f} BY {size_y:.3f} ;\nEND CoreSite\n\n"
        return lef_str
        

    def lef_footer(self):
        lef_str = "END LIBRARY\n"
        return lef_str