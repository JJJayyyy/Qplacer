import os
import sys
import time
import logging
import matplotlib
matplotlib.use('Agg')
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
operator_module_path = f'{root_dir}/operators'
if operator_module_path not in sys.path:
    sys.path.append(operator_module_path)

from Params import Params
from PlaceDB import PlaceDB
from NonLinearPlace import NonLinearPlace
import operators.qplacement.configure as configure


def place(params):
    """
    Top API to run the entire placement flow.

    Args:
        - params (Params): Placement parameters
    """
    assert (not params.gpu) or configure.compile_configurations["CUDA_FOUND"] == 'TRUE', \
            "CANNOT enable GPU without CUDA compiled"
    np.random.seed(params.random_seed)

    tt = time.time()
    placedb = PlaceDB()
    placedb(params)
    logging.info("reading database takes %.4f seconds" % (time.time() - tt))

    tt = time.time()
    placer = NonLinearPlace(params, placedb)
    logging.info("placement initialization takes %.4f seconds" % (time.time() - tt))
    placer(params, placedb)
    logging.info("placement takes %.4f seconds" % (time.time() - tt))



if __name__ == "__main__":
    logging.root.name = 'QPlacer'
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)
    params = Params()
    params.printWelcome()
    if len(sys.argv) == 1 or '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        params.printHelp()
        exit()
    elif len(sys.argv) != 2:
        logging.error("Input parameters is required in json format")
        params.printHelp()
        exit()

    params.load(sys.argv[1])
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)
    
    tt = time.time()
    place(params)
    