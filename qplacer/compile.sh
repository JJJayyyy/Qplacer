#!/bin/bash

set -euo pipefail
CURRENT_DIR=$(pwd)
cd ../build
cmake .. -DCMAKE_INSTALL_PREFIX=../qplacer/operators
make
make install
cd ../qplacer/