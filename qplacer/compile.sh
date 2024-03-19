#!/bin/bash

set -euo pipefail
CURRENT_DIR=$(pwd)

BUILD_DIR="../build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi
cd "$BUILD_DIR"

cmake .. -DCMAKE_INSTALL_PREFIX=${CURRENT_DIR}/operators
make
make install
cd ${CURRENT_DIR}
# cd ../qplacer/