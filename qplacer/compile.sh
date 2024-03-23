#!/bin/bash

set -euo pipefail
DEST_DIR=$(pwd)

BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi
cd "$BUILD_DIR"

cmake .. -DCMAKE_INSTALL_PREFIX=${DEST_DIR}/operators
make
make install
cd ${DEST_DIR}
