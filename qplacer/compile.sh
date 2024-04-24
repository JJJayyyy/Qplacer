#!/bin/bash

set -euo pipefail

DEST_DIR=$(pwd)
BUILD_DIR="../build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

if [ $# -eq 0 ]; then
    echo "No INSTALL_BACKUP option provided, defaulting to ON"
    INSTALL_BACKUP="ON"
else
    INSTALL_BACKUP=$1
fi

cmake .. -DCMAKE_INSTALL_PREFIX="${DEST_DIR}/operators" -DINSTALL_BACKUP=${INSTALL_BACKUP}
make
make install

cd "${DEST_DIR}"
