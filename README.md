<h2><p align="center"> Deep learning toolkit-enable Superconducting Quantum Placement </p></h2>


## Build

[CMake](https://cmake.org) is adopted as the makefile system.
To build, go to the root directory.

```
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../qplacer/operators -DPython_EXECUTABLE=$(which python)
make
make install
```
Where `build` is the directory where to compile the code, and `qplacer/operators` is the directory where to install operators.
