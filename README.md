<h1><p align="center"> Qplacer: Deep learning toolkit-enable Superconducting Physical Qubit Placement </p></h2>


#### What it is doing
An efficient tool leveraging deep learning techniques to optimize nonlinear superconducting physical qubit layouts. Developed based on [Dreamplace](https://github.com/limbo018/DREAMPlace) with flexible deep learning toolkits, it runs on both CPUs and GPUs for versatile deployment.

#### Who will benefit
Researchers on quantum hardware device design, quantum device manufacture, and quantum design automation.

#### Reference Flow
<p align="center">
    <img src="images/overview.png" width="100%">
</p>

#### Sample Result
<p align="center">
    <img src="images/demo.gif" width="25%">
    <img src="images/eagle.png" width="68%">
</p>


## Publications

- Junyao Zhang, Hanrui Wang, Qi Ding, Jiaqi Gu, Reouven Assouly, William D Oliver, Song Han, Kenneth R Brown, Hai Li, Yiran Chen,
  "**Qplacer: Frequency-Aware Component Placement for Superconducting Quantum Computers**",
  arxiv, 2023
  ([preprint](https://arxiv.org/abs/2401.17450))


## Installation

This project is best built using a [Docker](https://hub.docker.com) container to streamline the process and manage all dependencies efficiently. [CMake](https://cmake.org) is adopted as the makefile system.

1. **Docker Installation:** Begin by installing Docker. Choose the appropriate version for your operating system:
   - [Windows](https://docs.docker.com/docker-for-windows/)
   - [Mac](https://docs.docker.com/docker-for-mac/)
   - [Linux](https://docs.docker.com/install/)

2. **Enabling GPU Features (Optional):** For GPU support, install [NVIDIA-docker](https://github.com/NVIDIA/nvidia-docker). This step can be skipped if GPU features are not needed.

3. **Clone the Repository:** Clone or navigate to the local copy of this repository.

    ```bash
    git clone --recursive https://github.com/JJJayyyy/Qplacer.git
    pip install -r requirements.txt
    ```

4. **Build the Docker Image:** Use the following command to build the Docker image, you can replace `jz420` with your name:

   ```bash
   docker build . --file Dockerfile --tag jz420/qplacer:cuda
   ```

5. **Launching the Container:** Depending on your setup (with or without GPU) and operating system, use one of the following commands to start the Docker container and enter its bash environment.

   - **With GPU on Linux:**
     ```bash
     docker run --gpus 1 -it -v $(pwd):/Qplacer jz420/qplacer:cuda bash
     ```
   
   - **With GPU on Windows:**
     ```bash
     docker run --gpus 1 -it -v /Qplacer jz420/qplacer:cuda bash
     ```

   - **Without GPU on Linux:**
     ```bash
     docker run -it -v $(pwd):/Qplacer jz420/qplacer:cuda bash
     ```

   - **Without GPU on Windows:**
     ```bash
     docker run -it -v /Qplacer jz420/qplacer:cuda bash
     ```


6. **Build:** 
Navigate to the `qplacer` directory. Execute the `compile.sh`
    ```
    ./compile.sh
    ```
    Where `build` is the directory where to compile the code, and `qplacer/operators` is the directory where to install operators.

    `build` directory can be removed after installation if you do not need incremental compilation later. To clean, go to the root directory.

    ```
    rm -r build
    ```
    

## Get Benchmarks

To get quantum topology benchmarks, please refer the `qplacer/qplacer_example.ipynb`. The benchmark files and placement configuration files will be saved in directory `qplacer/benchmark` and `qplacer/test`, respectively


## Run Placer

Before running, make sure the **benchmarks** have been created and the python dependency packages/operators have been installed/built successfully.

Navigate to directory `qplacer` and run `qplacer_engine/Placer.py` with `json` configuration file for full placement.

```
python qplacer_engine/Placer.py test/grid-25/wp_wf/grid-25_wp_wf.json
```

## Contact
Junyao Zhang [Email](mailto:jz420@duke.edu), [Github issue](https://github.com/JJJayyyy/Qplacer/issues)


## Citation
```
@misc{zhang2024qplacer,
      title={Qplacer: Frequency-Aware Component Placement for Superconducting Quantum Computers}, 
      author={Junyao Zhang and Hanrui Wang and Qi Ding and Jiaqi Gu and Reouven Assouly and William D. Oliver and Song Han and Kenneth R. Brown and Hai "Helen" Li and Yiran Chen},
      year={2024},
      eprint={2401.17450},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```