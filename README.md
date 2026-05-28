# Radio Map Prediction Benchmarking

A deep learning benchmark repository for evaluating and comparing different network architectures on radio map prediction tasks.

## Table of Contents
1. [About The Project](#1-about-the-project)
2. [Roadmap / TODO List](#2-roadmap--todo-list)
3. [Environment Setup](#3-environment-setup)
   - [Option A: Local Setup (Conda)](#option-a-local-setup-conda)
   - [Option B: Cluster Setup (Apptainer / Singularity)](#option-b-cluster-setup-apptainer--singularity)
4. [Data Preparation](#4-data-preparation)
   - [Download Data](#download-data)
   - [Directory Structure](#directory-structure)
5. [Configuration Guide](#5-configuration-guide)
   - [Configuration Files Management](#configuration-files-management)
   - [Key Parameter Groups](#key-parameter-groups)
   - [Configuration Validation](#configuration-validation)
6. [How to Run](#6-how-to-run)
   - [Run on a Local Machine](#run-on-a-local-machine)
   - [Run on the Telecom Paris Cluster](#run-on-the-telecom-paris-cluster)

## 1. About The Project

This project is designed to evaluate and compare the performance of different deep learning network architectures for the **radio map prediction** task. 

This repository is built and modified based on the original [RadioUNet](https://github.com/RonLevie/RadioUNet) project.

### Key Modifications & Improvements

Compared to the original RadioUNet repository, we made several important updates to improve code structure, training flexibility, and timing accuracy:

* **Dataset Splitting:** The original project splits data manually. We added an automatic random splitting feature based on the ratios defined in the configuration file. The maps are randomly split into training, validation, and test sets.
* **Transmitter (TX) Settings:** The number of transmitters for training and validation can be changed via the configuration file. For the test set, the number of transmitters is fixed to 2 because the IRT4 simulation supports a maximum of 2 transmitters.
* **Dynamic Sampling:** In the original project, the sampled points on the same map were fixed. We removed this limit. Sampling points now change between different training epochs, which helps the model adapt better to the simulation results.
* **Term Meanings:** Please note that `mask` in this project has the same physical meaning as `sparse_samples` in the original project. Similarly, `samples_gain` has the same meaning as `input_samples`.
* **Simplified Features:** We removed the feature of randomly selecting the number of samples because it is not necessary for our research goal.
* **Model Architecture Clean-up:** In the model, we removed the `MaxPool2d` layers where `kernel_size=1` and `stride=1` because they do not change the output data. 
* **Code Decoupling & Refactoring:** We refactored some original project's model's **Functions** into **Classes** to make the code cleaner and easier to manage.
  * The original RadioUNet uses a two-stage (double-layer) Unet architecture. We separated the first layer of Unet into an independent model. This decouples the network and makes the training and testing pipeline more standard, while still maintaining the double-layer comparison framework.
  * The parameters for convolutional and deconvolutional layers are now calculated automatically, keeping the exact same performance with the original project.
* **Testing & Evaluation:** 
  * **Global Averaging:** During testing, we calculate the average results globally. The original project averages each batch first and then takes the global average. Mathematically, these two methods are equivalent.
  * **IRT4 Focus:** This project only supports accuracy testing on the IRT4 simulation results (which represents realistic measurements), as other simulations are not needed as testing standards for our current goals.
* **Cars Information:** In original project, we've already known that cars presence affects model performance. So, in this project, if cars exist in the simulation, the cars' data will be fed into the model as an extra feature channel. You can turn this on or off in the configuration file using `cars_exist: true/false`.
* **Accurate Inference Timing:** We added `torch.cuda.synchronize()` for GPU devices. This ensures that the recorded inference time is precise when running on CUDA. (The code still supports running on CPU).

## 2. Roadmap / TODO List

Here are the planned features and updates for this project. Feel free to open an issue if you want to contribute!

* **Configuration & CLI Improvements**
  - [ ] Move model architecture switching from hard-coded Python variables into the configuration file.
  - [ ] Add command-line arguments (CLI) support (e.g., `python main.py --config config/config.yaml --mode train`).

* **Model Architectures Exploration**
  - [ ] Integrate **Transformer-based** architectures for radio map prediction.

## 3. Environment Setup

You can set up the running environment in two ways: using Conda (for local/GPU/CPU runs) or using Apptainer (for the Telecom Paris cluster).

### Option A: Local Setup (Conda)
If you run the code locally, you can use the provided `environment.yml` file. 
* **Note:** This file contains all necessary libraries. If you want to use a GPU, please make sure you have CUDA 13.0.0 or above installed on your system.

To create the environment, run:
```bash
conda env create -f environment.yml
```

*The virtual environment name is **rmp**.*

### Option B: Cluster Setup (Apptainer / Singularity)

If you run experiments on the **Telecom Paris cluster** ([Website](https://computing.telecom-paris.fr/)), you can use the `rmp_env.def` file to build a `.sif` image.

* **Important:** You do not have enough permissions to build the image directly on the cluster. You must build the image on your local machine first, and then upload the `.sif` file to the cluster.
* **Base Image:** The image is built from `nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04`. The final image size is large (close to 10GB).

Use the standard command to build the image locally:

```bash
apptainer build rmp_env.sif rmp_env.def
```

## 4. Data Preparation

### Download Data

You can find the download links and dataset information here:

* **Dataset Download:** [RadioMapSeer Dataset](https://radiomapseer.github.io/)
* **More Information & Tips:** [RadioUNet Reproduction Tips](https://my-website-five-gray-23.vercel.app/docs/radiounet#reproduction-tips)

### Directory Structure

After downloading the dataset, please extract the files. We recommend placing the extracted folders directly inside a directory named `data` in the project root folder.

Your project directory should look like this:

```text
radio-map-prediction/
├── config
├── data/
│   ├── antenna/
│   ├── gain/
│   ├── png/
│   ├── polygon/
│   └── dataset.csv
├── src
├── .env
└── ... (other files)

```

*Note: Please check your configuration file to ensure the data paths match this structure.*

## 5. Configuration Guide

All parameters for training, testing, and data processing are managed in YAML configuration files.

### Configuration Files Management

We recommend putting all your configuration files inside the `config/` folder. To keep things organized, please name your configuration files using this clear format:
`{architecture}_{simulation_type}_{cars}_{map_type}_{samples}.yaml`

The project already includes some example configuration files in the `config/` directory:

```text
radio-map-prediction/
├── config/
│   ├── radiounet_dpm_nocars_missing0_samples0.yaml
│   ├── radiownet_dpm_nocars_missing0_samples0.yaml
│   └── ... (more examples in the future)
└── ... (other directories and files)
```

> **Note on Model Architecture Switching:** Currently, changing the network architecture is still hard-coded. You need to change the model class manually in the main entry `job.py` file. In the future, we plan to update the code so you can switch architectures directly inside the configuration file.

### Key Parameter Groups

#### 1. Global Setting

* `seed`: Random seed to ensure dataset splitting and experiment result can be exactly reproduced.

#### 2. Load Configuration (`load:`)

* `train_ratio` & `val_ratio`: Ratios for splitting the city maps automatically. For example, if `maps_number: 700`, `train_ratio: 0.7`, and `val_ratio: 0.15`:
  * **Train Set:** $700 \times 0.7 = 490$ random maps.
  * **Validation Set:** $700 \times 0.15 = 105$ random maps.
  * **Test Set:** The remaining maps ($700 - 490 - 105 = 105$ maps).
* `train_batch_size` / `val_batch_size` / `test_batch_size`: Batch sizes for training, validation, and testing phases.
* `num_workers`: Number of subprocesses to use for data loading.

#### 3. Train Configuration (`train:`)

* `epoch`: Total number of training epochs.
* `learning_rate`: Initial learning rate for the optimizer.
* `scheduler`: Learning rate decay settings. It decreases the learning rate by multiplying `gamma` (attenuation ratio) every `step_size` epochs.
* `early_stop`: Settings to stop training early if the validation loss stops improving. The model must improve by at least `delta` within the `patience` number of epochs.
* `out_dir`: The directory path where the trained models, logs, and results are saved.

#### 4. Data Configuration (`data:`)

* **Directory Paths:**
  * `root_dir`: The root folder of the dataset.
  * `DPM_dir` / `DPM_cars_dir` / `IRT2_dir` / `IRT2_cars_dir` / `IRT4_dir` / `IRT4_cars_dir`: Directories containing simulation gain data with or without cars.
  * `buildings_complete_dir` / `buildings_missing_dir` / `antennas_dir` / `cars_dir`: Directories containing PNG maps for city buildings, antenna positions, and car positions.
* **Simulation & Map Settings:**
  * `simulation`: Choose the simulation mode (`DPM`, `IRT2`, or `rand`). *Note:* `rand` mode mixes DPM and IRT2 based on the `IRT2_weight`.
  * `city_map`: Choose the type of city map (`complete`, `missing`, or `rand`).
  * `missing`: The number of missing buildings (Range: `[1, 4]`). *Note:* This parameter **only works** when `city_map` is set to `"missing"`.
* **Advanced Dataset Settings:**
  * `sparse_IRT4_number`: Number of sparse IRT4 points on the map (Range: `[0, total_img_size)`).
    * Important Logic: If `sparse_IRT4_number > 0`, the code **automatically forces** the system to switch to **IRT4 simulation mode** for training and validation targets, and the `simulation` setting above will be ignored.
  * `samples_number`: Number of extra simulation gain samples to input. The range depends on whether `sparse_IRT4_number` is 0 or not.
  * `cars_exist`: Set to `true` or `false`. If `true`, cars information is added into the model as an extra feature channel.
  * `maps_number`: Total number of city maps to use (Range: `[1, 700]`).
  * `transmitters_number`: The number of transmitters per map.
    * Train/Val vs. Test: This configuration **only applies to Training and Validation sets**. For the **Test set**, the code hardcodes the transmitter number to **2** because the IRT4 simulation in dataset only supports a maximum of 2 transmitters.
    * Conflict Warning: If `sparse_IRT4_number > 0`, `transmitters_number` must be between `[1, 2]`. If you set it to a large number like `80`, the code will throw a configuration error and stop.
  * `threshold`: Pathloss threshold filter value (Range: `[0, 1)`).
  * `img_size`: The resolution of input images, default is `[256, 256]`.

### Configuration Validation

You do not need to worry about making mistakes in the configuration file. The project includes a validation script at `src/utils/config.py`. When you start the program, it will automatically check your YAML configuration file. If there are conflicting or incorrect settings, the system will print a clear error message to remind you what to fix.

## 6. How to Run

Currently, the project uses `job.py` as the main entry point. In the future, we plan to support command-line arguments (like `python main.py --config ... --mode ...`). For now, you can run the project locally or on the Telecom Paris cluster by following the steps below.

### Run on a Local Machine

Before running the code, make sure you have installed the conda environment as described in the [Environment Setup](#environment-setup) section.

#### Step 1: Activate the Environment

Open your terminal and activate the virtual environment:
```bash
conda activate rmp
```

#### Step 2: Set Your Mode and Config File

Currently, you need to open `job.py` and manually change two variables in the code:

1. **Config Path:** Find the configuration file path string and change it to the file you want to use (for example: `"config/radiounet_dpm_nocars_missing0_samples0.yaml"`).
2. **Mode:** Find the `mode` variable and set it to:
  * `'train'`: For running both training and testing.
  * `'test'`: For running testing only.

#### Step 3: Start the Program

Run the following command:
```bash
python job.py
```

### Run on the Telecom Paris Cluster

If you want to run your experiments on the **Telecom Paris cluster**, you can submit a batch job using Slurm.

#### Step 1: Prepare Your Job Script

The repository includes a cluster submission script named `job.sh`. Make sure you have already built your Apptainer image (`rmp_env.sif`) and uploaded it to your cluster workspace.

#### Step 2: Submit the Job

Use the `sbatch` command to submit `job.sh` to the cluster:

```bash
sbatch job.sh
```

*Note: You can open the `job.sh` file to view or modify its contents.*