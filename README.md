# DDI-VO: Deep Direct-Indirect Visual Odometry

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

DDI-VO is a hybrid 6-DoF Visual Odometry architecture that leverages a pure Vision Transformer (ViT) backbone to seamlessly integrate direct (photometric/global) and indirect (feature-based) tracking paradigms. By combining globally consistent representations with robust sparse tracking (SuperPoint + LightGlue), this model achieves strong generalization across diverse motion profiles including autonomous driving, aerial flight, and handheld scenarios.

---

## Installation

You can run DDI-VO either natively using a Python virtual environment or via Docker for guaranteed reproducibility.

### Option A: Docker
We provide a `Dockerfile` that packages all required system dependencies, CUDA drivers, and Python libraries.

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/larocs/DDI-VO.git
cd DDI-VO
```

2. Build the Docker image:

```bash
docker build -t ddi-vo .
```

3. Run the container (mounting your local dataset folder):

```bash
docker run --gpus all -it -v /path/to/your/local/datasets:/workspace/DDI-VO/datasets ddi-vo /bin/bash
```

### Option B: Local Setup (Conda / Venv)

1. Clone the repository with submodules:

```bash
git clone --recursive https://github.com/larocs/DDI-VO.git
cd DDI-VO
```

(If you already cloned without submodules, run: `git submodule update --init --recursive`)

2. Install dependencies:
Ensure you have PyTorch and Torchvision installed properly according to your CUDA version, then run:

```bash
pip install -r requirements.txt
```

## Dataset Preparation
To use the default dataloaders (kitti.py, queenscamp.py, tartanair.py), your datasets must be strictly organized in the following hierarchy inside the datasets/ directory:

```Plaintext
datasets/
├── kitti/
│   ├── sequences/
│   │   ├── 00/
│   │   │   ├── image_2/
│   │   │   └── calib.txt
│   │   └── 01/
│   └── poses/
│       ├── 00.txt
│       └── 01.txt
├── queenscamp/
│   ├── rgb_camera_info.txt
│   └── sequences/
│       ├── 01/
│       │   ├── images/
│       │   └── traj.txt
│       └── 02/
└── tartanair/
    ├── rgb_camera_info.txt
    └── abandonedfactory/
        ├── Easy/
        │   ├── P000/
        │   │   ├── image_left/
        │   │   └── pose_left.txt
        │   └── P001/
        └── Hard/
```

## Model Weights
DDI-VO requires pre-trained weights to run, which can be obtained as follows:

```bash
chmod +x download_weights.sh
./download_weights.sh
```

## Usage

### Training
To train or fine-tune the model on the supported datasets, configure your parameters in configs/train_example.yaml and run:

```bash
python train.py checkpoints/ddi_vo_experiment \
    --conf configs/train_example.yaml \
    --use_cuda
```

### Testing / Inference
To run inference and generate trajectory files (traj.txt) for evaluation against ground truth, run:

```bash
python test.py \
    --dataset_config configs/ddi_vo.yaml \
    --model_config configs/ddi_vo_model.yaml \
    --model_path checkpoints/ddi_vo_experiment/best_model.tar \
    --output_path results \
    --trajectory_file traj.txt
```


## Other publications
Check out our deep homography estimation visual transformer, which is available [here](https://github.com/larocs/deep-homography-vit).