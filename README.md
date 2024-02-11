## Introduction
This repo contains training and evaluation code of CCTV-GUN benchmark. It uses [mmdetection](https://mmdetection.readthedocs.io/en/latest/) to train object detection models.


## Requirements
We follow the installation instructions in the mmdetection documentation [here](https://mmdetection.readthedocs.io/en/v2.2.0/install.html). Specifically, our code requires `mmcls=0.25.0,` `mmcv-full=1.7.0` and `torch=1.13.0`. 

The output of `conda env export > env.yml ` can be found in [env.yml](./requirements/env.yml). It can be used to create a conda virtual environment with

```
conda env create  -f env.yml
conda activate env_cc
pip install openmim
mim install mmcv-full==1.7.0
pip install -e . 
```

## Data
We use images from three datasets : 

1. Youtube Images
2. US Real-time Gun detection dataset (USRT)

Instructions on how to download these datasets can be found in [dataset_instructions.md](./dataset_instructions.md) .

## Training

All of the above datasets consists of two classes : Person (class 0) and Handgun (class 1). To train a detection model on this dataset, run
```bash
python tools/train.py --config <path/to/model/config.py> --dataset-config <path/to/dataset/config.py> <extra_args>
```

- Model config files [link](./configs/gun_detection/)

- Dataset config files [link](./configs/_base_/datasets/gun_detection/)

- Trained models [link](https://drive.google.com/drive/folders/1uvNthQ_iSjDDf2nlPY9g3iEYA16Dn60H?usp=sharing)

### Extra args
To adjust the training batch size
```
<base_command> --cfg-options data.samples_per_gpu=<batch-size>
```
Using [weights and biases](https://wandb.ai/) to log metrics:
After you create an account in wandb, change `entity` and `project` in [train.py](./tools/train.py) to your wandb username and project name. Then 
```
<base_command> --use-wandb --wandb-name <name-of-the-experiment>
```
### Examples:

Train a Swin-T on Ytimgs (Intra-dataset)
```bash
python tools/train.py --config configs/gun_detection/swin_transformer.py --dataset-config configs/_base_/datasets/gun_detection/ytimgs.py --cfg-options data.samples_per_gpu=6
```

## Testing
To evaluate a trained model, run
```bash
python tools/test.py --config <path/to/model/config.py> --dataset-config <path/to/dataset/config.py> --checkpoint <path/to/trained/model> --work-dir <path/to/save/test/scores> --eval bbox
```

### Examples:

Evaluate a ConvNeXt trained on USRT

```bash
python tools/test.py --config configs/gun_detection/convnext.py --dataset-config configs/_base/datasets/gun_detection/usrt.py --checkpoint <path/to/mgd+usrt/trained/model.pth> --work-dir <path/to/save/test/scores> --eval bbox
```

To save the bounding box predictions on test set , add `--save-path <path/to/output/folder>` to the above command.
