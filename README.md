# CPAL

# Requirements

```
conda create -n cpal python=3.7 -y
conda activate cpal
```
 - Install CUDA>=10.2 with cudnn>=7
 - Install PyTorch>=1.10.0 and torchvision>=0.9.0 with CUDA>=10.2

## Models

[Models](https://njupteducn-my.sharepoint.com/:f:/g/personal/1223055916_njupt_edu_cn/ElbL2uJLmMhDk6C5z454CWkBDkIBCNWULN_x-AKCvcfYdw?e=f0Wq0O) (OneDrive)

## Data

Your dataset folder under "data" should be like:

```
data
|   ----RGB-D
|       ----nyu_depth_v2
|       |   ----images
|       |   |       0.png
|       |   |       .....
|       |   |
|       |   ----images_x
|       |   |       0.png
|       |   |       .....
|       |   |
|       |   ----annotations
|       |   |       0.png
|       |   |       .....
|       |   |
|       |   ----splits
|       |   |       train.txt
|       |   |       .....
|       |   |
|       ----sun_rgbd
|       |   .....
|   ----RGB-T
|
|   ----RGB-E

```

## How to reproduce results
```python
python test.py "$path_of_config" "$path_of_checkpoint" --eval mIoU --show-dir "$path_of_output"
```

## For training
```python
python train.py --work-dir "$path_of_work-dir" --load-from "$path_of_pretrained_model" "$path_of_config"
```
