# CPAL

This project provides the code and results for 'CPAL: Cross-Prompting Agent of Large Vision Model for Multi-Modal Semantic Segmentation'

The source code will be released upon publication.


# Requirements

```
conda create -n cpal python=3.7 -y
conda activate cpal

pip install -r requirements.txt
```
 - Install CUDA>=10.2 with cudnn>=7
 - Install PyTorch>=1.10.0 and torchvision>=0.9.0 with CUDA>=10.2

# Multi-modal Semantic Segmentation Performance

## RGB-D
### NYU-Depth v2 and SUN-RGBD

<img src=./pic/image-1.png width=850>

<img src=./pic/image.png width=850>

## RGB-T
### FMB

<img src=./pic/image-2.png width=450>

<img src=./pic/image-7.png width=450>

### PST900

<img src=./pic/image-3.png width=850>

<img src=./pic/image-6.png width=850>

## RGB-E
### DDD17

<img src=./pic/image-4.png width=450>

## Multi-modal VOS
### MVseg

<img src=./pic/image-5.png width=450>


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