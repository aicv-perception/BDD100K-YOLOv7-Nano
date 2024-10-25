# Pedestrian-and-Vehical-Detection

## Introduction
This model of this project is baesd on [YOLOV7-tiny](https://github.com/bubbliiiing/yolov7-tiny-pytorch)

## Requirements
### Dependencies
* Python 3.8
* torch==2.0.1
* torchvision==0.15.2
* tensorboard==2.2.2
* scipy==1.10.1
* numpy==1.19.0
* matplotlib==3.1.2
* opencv_python==4.10.0.84
* tqdm==4.64.1
* Pillow==9.4.0
* h5py==2.10.0

### It was ran and tested under the following OSs:
* Ubuntu 20.04 with NVIDIA 4070 GPU

## Preparing Data
1. To build **training** dataset, you'll also need following datasets.
* [BDD100K](https://doc.bdd100k.com/download.html)

## Getting Started:
### Usage
* Training
    * To train this code.
    ```
    python train.py
    ```

* Testing
    * To test this code.
    ```
    python test.py
    ```

## Reference 
[Official YOLOv7](https://github.com/WongKinYiu/yolov7)
