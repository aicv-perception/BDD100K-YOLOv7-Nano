# BDD100K-YOLOv7-Nano

## Introduction
The project will be updated continuously.

<!--This model of this project is baesd on [YOLOV7-tiny](https://github.com/bubbliiiing/yolov7-tiny-pytorch)-->

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

<!----2. Place the bkk100k images and its corresponding JSON file into folder, and run following script.
```
python read_bdd_json.py
```--->
2. Structure of the generated data should be：
```
├── new dataset folder
    ├──Annotations
    │  ├── b1c9c847-3bda4659.json
    │  ├── b1c66a42-6f7d68ca.json
    │  └── ...
    ├──Images
    │  ├── b1c9c847-3bda4659.jpg
    │  ├── b1c66a42-6f7d68ca.jpg
    │  └── ...
    ├──Lanes
    │  ├── b1c66a42-6f7d68ca.png
    │  ├── b1c66a42-6f7d68ca.png
    │  └── ...
    ├──train.txt
    ├──val.txt
    └──class.txt
```

## Pre-trained Model
You can get the pre-trained model based on the original bdd100k annotation from <a  href="https://drive.google.com/file/d/1w1WeaOac6WeMfgXEZ9TgYtuRkCODIsyd/view?usp=sharing">here</a>.


## Getting Started:
### Usage

* Training

Coming soon.....

* Testing
```
python demo.py --cuda True --mode video --input_shape 640 640 --video_path /video/path --model_path /model/path --classes_path /class/path --anchors_path /anchor/path
```
### Demo
```
python demo.py --cuda True --mode  --video_path ./video.avi --model_path ./Nano640_bdd100k_original_annotation.pt --classes_path ./utils/bdd100k_class.txt --anchors_path ./utils/bdd_nano_640_6_anchors.txt
```

## Reference 
[Official YOLOv7](https://github.com/WongKinYiu/yolov7)
