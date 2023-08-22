# ICDAR 2019 Competition
Official Code for the 3rd Place in DIBCO 2019 (Held in Conjunction with ICDAR 2019) 
[Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8978205)

## Table of Contents
- [Installation](#Installation)
- [Inference](#Inference)
- [Training](#Training)
- [Data](#Data)
- [Pretrained weights](#checkpoint)
- [Acknowledgments](#Acknowledgments)
- [Citation](#Citation)
## Installation
**Requirements:**
- Python3 
- PyTorch 0.4 
- CV2  3.4.5 
- imutils 
- argparse 
- natsort 
- Configparser
## Inference
(From src/)

The pre-trained models are provided in folders ( src/checkpoint/ )

**1. Testing by method a**
   
$ python binarize_a.py arg1 arg2

+ arg1: direction of folder of input Images  ( default: “./input/” ) ( you should put input images in folder: ./input/  and $ python binarize_a.py)

+ arg2: direction of folder of result ( default: “./result_a/” )

EX: 

```
python binarize_a.py --input ./inputa/ --result ./resulta/ 
```

**2. Testing by method b**

$ python binarize_b.py arg1 arg2

+ arg1: direction of folder of input Images  ( default: “./input/” ) ( you should put input images in folder: ./input/  and $ python binarize_b.py)

+ arg2: direction of folder of result ( default: “./result_b/” )

EX: 

```
python binarize_b.py --input ./inputb/ --result ./resultb/  
```
## Training

- Run the Python code for training: retinaNN_training.py

(choose layer=3, filter=16 for model 1, choose layer=4, filter =10 for model 2 in code)

- Model1 and model2 will be stored src/checkpoint/

## Data

The dataset is created from the DIBCO 2009, DIBCO 2011, DIBCO 2013, H-DIBCO 2010, HDIBCO 2012, H-DIBCO 2014, H-DIBCO 2016 datasets and H-DIBCO 2018.

## Pretrained weights

In the folder: src/checkpoint/

## Acknowledgments

Refer to segmentation code [LadderNet](https://github.com/juntang-zhuang/LadderNet)

## Citation
If you found this code helpful, please consider citing our extended paper: 

```
@article{dang2021document,
  title={Document image binarization with stroke boundary feature guided network},
  author={Dang, Quang-Vinh and Lee, Guee-Sang},
  journal={IEEE Access},
  volume={9},
  pages={36924--36936},
  year={2021},
  publisher={IEEE}
}
```


