# ICDAR 2019 Competition
Official Code for the 3rd Place in DIBCO 2019 (Held in Conjunction with ICDAR 2019)

## Table of Contents
- [Installation](#Installation)
- [Inference](#Inference)
- [Train](#Train)
- [Data](#Data)
- [Pretrained weights](#checkpoint)
- [Acknowledgments](#Acknowledgments)
- [Citation](#Citation)
## Installation: 
**Requirements:**
- Python3 
- PyTorch 0.4 
- CV2  3.4.5 
- imutils 
- argparse 
- natsort 
- Configparser
## Inference
The pre-trained models are provided in folders ( LadderNet-master/src/checkpoint/ )
1. Testing by method a
From LadderNet-master/src/
$ python binarize_a.py arg1 arg2

arg1: direction of folder of input Images  ( default: “./input/” ) ( you should put input images in folder: ./input/  and $ python binarize_a.py)

arg2: direction of folder of result ( default: “./result_a/” )

EX: 
\```bash
python binarize_a.py --input ./inputa/ --result ./resulta/ 
\```


