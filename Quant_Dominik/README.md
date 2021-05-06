# Precision
8 bit weights, 8bit activations, 32bit bias
mAP: 0.2726801863851039

# Git Version
https://github.com/eriklindernoren/PyTorch-YOLOv3
$ git log -1
commit 47b7c912877ca69db35b8af3a38d6522681b3bb3 (HEAD -> master, origin/master, origin/HEAD)
Merge: 6442556 38c2465
Author: Erik Linder-Norén <eriklindernoren@live.se>
Date:   Mon May 6 00:27:54 2019 +0200


# Files
quant_util.py
yolo_quant.py




# PyTorch-YOLOv3
**Note:** YoloV3-tiny is based on this git repository

A minimal PyTorch implementation of YOLOv3, with support for training, inference and evaluation.

## Installation
##### Clone and install requirements
    $ git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
    $ cd PyTorch-YOLOv3/
    $ sudo pip3 install -r requirements.txt

##### Download pretrained weights
    $ cd weights/
    $ bash download_weights.sh

##### Download COCO
    $ cd data/
    $ bash get_coco_dataset.sh