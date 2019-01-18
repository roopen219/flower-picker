Forked from https://github.com/ultralytics/yolov3

# Requirements

Python 3.7 or later with the following `pip3 install -U -r requirements.txt` packages:

- `numpy`
- `torch`
- `opencv-python`

# Setting up the environment

Download all the folders from [here](https://drive.google.com/drive/folders/1GGmVZiYwgbsz0rJ-w1YrRJdBZJeGu66u?usp=sharing) and paste in the root/repo directory

# Detecting flowers

Put the images you want to detect flowers in into the `data/samples/flower` folder and run
```
python detect.py
```

Check the `output` folder for results which are images with bounding boxes.

# Training

You can backup your `weights/best.pt` weight file before starting a fresh training session because it **may** get overwritten.

**Start Training:**
```
python train.py --freeze
```

**Resume Training:**
```
python train.py --freeze --resume
```
to resume training from the most recently saved checkpoint `weights/latest.pt`.

## Image Augmentation

`datasets.py` applies random OpenCV-powered (https://opencv.org/) augmentation to the input images in accordance with the following specifications. Augmentation is applied **only** during training, not during inference. Bounding boxes are automatically tracked and updated with the images. 416 x 416 examples pictured below.

Augmentation | Description
--- | ---
Translation | +/- 10% (vertical and horizontal)
Rotation | +/- 5 degrees
Shear | +/- 2 degrees (vertical and horizontal)
Scale | +/- 10%
Reflection | 50% probability (horizontal-only)
H**S**V Saturation | +/- 50%
HS**V** Intensity | +/- 50%
