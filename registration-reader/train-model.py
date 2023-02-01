# The Following is designed to train a Yolov5 model in Google Colab

# !pip install xmltodict
# !pip install split-folders
# !pip install easyocr
# !pip install GPUtil
# !git clone https: // github.com/ultralytics/yolov5  # clone
# !cd yolov5 & & pip install - r requirements.txt  # install

# Import Libraries
import yaml
from google.colab import drive
from matplotlib import patches as mpatches
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
import matplotlib
import torch
from timeit import default_timer as timer
from numba import cuda
from GPUtil import showUtilization as gpu_usage
from tqdm.auto import tqdm
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from pathlib import Path
import copy
import PIL
import easyocr
import splitfolders
import random as rnd
import xml.etree.ElementTree as ET
import glob
import xmltodict
import pandas as pd
import time
import uuid
import cv2
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

pd.options.mode.chained_assignment = None  # default='warn'


matplotlib.use('TkAgg')


# Prepare Dataset
dataset = {
    "file": [],
    "width": [],
    "height": [],
    "xmin": [],
    "ymin": [],
    "xmax": [],
    "ymax": []
}

# Mount Google Drive
drive.mount('/content/drive')

img_names = []
annotations = []
for dirname, _, filenames in os.walk("/content/drive/TrainingData"):
    for filename in filenames:
        if os.path.join(dirname, filename)[-3:] == ("png" or "jpg"):
            img_names.append(filename)
        elif os.path.join(dirname, filename)[-3:] == "xml":
            annotations.append(filename)

img_names[:10]
annotations[:10]

# XML to Dict
path_annotations = "/content/drive/TrainingData/annotations/*.xml"

for item in glob.glob(path_annotations):
    tree = ET.parse(item)

    for elem in tree.iter():
        if 'filename' in elem.tag:
            filename = elem.text
        elif 'width' in elem.tag:
            width = int(elem.text)
        elif 'height' in elem.tag:
            height = int(elem.text)
        elif 'xmin' in elem.tag:
            xmin = int(elem.text)
        elif 'ymin' in elem.tag:
            ymin = int(elem.text)
        elif 'xmax' in elem.tag:
            xmax = int(elem.text)
        elif 'ymax' in elem.tag:
            ymax = int(elem.text)

            dataset['file'].append(filename)
            dataset['width'].append(width)
            dataset['height'].append(height)
            dataset['xmin'].append(xmin)
            dataset['ymin'].append(ymin)
            dataset['xmax'].append(xmax)
            dataset['ymax'].append(ymax)

classes = ['license']

df = pd.DataFrame(dataset)
df
df.info()

# Data Normalization
x_pos = []
y_pos = []
frame_width = []
frame_height = []

labels_path = Path("kaggle/CV/Plate_recognition/labels")

labels_path.mkdir(parents=True, exist_ok=True)

save_type = 'w'

for i, row in enumerate(df.iloc):
    current_filename = str(row.file[:-4])

    width, height, xmin, ymin, xmax, ymax = list(df.iloc[i][-6:])

    x = (xmin+xmax)/2/width
    y = (ymin+ymax)/2/height
    width = (xmax-xmin)/width
    height = (ymax-ymin)/height

    x_pos.append(x)
    y_pos.append(y)
    frame_width.append(width)
    frame_height.append(height)

    txt = '0' + ' ' + str(x) + ' ' + str(y) + ' ' + \
        str(width) + ' ' + str(height) + '\n'

    if i > 0:
        previous_filename = str(df.file[i-1][:-4])
        save_type = 'a+' if current_filename == previous_filename else 'w'

    with open("kaggle/CV/Plate_recognition/labels/" + str(row.file[:-4]) + '.txt', save_type) as f:
        f.write(txt)


df['x_pos'] = x_pos
df['y_pos'] = y_pos
df['frame_width'] = frame_width
df['frame_height'] = frame_height

df

# Split Dataset
input_folder = Path("drive/My Drive/TrainingData")
output_folder = Path("yolov5/data/Plate_recognition")
splitfolders.ratio(
    input_folder,
    output=output_folder,
    seed=42,
    ratio=(0.8, 0.2),
    group_prefix=None
)
print("Moving files finished.")


# Convert to YAML
yaml_file = 'yolov5/data/plates.yaml'

yaml_data = dict(
    path="data/Plate_recognition",
    train="train",
    val="val",
    nc=len(classes),
    names=classes
)

with open(yaml_file, 'w') as f:
    yaml.dump(yaml_data, f, explicit_start=True, default_flow_style=False)


# Free GPU Cache
def free_gpu_cache() -> None:
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()


free_gpu_cache()
device = '0' if torch.cuda.is_available() else 'cpu'
device


# Train Model
start_time = timer()


# %cd yolov5
#!python train.py - -workers 2 - -img 640 - -batch 16 - -epochs 100 - -data "data/plates.yaml" - -weights yolov5s.pt - -device {device} - -cache

end_time = timer()

print(f'Training time: {(end_time-start_time):.2f}')
