# this code execute in wsl2 enviorment (trivial)

import torch
from PIL import Image
import cv2
import os
import numpy as np


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
img = 'https://ultralytics.com/images/zidane.jpg'
results = model(img)
results.show()


img2 = cv2.imread('karina.PNG', cv2.IMREAD_COLOR)

cv2.imshow('img', img2)
while(1):
    print("1")