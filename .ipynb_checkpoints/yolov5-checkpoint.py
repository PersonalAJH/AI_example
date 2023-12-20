# this code execute in wsl2 enviorment (trivial)
# if there is an error in pafy import youtube, pip install youtube-dl==2020.12.2


# https://github.com/ultralytics/yolov5
# git clone https://github.com/ultralytics/yolov5 
# cd yolov5
# pip install -r requirements.txt 




import torch
from PIL import Image
import cv2
import os
import numpy as np
import pafy


# yolo v5 offered example 
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# img = 'https://ultralytics.com/images/zidane.jpg'
# results = model(img)
# results.print()

# object detection in youtube video
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
url = "www.youtube.com/watch?v=Nj2U6rhnucI"
video = pafy.new(url)
preftype = video.getbest(preftype="mp4")
cap = cv2.VideoCapture(preftype.url) 


while cv2.waitKey(1) <0:
    hasFrame, frame = cap.read()
    results = model(frame)
    results.print()

