import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import tensorflow as tf
import PIL.Image as pilimg
import os

import pandas as pd
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose

from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.models import load_model

import random
import pickle
import cv2
import matplotlib

from glob import glob
from numpy import random
from PIL import Image

# 모델 로드

from tensorflow.keras.models import load_model
model = load_model("ae_bottleneck_16_179epoch.h5")

threshold = 4633

# 시작
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cv2.waitKey(1800) < 0:
    ret, frame = capture.read()
    frame_r = cv2.resize(frame, (224,224))
    y = 48
    h = 128
    x = 48
    w = 128
    crop_frame = frame_r[y: y + h, x: x + w]
    crop_n = np.array(crop_frame)/255
    c_frame = np.reshape(crop_n, (1,128,128,3))
    
    cv2.imshow("VideoFrame", crop_n)

    
    reconstructed = model.predict(c_frame)
    reconstructed_r = np.reshape(reconstructed, (128, 128, 3))


    tmp = crop_n - reconstructed_r
    tmp = np.abs(tmp)
    tmp_2d = np.sum(tmp, axis = 2)
    tmp_1d = np.sum(tmp_2d, axis = 1)
    tmp_sum = np.sum(tmp_1d)
    print(tmp_sum)
    if tmp_sum >= threshold:
        print("warning")
    


capture.release()
cv2.destroyAllWindows()