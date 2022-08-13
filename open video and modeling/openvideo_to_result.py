import tensorflow as tf
import cv2
import numpy as np

# 모델 로드

from tensorflow.keras.models import load_model
model = load_model("ae_bottleneck_16_179epoch.h5")

# threshold 설정
threshold = 1.198503541946411

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
    rge = np.max(tmp) - np.min(tmp)
    print(rge)
    if rge >= threshold:
        print("warning")
    


capture.release()
cv2.destroyAllWindows()