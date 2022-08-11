import tensorflow as tf
import cv2
import numpy as np

# 모델 로드
interpreter = tf.lite.Interpreter("ae_bottleneck_16_179epoch.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# threshold 설정
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
    crop_n = np.array(crop_frame).astype('float32')/255
    c_frame = np.reshape(crop_n, (1,128,128,3))
    
    cv2.imshow("VideoFrame", crop_n)

    # tflite에 input
    input_data = c_frame
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # tflite에서 output(predict 역할)
    reconstructed = interpreter.get_tensor(output_details[0]['index'])
    reconstructed_r = np.reshape(reconstructed, (128, 128, 3))

    # error 계산
    tmp = crop_n - reconstructed_r
    tmp = np.abs(tmp)
    tmp_2d = np.sum(tmp, axis = 2)
    tmp_1d = np.sum(tmp_2d, axis = 1)
    tmp_sum = np.sum(tmp_1d)
    # error 출력
    print(tmp_sum)
    
    # 탐지 실행
    if tmp_sum >= threshold:
        print("warning")



capture.release()
cv2.destroyAllWindows()