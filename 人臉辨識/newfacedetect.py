#!/usr/bin/python3
# coding=utf8
import sys
sys.path.append('/home/pi/TonyPi/')
import cv2
import math
import time
import Camera
import threading
import numpy as np
import yaml_handle
import HiwonderSDK.Misc as Misc
import HiwonderSDK.Board as Board
import HiwonderSDK.ActionGroupControl as AGC

# 人臉偵測

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

# 閾值
conf_threshold = 0.6

# 模型位置
modelFile = "/home/pi/TonyPi/models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "/home/pi/TonyPi/models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

servo_data = None
def load_config():
    global servo_data
    servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)

load_config()

servo2_pulse = servo_data['servo2']
# 初始位置
def initMove():
    Board.setPWMServoPulse(1, 1800, 500)
    Board.setPWMServoPulse(2, servo2_pulse, 500)

d_pulse = 10
start_greet = False
action_finish = True

# 變量重置
def reset():
    global d_pulse
    global start_greet
    global servo2_pulse    
    global action_finish

    d_pulse = 10
    start_greet = False
    action_finish = True
    servo2_pulse = servo_data['servo2']    
    initMove()  
    
# app初始化調用
def init():
    print("FaceDetect Init")
    reset()

__isRunning = False
# app開始玩法調用
def start():
    global __isRunning
    __isRunning = True
    print("FaceDetect Start")

# app停止玩法調用
def stop():
    global __isRunning
    __isRunning = False
    reset()
    print("FaceDetect Stop")

# app退出玩法調用
def exit():
    global __isRunning
    __isRunning = False
    AGC.runActionGroup('stand')
    print("FaceDetect Exit")

def move():
    global start_greet
    global action_finish
    global d_pulse, servo2_pulse
    global detect
    
    while True:
        if __isRunning:
            if start_greet:
                start_greet = False
                action_finish = False
                Board.setPWMServoPulse(2, servo2_pulse, 500)
                action_finish = True
                time.sleep(5)
            else:
                if servo2_pulse > 2000 or servo2_pulse < 1000:
                    d_pulse = -d_pulse
            
                servo2_pulse += d_pulse       
                Board.setPWMServoPulse(2, servo2_pulse, 50)
                time.sleep(0.05)
        else:
            time.sleep(0.01)
            
# 運行子線程
th = threading.Thread(target=move)
th.setDaemon(True)
th.start()

size = (320, 240)
def run(img):
    global start_greet
    global action_finish
       
    img_copy = img.copy()
    img_h, img_w = img.shape[:2]

    if not __isRunning:
        return img

    blob = cv2.dnn.blobFromImage(img_copy, 1, (150, 150), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()  # 計算識別
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            # 識別到的人臉的各個座標轉換為未縮放前的座標
            x1 = int(detections[0, 0, i, 3] * img_w)
            y1 = int(detections[0, 0, i, 4] * img_h)
            x2 = int(detections[0, 0, i, 5] * img_w)
            y2 = int(detections[0, 0, i, 6] * img_h)             
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2, 8)  # 將識別到的人臉框出
            
            # 計算人臉的中心點
            face_center_x = (x1 + x2) / 2
            face_center_y = (y1 + y2) / 2
            
            # 檢查人臉是否在視野中心
            if abs(face_center_x - img_w / 2) < img_w / 4:
                if action_finish:
                    start_greet = True

            # 控制機器人跟隨人臉
            if face_center_x < img_w / 3:
                AGC.runActionGroup('turn_left')
            elif face_center_x > img_w * 2 / 3:
                AGC.runActionGroup('turn_right')
            elif face_center_y < img_h / 3:
                AGC.runActionGroup('go_forward')
            elif face_center_y > img_h * 2 / 3:
                AGC.runActionGroup('back')
    
    return img

if __name__ == '__main__':
    from CameraCalibration.CalibrationConfig import *
    
    # 加載參數
    param_data = np.load(calibration_param_path + '.npz')

    # 獲取參數
    mtx = param_data['mtx_array']
    dist = param_data['dist_array']
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (640, 480), 0, (640, 480))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (640, 480), 5)    
    
    init()
    start()
    my_camera = Camera.Camera()
    my_camera.camera_open()
    AGC.runActionGroup('stand')
    while True:
        img = my_camera.frame
        if img is not None:
            frame = img.copy()
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)  # 畸變校正
            Frame = run(frame)           
            cv2.imshow('Frame', Frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        else:
            time.sleep(0.01)
    my_camera.camera_close()
    cv2.destroyAllWindows()