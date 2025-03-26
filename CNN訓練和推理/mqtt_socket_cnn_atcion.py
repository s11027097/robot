import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import paho.mqtt.client as paho
import json
#////////////////////////
import latest_model_path
#////////////////////////

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  
        self.pool = nn.MaxPool2d(kernel_size=2)  
        self.fc1 = nn.Linear(64 * 62 * 62, 16) 
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(-1, 64 * 62 * 62)  
        x = self.dropout(x)
        x = self.fc1(x)
        return x

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

model_path = latest_model_path.modle_path()

model = CNN()
# 正確地將模型加載到 CPU
model.load_state_dict(torch.load(os.path.join(os.getcwd(), "action_model", model_path)))
model.eval()

labels_file = os.path.join(os.getcwd(), "action.csv")
labels_df = pd.read_csv(labels_file)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(labels_df.iloc[:, 1:].values)

def denormalize(y_normalized, scaler):
    return scaler.inverse_transform(y_normalized)

unlabeled_dir = os.path.join(os.getcwd(), "image_test_data")

def infer_images(directory):
    predictions = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                output = model(image)
                predictions.append((filename, output.cpu().numpy()))
    return predictions

predicted_values = infer_images(unlabeled_dir)

predicted_motor_values = []
for filename, pred in predicted_values:
    pred_original_scale = np.round(denormalize(pred, scaler)).astype(int)
    predicted_motor_values.append((filename, pred_original_scale))

import socket
RASPBERRY_PI_IP = '192.168.138.93'
PORT = 65432

for filename, motor_values in predicted_motor_values:
    print(f'圖片為: {filename}, 預測的馬達數值: {motor_values}')
    motor_values_list = motor_values.tolist()  
    motor_values_json = json.dumps(motor_values_list)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

    # 連接到Raspberry pi
    s.connect((RASPBERRY_PI_IP, PORT))
    
    s.sendall(motor_values_json.encode('utf-8'))