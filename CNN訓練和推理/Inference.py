import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

#////////////////////////
import latest_model_path  # 自訂模組，負責提供最新的模型路徑
#////////////////////////

# 設定設備，若有可用 GPU 則使用，否則使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義 CNN 模型架構
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)  # 第一層卷積，輸入通道數為 3（RGB），輸出 32 個通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 第二層卷積，輸入 32 通道，輸出 64 通道
        self.pool = nn.MaxPool2d(kernel_size=2)  # 最大池化層，降低特徵圖大小
        self.fc1 = nn.Linear(64 * 62 * 62, 16)  # 全連接層，輸入維度 64×62×62，輸出 16
        self.dropout = nn.Dropout(p=0.5)  # 丟棄 50% 的神經元，防止過擬合

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 卷積 → ReLU 激活 → 最大池化
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(-1, 64 * 62 * 62)  # 展平成一維張量
        x = self.dropout(x)  # Dropout 避免過擬合
        x = self.fc1(x)  # 全連接層輸出
        return x

# 影像轉換處理：調整大小為 256x256 並轉換為張量
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 取得最新的模型檔案名稱
model_path = latest_model_path.modle_path()

# 載入模型
model = CNN().to(device)  # 模型加載至指定設備
model.load_state_dict(torch.load(os.path.join(os.getcwd(), "action_model", model_path)))  # 讀取訓練好的模型參數
model.eval()  # 設置模型為評估模式（不進行梯度計算）

# 讀取標籤檔案（包含馬達數值的對應標籤）
labels_file = os.path.join(os.getcwd(), "action.csv")
labels_df = pd.read_csv(labels_file)

# 使用 MinMaxScaler 進行數據正規化，使數值介於 0 到 1 之間
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(labels_df.iloc[:, 1:].values)  # 對所有數值（不包含標籤名稱）進行擬合

# 反正規化函數，將模型預測的數值還原至原始尺度
def denormalize(y_normalized, scaler):
    return scaler.inverse_transform(y_normalized)

# 設定要預測的影像資料夾
# unlabeled_dir = os.path.join(os.getcwd() , "pretrain_data")  # 預訓練數據
unlabeled_dir = os.path.join(os.getcwd(), "image_test_data")  # 測試數據

# 影像推理函數
def infer_images(directory):
    predictions = []
    for filename in os.listdir(directory):  # 遍歷資料夾中的所有檔案
        if filename.endswith(".jpg") or filename.endswith(".png"):  # 確保為影像檔
            img_path = os.path.join(directory, filename)
            image = Image.open(img_path).convert('RGB')  # 讀取影像並轉換為 RGB
            image = transform(image).unsqueeze(0)  # 應用轉換並新增 batch 維度
            
            with torch.no_grad():  # 不計算梯度，加速推理
                image = image.to(device)
                output = model(image)  # 輸入模型進行預測
                predictions.append((filename, output.cpu().numpy()))  # 儲存結果
    return predictions

# 進行影像推理
predicted_values = infer_images(unlabeled_dir)

# 反正規化預測結果並轉為整數
predicted_motor_values = []
for filename, pred in predicted_values:
    pred_original_scale = np.round(denormalize(pred, scaler)).astype(int)  # 反正規化並四捨五入取整數
    predicted_motor_values.append((filename, pred_original_scale))

# 輸出預測結果
for filename, motor_values in predicted_motor_values:
    print(f'圖片為: {filename}, 預測的馬達數值: {motor_values}')

# 計算評估指標（可選）
# y_true = labels_df.iloc[:, 1:].values  # 取得真實值
# y_pred = np.array([denormalize(pred, scaler).flatten() for _, pred in predicted_values])  # 取得預測值

# mse = mean_squared_error(y_true, y_pred)  # 計算均方誤差 (MSE)
# r2 = r2_score(y_true, y_pred)  # 計算 R² 分數

# print(f'均方差: {mse:.4f}')
# print(f'R² 分數: {r2:.4f}')
