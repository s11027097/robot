import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import logging
import time

# 設定隨機種子，確保結果可重現
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

set_random_seed(3)

# 設定裝置為 GPU（如果可用），否則使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 設定資料集目錄與影像大小
train_dir = os.path.join(os.getcwd(), "pretrain_data")
image_size = (256, 256)

# 自定義 Dataset 類別，用於加載影像與標籤
class DataSet(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 定義 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 定義卷積層與池化層
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)  # 第一層卷積，輸入通道數=3（RGB），輸出通道數=32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 第二層卷積，輸出通道數=64
        self.pool = nn.MaxPool2d(kernel_size=2)  # 最大池化層
        self.fc1 = nn.Linear(64 * 62 * 62, 16)  # 全連接層，輸入展平後的大小為 64*62*62
        self.dropout = nn.Dropout(p=0.5)  # Dropout 以防止過擬合

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 第一層卷積 + ReLU + 池化
        x = self.pool(F.relu(self.conv2(x)))  # 第二層卷積 + ReLU + 池化
        x = x.view(-1, 64 * 62 * 62)  # 展平張量
        x = self.dropout(x)  # Dropout
        x = self.fc1(x)  # 全連接層
        return x

# 讀取標籤檔案
labels_file = os.path.join(os.getcwd(), "action.csv")
labels_df = pd.read_csv(labels_file)

# 使用 MinMaxScaler 進行標籤正規化（將數值縮放到 0 到 1）
scaler = MinMaxScaler(feature_range=(0, 1))
y_normalized = scaler.fit_transform(labels_df.iloc[:, 1:].values)

# 加載影像與標籤
def load_labeled_data(directory, labels_df):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # 過濾圖片檔案
            img_path = os.path.join(directory, filename)
            image = Image.open(img_path).convert('RGB')  # 轉換為 RGB
            image = image.resize(image_size)  # 調整大小
            images.append(image)

            # 根據文件名稱查找對應的標籤
            label_row = labels_df[labels_df['filename'].str.strip() == filename.strip()]
            if label_row.empty:
                continue
            label = label_row.iloc[:, 1:].values[0]  # 取得標籤數值
            labels.append(label)
    return images, labels

# 加載訓練數據
X, y = load_labeled_data(train_dir, labels_df)

# 定義數據增強（Data Augmentation）與轉換
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(p=0.5),  # 以 50% 機率水平翻轉
    transforms.RandomRotation(degrees=30),  # 隨機旋轉 ±30 度
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 調整亮度與對比度
    transforms.ToTensor(),  # 轉換為 PyTorch 張量
])

# 轉換標籤為張量格式
y_tensor = torch.tensor(y_normalized, dtype=torch.float32) 

# 建立 Dataset 與 DataLoader
dataset = DataSet(images=X, labels=y_tensor, transform=transform)
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

# 設定學習率與 Adam 優化器的權重衰減（L2 正則化）
lr = 0.00009
weight_decay = 0.000001
model = CNN().to(device)
criterion = nn.MSELoss()  # 使用均方誤差（MSE）作為損失函數
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# 定義模型儲存函式
def train_savetype(epoch, model, optimizer, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)

# 設定日誌記錄
log_dir = os.path.join(os.getcwd(), "log")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "Action_" + time.strftime("%Y%m%d%H%M%S") + '.log')
logging.basicConfig(filename=log_file, level=logging.INFO)

# 訓練模型
num_epochs = 220
ckpt_dir = os.path.join(os.getcwd(), "action_train_savetype")
os.makedirs(ckpt_dir, exist_ok=True)

for epoch in range(num_epochs):
    model.train().to(device)
    epoch_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()  # 移動到 GPU（如果可用）
        outputs = model(inputs)  # 前向傳播
        loss = criterion(outputs, labels)  # 計算損失
        optimizer.zero_grad()  # 梯度歸零
        loss.backward()  # 反向傳播
        optimizer.step()  # 更新權重
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # 清除舊的 checkpoint，保留最新的
    for file in os.listdir(ckpt_dir):
        if file.endswith(".ckpt"):
            os.remove(os.path.join(ckpt_dir, file))

    # 儲存 checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"Action_{time.strftime('%Y%m%d%H%M%S')}.ckpt")
    train_savetype(epoch, model, optimizer, avg_loss, ckpt_path)

# 儲存最終模型
model_dir = os.path.join(os.getcwd(), "action_model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, time.strftime('%Y%m%d%H%M%S') + ".pth")
torch.save(model.state_dict(), model_path)
print(f'已保存模型: {model_path}')
