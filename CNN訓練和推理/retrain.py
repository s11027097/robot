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

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

set_random_seed(3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = os.path.join(os.getcwd(), "pretrain_data")
image_size = (256, 256)

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

labels_file = os.path.join(os.getcwd(), "action.csv")
labels_df = pd.read_csv(labels_file)

scaler = MinMaxScaler(feature_range=(0, 1))
y_normalized = scaler.fit_transform(labels_df.iloc[:, 1:].values)

# 載入資料集和建立 DataLoader
def load_labeled_data(directory, labels_df):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            image = Image.open(img_path).convert('RGB')
            image = image.resize(image_size)
            images.append(image)
            label_row = labels_df[labels_df['filename'].str.strip() == filename.strip()]
            if label_row.empty:
                continue
            label = label_row.iloc[:, 1:].values[0]
            labels.append(label)
    return images, labels

X, y = load_labeled_data(train_dir, labels_df)

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

y_tensor = torch.tensor(y_normalized, dtype=torch.float32) 

dataset = DataSet(images=X, labels=y_tensor, transform=transform)
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

lr = 0.00009 
model = CNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)

def train_savetype(epoch, model, optimizer, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # 設定從下一個 epoch 開始
    loss = checkpoint['loss']
    return start_epoch, loss

# 載入檢查點的邏輯
ckpt_dir = os.path.join(os.getcwd(), "action_train_savetype")
latest_ckpt = None

# 獲取最新的檢查點文件
for file in os.listdir(ckpt_dir):
    if file.endswith(".ckpt"):
        latest_ckpt = os.path.join(ckpt_dir, file)

# 確認檢查點是否存在
if latest_ckpt is not None:
    start_epoch, _ = load_checkpoint(model, optimizer, latest_ckpt)
    print(f'從檢查點繼續訓練，開始於 epoch: {start_epoch}')
else:
    start_epoch = 0
    print('未找到檢查點，從 epoch 0 開始訓練')

num_epochs = 220
for epoch in range(start_epoch, num_epochs):
    model.train().to(device)
    epoch_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float() 
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # 刪除舊的檢查點（如果存在）
    for file in os.listdir(ckpt_dir):
        if file.endswith(".ckpt"):
            os.remove(os.path.join(ckpt_dir, file))

    # 保存當前的檢查點
    ckpt_path = os.path.join(ckpt_dir, f"Action_{time.strftime('%Y%m%d%H%M%S')}.ckpt")
    train_savetype(epoch, model, optimizer, avg_loss, ckpt_path)

model_dir = os.path.join(os.getcwd(), "action_model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, time.strftime('%Y%m%d%H%M%S') + ".pth")
torch.save(model.state_dict(), model_path)
print(f'已保存模型: {model_path}')