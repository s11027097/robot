# 匯入所需的模組
import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from openai import OpenAI
from sklearn.model_selection import train_test_split
import latest_log_path

# 設定隨機種子，確保每次運行結果一致
def set_random_seed(seed):
    torch.manual_seed(seed)  # 設定PyTorch隨機種子
    torch.backends.cudnn.benchmark = False  # 禁用cudnn的自動選擇最優算法
set_random_seed(4)  # 設定隨機種子為4

# 讀取數據
data = pd.read_csv('action.csv')  # 從CSV文件讀取動作數據
train_texts, valid_texts, train_joints, valid_joints = train_test_split(
    data['filename'],  # 使用“filename”列作為文本輸入
    data.iloc[:, 1:].values,  # 使用數據框中的其他列作為關節值
    test_size=0.15,  # 設定驗證集大小為15%
    random_state=3  # 隨機種子設定為3
)

# 初始化OpenAI API客戶端
api_key = ""  # 請填入OpenAI API金鑰
client = OpenAI(api_key=api_key)
messages = []  # 用於存儲聊天歷史

# 自定義Dataset類別，用於加載訓練和驗證數據
class ActionDataset(Dataset):
    def __init__(self, texts, joints):
        self.texts = texts  # 動作描述文本
        self.joints = joints  # 相應的關節值
    def __len__(self):
        return len(self.texts)  # 返回數據集的大小
    def __getitem__(self, idx):
        # 將文本編碼為向量，並返回對應的關節值
        text_vector = self.encode_text(self.texts.iloc[idx])  # 編碼動作描述文本
        joint_values = torch.tensor(self.joints[idx], dtype=torch.float32)  # 轉換關節值為tensor
        return text_vector, joint_values

    def encode_text(self, text):
        # 利用OpenAI API將動作描述文本轉換為16個關節值
        messages.clear()  # 清空之前的消息
        messages.append({
            "role": "user",
            "content": (
                f"請將以下動作描述轉換為 16 個關節值, 依序為右腳底板,右小腿,右膝蓋,右大腿,右腰部,右手掌,右手臂,右肩膀,左腳底板,左小腿,左膝蓋,左大腿,左腰部,左手掌,左手臂,左肩膀。"
                f"動作描述是：{text}。"
                f"直接返回數值，不要其他文字。例如：500, 346, 621, 678, 500, 576, 801, 725, 500, 654, 379, 322, 500, 424, 199, 275。"
            )
        })
        try:
            # 向OpenAI API發送請求
            chat_completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=100,
                temperature=0.7
            )
            # 解析API回應
            reply = chat_completion.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": reply})  # 儲存助手的回應
            print("馬達數值:", reply)
            joint_tensor = self.process_reply(reply)  # 處理API回應，將其轉換為16個關節值的tensor
            return joint_tensor
        except Exception as e:
            print(f"API 調用失敗: {e}")
            return torch.tensor([0.0] * 16, dtype=torch.float32)  # 如果API調用失敗，返回16個0的tensor

    def process_reply(self, reply):
        try:
            joint_values = list(map(float, reply.split(',')))  # 將回應的字符串轉換為浮點數列表
            if len(joint_values) < 16:  # 如果返回的數字少於16個，報錯
                print(f"獲取的數字不足，獲取到的數字：{joint_values}")
                raise ValueError("數字數量不足")
            return torch.tensor(joint_values[:16], dtype=torch.float32)  # 返回16個關節值的tensor
        except Exception as e:
            print(f"無法將回應轉換為數字: {reply}, 錯誤: {e}")
            return torch.tensor([0.0] * 16, dtype=torch.float32)  # 如果處理回應失敗，返回16個0的tensor

# 檢查是否有可用的GPU，並設定設備為GPU或CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義神經網絡模型
class Action(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(Action, self).__init__()
        # 定義神經網絡的各層
        self.fc1 = nn.Linear(16, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)   
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, 32)     
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(32, 16)     
        self._initialize_weights()  # 初始化權重

    def forward(self, x):
        # 定義前向傳播過程
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu3(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)  # 最後一層輸出16個值
        return x

    def _initialize_weights(self):
        # 初始化所有線性層的權重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

# 計算平均絕對誤差（MAE）
def calculate_mae(predictions, targets):
    error = torch.abs(predictions - targets)  # 計算預測值與真實值之間的誤差
    mae = error.mean()  # 計算MAE
    return mae

# 自定義損失函數，結合了MSE和多樣性懲罰
def custom_loss(predictions, targets, lambda_diversity=0.1):
    mse_loss = nn.MSELoss()(predictions, targets)  # 均方誤差
    diversity_penalty = torch.mean(torch.pdist(predictions, p=2))  # 多樣性懲罰（基於預測值之間的歐氏距離）
    return mse_loss + lambda_diversity * diversity_penalty  # 返回損失

# 讀取日志目錄，獲取最新的學習率
lo_path = os.path.join(os.getcwd(), "action_log")
lofiles = os.listdir(lo_path)
has_files = 1 if any(os.path.isfile(os.path.join(lo_path, f)) for f in lofiles) else 0
if has_files == 1: 
    l_path = latest_log_path.modle_path()  # 查找最新的日志文件
    with open(l_path, 'r') as l_file:
        lines = l_file.readlines()
        second_last_line = lines[-2] if len(lines) > 1 else lines[-1]  # 獲取倒數第二行（通常包含學習率）
        if 'Learning Rate:' in second_last_line:
            lr_str = second_last_line.split('Learning Rate:')[-1].strip()
            lr_ = float(lr_str)  # 解析學習率
            print(lr_)
else:
    lr_ = 0.001  # 如果沒有日志文件，設置默認學習率為0.001

# 初始化模型和優化器
lr = lr_
model = Action().to(device)  # 將模型放置在選擇的設備上（GPU或CPU）
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 使用Adam優化器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.90, patience=10)  # 設定學習率調度器

# 設定訓練參數
batch_size = 4
num_epochs = 1200

# 構建訓練和驗證數據集
train_dataset = ActionDataset(train_texts.reset_index(drop=True), train_joints)
valid_dataset = ActionDataset(valid_texts.reset_index(drop=True), valid_joints)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# 設定模型保存目錄
ckpt_dir = 'action_ckpt'
os.makedirs(ckpt_dir, exist_ok=True)

# 檢查是否有已經保存的模型檔案，並恢復訓練
latest_ckpt = None
latest_epoch = -1
start_epoch = -1
for file in os.listdir(ckpt_dir):
    if file.endswith('.ckpt'):
        epoch = int(file[:-5])
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_ckpt = os.path.join(ckpt_dir, file)
if latest_ckpt:
    checkpoint = torch.load(latest_ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"恢復訓練，從 epoch {start_epoch} 開始...")
else:
    start_epoch = 1
    print("開始新的訓練...")

# 訓練和驗證的日志記錄設定
log_dir = 'action_log'
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, time.strftime("%Y%m%d%H%M%S") + ".log")
model_dir = 'action_model'
os.makedirs(model_dir, exist_ok=True)

# 開始訓練過程
with open(log_file_path, 'w') as log_file:
    for epoch in range(start_epoch, num_epochs):
        model.train()  # 訓練模式
        total_mae_train = 0
        for actions, joint_values in train_dataloader:
            actions = actions.to(device)
            joint_values = joint_values.to(device)
            optimizer.zero_grad()
            outputs = model(actions)  # 預測輸出
            loss = custom_loss(outputs, joint_values)  # 計算損失
            loss.backward()  # 反向傳播
            optimizer.step()  # 更新模型參數
            mae_train = calculate_mae(outputs, joint_values)  # 計算MAE
            total_mae_train += mae_train.item()  # 累加訓練誤差
            predictions_list = outputs.detach().cpu().numpy().round().astype(int).tolist()  # 轉換為數字並四捨五入
            targets_list = joint_values.detach().cpu().numpy().round().astype(int).tolist()
            print(f"Predictions: {predictions_list}")
            print(f"Target: {targets_list}")
            print()
        avg_train_mae = total_mae_train / len(train_dataloader)
        log_message = f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}, MAE: {avg_train_mae:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n'
        print(log_message)
        log_file.write(log_message)
        scheduler.step(loss.item())  # 更新學習率
        ckpt_path = os.path.join(ckpt_dir, str(epoch) + ".ckpt")
        torch.save({
            'epoch': epoch + 1,
            'loss': loss.item(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, ckpt_path)
        torch.save(model.state_dict(), os.path.join(model_dir, str(epoch) + ".pth"))
        model.eval()  # 評估模式
        total_mae_val = 0
        with torch.no_grad():
            for actions, joint_values in valid_dataloader:
                actions = actions.to(device)
                joint_values = joint_values.to(device)
                outputs = model(actions)
                mae_val = calculate_mae(outputs, joint_values)
                total_mae_val += mae_val.item()
        avg_val_mae = total_mae_val / len(valid_dataloader)
        log_message = f'Validation MAE: {avg_val_mae:.4f}\n'
        print(log_message)
        log_file.write(log_message)
