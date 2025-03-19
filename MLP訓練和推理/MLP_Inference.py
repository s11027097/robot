import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from openai import OpenAI
from sklearn.model_selection import train_test_split

def GPT_MLP(msg):
    # 定義 MLP 模型，包含三層全連接層與 ReLU 激活函數及 Dropout
    class Action(nn.Module):
        def __init__(self, dropout_rate=0.5):
            super(Action, self).__init__()
            self.fc1 = nn.Linear(16, 64)  # 第一層: 16 -> 64
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(64, 32)  # 第二層: 64 -> 32
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout_rate)
            self.fc3 = nn.Linear(32, 16)  # 第三層: 32 -> 16
            self.relu3 = nn.ReLU()

        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.dropout1(x)
            x = self.relu2(self.fc2(x))
            x = self.dropout2(x)
            x = self.relu3(self.fc3(x))
            return x

    # 設定 OpenAI API 金鑰 (應避免硬編碼，建議使用環境變數管理)
    api_key = "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"
    client = OpenAI(api_key=api_key)

    # 透過 GPT-4 API 解析動作描述，轉換為 16 個關節值
    def encode_text(text):
        messages = []
        messages.append({
            "role": "user",
            "content": (
                f"請將以下動作描述轉換為 16 個關節值, 依序為右腳底板,右小腿,右膝蓋,右大腿,右腰部,右手掌,右手臂,右肩膀,"
                f"左腳底板,左小腿,左膝蓋,左大腿,左腰部,左手掌,左手臂,左肩膀。動作描述是：{text}。"
                f"直接返回數值，不要其他文字。例如：500, 346, 621, 678, 500, 576, 801, 725, 500, 654, 379, 322, 500, 424, 199, 275。"
            )
        })

        try:
            chat_completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=100,
                temperature=0.3
            )
            
            reply = chat_completion.choices[0].message.content.strip()
            joint_values = process_reply(reply)  # 解析回傳的數值
            return joint_values

        except Exception as e:
            print(f"API 調用失敗: {e}")
            return torch.tensor([0.0] * 16, dtype=torch.float32)

    # 解析 GPT-4 回傳的關節數值字串，轉換為 Tensor
    def process_reply(reply):
        try:
            joint_values = list(map(float, reply.split(',')))
            if len(joint_values) < 16:
                raise ValueError("數字數量不足")
            return torch.tensor(joint_values[:16], dtype=torch.float32)
        except Exception as e:
            print(f"無法將回應轉換為數字: {reply}, 錯誤: {e}")
            return torch.tensor([0.0] * 16, dtype=torch.float32)

    # 使用 MLP 模型預測新的關節數值
    def predict_action(text_description):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = Action().to(device)
        m_lopth = "C:\\Users\\User\\Desktop\\AI_entity_robot\\184.pth"  # 預訓練模型權重路徑
        model.load_state_dict(torch.load(m_lopth))  # 加載模型權重
        model.eval()

        joint_values = encode_text(text_description).unsqueeze(0).to(device)  # 將文本轉換為關節值並調整形狀

        with torch.no_grad():
            predicted_joint_values = model(joint_values).cpu().numpy().astype(int).flatten().tolist()  # 轉換為整數列表

        return predicted_joint_values

    text_input = msg  # 接收輸入文本
    predicted_joint_values = predict_action(text_input)  # 預測動作關節值
    return predicted_joint_values  # 回傳預測結果