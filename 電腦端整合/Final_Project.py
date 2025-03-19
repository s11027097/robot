# -*- coding: utf-8 -*-
from openai import OpenAI
import socket
import time
import os
import sys
import json 
import pandas as pd
import csv
api_key = ""

client = OpenAI(api_key=api_key)

result = "大家好"

prompt = [  "站立","舉右手","右手微微舉高","右手舉高","右手放下","右手往內舉","左手舉高","舉左手",
            "左手微微舉高","左手放下","左手往內舉","雙手張開","雙手舉高","雙手舉到頭頂","雙腳蹲低",
            "雙手抱胸","右手向右揮動","右手向左揮動","左手向右揮動","左手向左揮動","右腳蹲低","右腳蹲下",
            "右腳跪地","右腳向右抬起","右腳向前抬起","右腳向後抬起","左腳蹲低","左腳蹲下","左腳跪地", 
            "左腳向左抬起", "左腳向前抬起" ,"左腳向後抬起", "左腳向前微微抬起", "左腳向後微微抬起" ,
            "左腳向左微微抬起",  "右腳向右微微抬起" , "右腳向後微微抬起", "右腳向前微微抬起" ,
            "左手完全放下" , "右手完全放下","右手舉起來" ]

messages_1 = [{
    "role": "assistant", 
    "content": "你現在是相聲演員，風格幽默且機智，語速快，擅長模仿各種人物，句子簡練且精確。"
               "你擅長即興創作，能夠迅速反應並把聽眾帶入情境中。常用獨特的語音和口音來增加趣味，"
               "並且在表演中展現出強烈的個人特色，吸引觀眾的注意力，讓人不由自主地笑出聲來。"
               "每句話請用20個字說完"
}]

messages_2 = [{
    "role": "assistant", 
    "content": "你現在是相聲演員，風格輕鬆幽默，語氣溫和且富有節奏感，擅長轉換語氣來增加情感層次。"
               "你喜歡在表演中加入一些意想不到的反轉，讓觀眾感到出乎意料的驚喜和歡笑。"
               "你的笑點通常來自日常生活的細節和語言的巧妙運用，讓人聽了不僅能會心一笑，"
               "還能感同身受。你的表演充滿智慧，常常讓觀眾在笑過後還會沉思。每句話請用20個字說完"
}]

messages_action = [
    {
        "role": "assistant",
        "content": (
            "根據文字:" + result +
            "，把文字轉為以下的動作，" + ", ".join(prompt) +
            "，並且是多個動作集合的。例如:大家好:就是左手舉高+左手向右揮動+左手向左揮動，只要回答我動作就可以了且每次回答不能是相同的動作。"
        )
    }
]
# 定義函數，讓 GPT 生成對應的動作
def Crosstalk_action(msg):
    messages_action.append({"role": "user", "content": msg})
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages_action,
    )
    action_msg = chat_completion.choices[0].message.content.replace('\n', '')
    messages_action.append({"role": "assistant", "content": action_msg})
    return action_msg  

# 定義第一位相聲演員的對話生成函數
def Crosstalk_actor_1(msg):
    messages_1.append({"role": "user", "content": msg})
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages_1,
        max_tokens=50,
        temperature=0.9
    )
    ai_msg = chat_completion.choices[0].message.content.replace('\n', '')
    messages_1.append({"role": "assistant", "content": ai_msg})
    return ai_msg  

# 定義第二位相聲演員的對話生成函數
def Crosstalk_actor_2(msg):
    messages_2.append({"role": "user", "content": msg})
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages_2,
        max_tokens=50,
        temperature=0.9
    )
    ai_msg = chat_completion.choices[0].message.content.replace('\n', '')
    messages_2.append({"role": "assistant", "content": ai_msg})
    return ai_msg  


# 設定 Raspberry Pi 的音頻通訊埠
PORT_AUDIO = 65432  

# 定義傳輸函數，將動作與音訊發送至 Raspberry Pi
def send_data_to_raspberry_pi(action_values, audio_path, host):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, PORT_AUDIO))
            
            action_json = json.dumps(action_values)
            s.sendall(action_json.encode('utf-8'))
            print(f"傳送的馬達數值: {action_values}")
            
            s.sendall(b'\nEND_OF_VALUES\n')

            with open(audio_path, 'rb') as f:
                audio_data = f.read()
                s.sendall(audio_data)
            print(f"傳送的音檔: {audio_path}")
    except Exception as e:
        print(f"傳送資料時發生錯誤: {e}")

# 載入 GPT-SoVITS 語音模型
# ====================================================================================================================
"""
這個是GPT-SoVITS的函數引入，需要使用就把底下區域程式移動到外圍
"""
gpt_so_vits_path = os.path.join(os.getcwd(), "GPT-SoVITS-main", "GPT_SoVITS")
os.chdir(os.path.join(os.getcwd(), "GPT-SoVITS-main")) 
sys.path.append(gpt_so_vits_path) 
import inference_webui
import inference_webui_2
# ====================================================================================================================
# 載入 MLP 監督學習模型
"""
MLP監督學習
"""
import MLP_Inference
result = "直接開始相聲"
action_data = pd.read_csv(r"C:\Users\user\Desktop\AI_entity_robot\action.csv")
check = 0
while True:
    if check == 0:
        print("!!!^1號^!!!")
        result = Crosstalk_actor_1(result)
        print("=====================================================================================================================================")
        print("相聲劇本:")
        print(result)
        print("=====================================================================================================================================")
        print("相聲動作:")
        result_action = Crosstalk_action(result)
        result_action = result_action.replace(" ","")
        result_action_list = []
        index = 0
        for s in range(len(result_action)):
            if result_action[s] == "+":
                result_action_list.append(result_action[index:s])
                index = s
            if s == len(result_action) - 1:
                result_action_list.append(result_action[index:s + 1])
                index = s
        cleaned_list = [action.replace('+', '') for action in result_action_list]
        print(cleaned_list)
        # ====================================================================================================================
        action_value_list = []
        for action in cleaned_list:
            predict_value = MLP_Inference.GPT_MLP(action)  
            action_value_list.append(predict_value)
        print("Predict:\n",action_value_list)
        send_data_to_raspberry_pi(action_value_list, inference_webui.sovits(result), '172.20.10.10')
        # ====================================================================================================================
        check = 1
    else:
        print("!!!^2號^!!!")
        result = Crosstalk_actor_2(result)
        print("=====================================================================================================================================")
        print("相聲劇本:")
        print(result)
        print("=====================================================================================================================================")
        print("相聲動作:")
        result_action = Crosstalk_action(result)

        result_action = result_action.replace(" ","")
        result_action_list = []
        index = 0
        for s in range(len(result_action)):
            if result_action[s] == "+":
                result_action_list.append(result_action[index:s])
                index = s
            if s == len(result_action) - 1:
                result_action_list.append(result_action[index:s + 1])
                index = s
        cleaned_list = [action.replace('+', '') for action in result_action_list]
        print(cleaned_list)
        # ====================================================================================================================
        action_value_list = []
        for action in cleaned_list:
            predict_value = MLP_Inference.GPT_MLP(action)  
            action_value_list.append(predict_value)
        print("Predict:\n",action_value_list)
        send_data_to_raspberry_pi(action_value_list, inference_webui_2.sovits(result), '172.20.10.10') 
        # ====================================================================================================================
        check = 0
    time.sleep(10)