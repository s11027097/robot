import socket  # 匯入 socket 模組，用於網路通訊
import json    # 匯入 json 模組，用於處理 JSON 格式資料
import os      # 匯入 os 模組，用於檔案操作

host = ''  # 設定目標裝置的 IP 位址
PORT = '65432'  # 設定目標裝置的通訊埠號

def send_data_to_raspberry_pi(action_values, audio_path, host):
    """
    傳送動作數值與音檔至樹莓派

    參數:
    action_values (dict) : 需傳送的馬達控制數值
    audio_path (str)     : 音檔路徑
    host (str)           : 樹莓派的 IP 位址
    """
    try:
        # 建立 TCP 連線
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, PORT))  # 連線到目標主機與指定通訊埠
            
            # 將動作數值轉換為 JSON 字串並傳送
            action_json = json.dumps(action_values)
            s.sendall(action_json.encode('utf-8'))
            print(f"傳送的馬達數值: {action_values}")

            # 傳送標記字串，表示動作數值傳輸結束
            s.sendall(b'\nEND_OF_VALUES\n')

            # 開啟並讀取音檔後傳送
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
                s.sendall(audio_data)
            print(f"傳送的音檔: {audio_path}")

    except Exception as e:
        print(f"傳送資料時發生錯誤: {e}")  # 錯誤處理，輸出錯誤訊息