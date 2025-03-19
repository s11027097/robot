import socket  # 匯入 socket 模組，用於網路通訊
import os      # 匯入 os 模組，用於檔案操作
import json    # 匯入 json 模組，用於處理 JSON 格式資料

def socket_result_msg():
    """
    接收來自客戶端的馬達數值與音檔，並儲存音檔。

    回傳:
    motor_values (dict) : 接收到的馬達控制數值
    """
    HOST = '0.0.0.0'  # 監聽所有網路介面的請求
    PORT = 65432      # 設定通訊埠號
    BUFFER_SIZE = 4096  # 設定每次接收的資料大小
    
    # 建立 TCP 伺服器
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 允許重複綁定同一個位址
        s.bind((HOST, PORT))  # 綁定 IP 位址與通訊埠
        s.listen()  # 開始監聽連線
        print("等待連接...")

        conn, addr = s.accept()  # 接受來自客戶端的連線請求
        print('已連接到', addr)

        received_data = b''  # 用於儲存接收到的完整資料

        try:
            # 持續接收資料，直到結束
            while True:
                data = conn.recv(BUFFER_SIZE)
                if not data:
                    break
                received_data += data

            # 尋找分隔符 '\nEND_OF_VALUES\n'，區分馬達數值與音檔
            separator_position = received_data.find(b'\nEND_OF_VALUES\n')

            if separator_position != -1:
                # 解析並解碼馬達控制數值
                value_data = received_data[:separator_position].decode('utf-8')
                motor_values = json.loads(value_data)  
                print(f"接收到的馬達數值: {motor_values}")

                # 取得音檔資料
                audio_data = received_data[separator_position + len('\nEND_OF_VALUES\n'):]
                
                # 設定音檔儲存路徑
                audio_filename = '/home/pi/Desktop/Audio/received_audio.wav'
                with open(audio_filename, 'wb') as audio_file:
                    audio_file.write(audio_data)  # 儲存音檔
                print(f"接收到的音檔已儲存為: {audio_filename}")
            else:
                print("未找到分隔符 'END_OF_VALUES'，請檢查資料格式")

        except Exception as e:
            print(f"處理資料時發生錯誤: {e}")  # 錯誤處理，輸出錯誤訊息

        s.close()  # 關閉伺服器連線

    return motor_values  # 回傳接收到的馬達數值