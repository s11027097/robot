import socket  # 引入 socket 模組來建立伺服器
import os  # 引入 os 模組來操作檔案
import json  # 引入 json 模組來解析馬達數值

# 讀取動作名稱檔案
def read_action_path_name():
    file_name = "action_name.txt"  # 設定檔案名稱
    if os.path.exists(file_name):  # 檢查檔案是否存在
        with open(file_name, "r", encoding="utf-8") as file:  # 以讀取模式開啟檔案
            text = file.read()  # 讀取檔案內容
    content = str(text)  # 轉換成字串
    return content  # 回傳讀取到的內容

# 建立 socket 伺服器，接收馬達數值與音檔
def socket_result_msg():
    HOST = '0.0.0.0'  # 允許來自所有網絡介面的連線
    PORT = 65432  # 設定監聽的端口號
    BUFFER_SIZE = 4096  # 設定緩衝區大小

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # 創建 TCP socket
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 設定 socket 選項，允許重複使用位址
        s.bind((HOST, PORT))  # 綁定 IP 和 PORT
        s.listen()  # 開始監聽連線請求
        print("等待連接...")

        conn, addr = s.accept()  # 接受來自客戶端的連線
        print('已連接到', addr)

        received_data = b''  # 初始化變數來存儲接收到的資料

        try:
            # 持續接收資料
            while True:
                data = conn.recv(BUFFER_SIZE)  # 從客戶端接收資料
                if not data:
                    break  # 若無資料則跳出迴圈
                received_data += data  # 累積收到的資料

            # 查找分隔符 '\nEND_OF_VALUES\n'，區分馬達數值和音檔
            separator_position = received_data.find(b'\nEND_OF_VALUES\n')
            if separator_position != -1:
                # 提取馬達數值部分
                value_data = received_data[:separator_position].decode('utf-8')  # 轉換為字串
                motor_values = json.loads(value_data)  # 使用 JSON 解析成 Python 物件
                
                # 讀取當前動作名稱作為音檔名稱
                audio_path = read_action_path_name()
                
                # 提取音檔部分
                audio_data = received_data[separator_position + len('\nEND_OF_VALUES\n'):]  # 取得音檔的位元組資料
                audio_filename = "/home/pi/Desktop/110進/2024Project/Audio/" + audio_path + ".wav"  # 設定音檔儲存路徑
                
                # 將音檔寫入指定路徑
                with open(audio_filename, 'wb') as audio_file:
                    audio_file.write(audio_data)
            else:
                print("未找到分隔符 'END_OF_VALUES'，請檢查資料格式")
        except Exception as e:
            print(f"處理資料時發生錯誤: {e}")  # 捕獲錯誤並印出錯誤訊息
        s.close()  # 關閉 socket 連線
    return motor_values  # 回傳解析出的馬達數值