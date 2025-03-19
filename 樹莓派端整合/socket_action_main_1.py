import socket_action_void_2  # 引入處理 socket 相關動作的模組
import Teminal_action_create_3  # 引入處理終端機動作的模組
import action_pygame_4  # 引入處理 pygame 動作的模組
import threading  # 引入多執行緒模組
import os  # 引入作業系統操作模組
import sys  # 引入系統相關模組

# 寫入動作名稱到檔案
def write_action_path_name(temp_name_str):
    file_name = "action_name.txt"  # 設定檔案名稱
    with open(file_name, "w", encoding="utf-8") as file:  # 以寫入模式開啟檔案
        file.write(temp_name_str)  # 將動作名稱寫入檔案
    return temp_name_str  # 回傳寫入的動作名稱

# 讀取動作名稱檔案
def read_action_path_name():
    file_name = "action_name.txt"  # 設定檔案名稱
    if os.path.exists(file_name):  # 檢查檔案是否存在
        with open(file_name, "r", encoding="utf-8") as file:  # 以讀取模式開啟檔案
            text = file.read()  # 讀取檔案內容
    content = str(text)  # 轉換成字串
    return content  # 回傳讀取到的內容

# 執行終端機動作
def run_terminal_action(motor_values):
    Teminal_action_create_3.main(motor_values)  # 調用終端機動作函式

# 執行 pygame 動作
def run_pygame_action(name_path_str):
    action_pygame_4.pygame_void(name_path_str)  # 調用 pygame 動作函式

# 進入無限迴圈來持續執行動作
while True:
    name_path = read_action_path_name()  # 讀取當前動作名稱
    
    if name_path == "1":  # 如果動作名稱為 "1"
        run_pygame_action(name_path)  # 執行 pygame 動作
        name_path_int = int(name_path) + 1  # 將名稱轉換為整數後加 1
        name_path_str = str(name_path_int)  # 轉換回字串
        w_path = write_action_path_name(name_path_str)  # 更新動作名稱到檔案
        print("Next_path_name:", w_path)  # 印出新的動作名稱
        print("==============================================================")
    
    name_path_str = "2"  # 強制設定動作名稱為 "2"
    print("==============================================================")
    
    motor_values = socket_action_void_2.socket_result_msg()  # 透過 socket 接收馬達數值
    print("motor_values:", motor_values)  # 印出接收到的馬達數值
    print("==============================================================")
    
    # 建立執行緒來執行機器人動作與音檔撥放動作
    thread1 = threading.Thread(target=run_terminal_action, args=(motor_values,))
    thread2 = threading.Thread(target=run_pygame_action, args=(name_path_str,))
    
    thread1.start()  # 啟動機器人動作執行緒
    thread2.start()  # 啟動音檔撥放執行緒
    
    thread1.join()  # 等待機器人動作執行緒結束
    thread2.join()  # 等待音檔撥放執行緒結束