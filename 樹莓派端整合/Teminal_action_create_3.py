import sqlite3  # 引入 SQLite 資料庫模組
import os  # 引入 os 模組來操作檔案
import ActionGroupControl as Act  # 引入機器人動作控制模組

# 在每個動作數據前插入數值 1000，作為時間間隔
def add_ten_hundred(table_widget):
    for sublist in table_widget:
        sublist.insert(0, 1000)  # 插入 1000 作為時間參數
    return table_widget

# 讀取動作名稱檔案
def read_action_path_name():
    file_name = "action_name.txt"
    if os.path.exists(file_name):  # 檢查檔案是否存在
        with open(file_name, "r", encoding="utf-8") as file:
            text = file.read()
    content = str(text)  # 轉換成字串
    return content  # 回傳讀取到的內容

# 寫入新的動作名稱到檔案
def write_action_path_name(temp_name_str):
    file_name = "action_name.txt"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(temp_name_str)

# 儲存動作群組至 SQLite 資料庫
def save_action_group(table_widget, chinese=False):
    if len(table_widget) == 0:  # 如果動作列表為空，則不儲存
        if chinese:
            print('動作列表是空的哦，沒有能保存的')
        else:
            print('The action list is empty, nothing to save')
        return
    
    path = read_action_path_name()  # 讀取當前動作名稱
    temp_name_int = int(path)
    path = "/home/pi/Desktop/110進/2024Project/Robot1_Aciton/" + path  # 設定儲存路徑
    
    if os.path.isfile(path):  # 如果檔案已存在，則刪除
        os.remove(path)
    
    if not path.endswith('.d6a'):
        path += '.d6a'  # 確保檔案副檔名為 .d6a

    try:
        conn = sqlite3.connect(path)  # 連接 SQLite 資料庫
        c = conn.cursor()
        
        # 創建動作群組資料表
        c.execute('''CREATE TABLE ActionGroup(
            [Index] INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL ON CONFLICT FAIL UNIQUE ON CONFLICT ABORT,
            Time INT,
            Servo1 INT,
            Servo2 INT,
            Servo3 INT,
            Servo4 INT,
            Servo5 INT,
            Servo6 INT,
            Servo7 INT,
            Servo8 INT,
            Servo9 INT,
            Servo10 INT,
            Servo11 INT,
            Servo12 INT,
            Servo13 INT,
            Servo14 INT,
            Servo15 INT,
            Servo16 INT);''')
        
        # 將動作數據寫入資料表
        for row in table_widget:
            insert_sql = "INSERT INTO ActionGroup(Time, Servo1, Servo2, Servo3, Servo4, Servo5, Servo6, Servo7, Servo8, Servo9, Servo10, Servo11, Servo12, Servo13, Servo14, Servo15, Servo16) VALUES ("
            insert_sql += ", ".join(map(str, row))  # 將數據轉換成 SQL 語法格式
            insert_sql += ");"
            c.execute(insert_sql)
        
        conn.commit()  # 提交更改
        conn.close()  # 關閉資料庫
        print("Action group saved successfully.")
        
        temp_name_str = str(temp_name_int + 1)  # 更新動作名稱
        write_action_path_name(temp_name_str)
        print("Next_Path_name:", temp_name_str)
    except Exception as e:
        print(f"Error: {e}")  # 錯誤處理，輸出錯誤訊息

# 主執行函式
def main(motor_values):
    updated_table_widget = add_ten_hundred(motor_values)  # 為馬達數值添加時間間隔
    save_action_group(updated_table_widget, chinese=False)  # 儲存動作群組
    
    pa = read_action_path_name()  # 讀取最新的動作名稱
    act = int(pa) - 1  # 計算要執行的動作名稱
    pa = str(act)
    
    # Act.runAction(pa)  # 執行儲存的動作（目前註解）
    # Act.runAction("Stand_still")  # 執行靜止動作（目前註解）