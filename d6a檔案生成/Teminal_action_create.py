import sqlite3
import os
import ActionGroupControl as Act


def save_action_group(table_widget, chinese=False):
    if len(table_widget) == 0:
        if chinese:
            print('動作列表是空的哦，沒有能保存的')
        else:
            print('The action list is empty, nothing to save')
        return

    name = input('Enter save file name: ')
    path = "/home/pi/Desktop/d6a檔案生成/" + name
    if os.path.isfile(path):
        os.remove(path)
    
    if not path.endswith('.d6a'):
        path += '.d6a'

    try:
        conn = sqlite3.connect(path)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE If NOT EXISTS ActionGroup(
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
        
        for row in table_widget:
            insert_sql = "INSERT INTO ActionGroup(Time, Servo1, Servo2, Servo3, Servo4, Servo5, Servo6, Servo7, Servo8, Servo9, Servo10, Servo11, Servo12, Servo13, Servo14, Servo15, Servo16) VALUES ("
            insert_sql += ", ".join(map(str, row))
            insert_sql += ");"
            c.execute(insert_sql)
        
        conn.commit()
        conn.close()
        print("Action group saved successfully.")
    except Exception as e:
        print(f"Error: {e}")

def main():
    table_widget = [
        [500, 500, 388, 500, 594, 500, 600, 870, 0, 500, 612, 500, 406, 500, 400, 130, 1000]
    ]
    save_action_group(table_widget, chinese=False)


if __name__ == "__main__":
    main()