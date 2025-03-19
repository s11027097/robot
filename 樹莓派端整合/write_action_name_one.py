import os

def write_action_path_name():
    file_name = "action_name.txt"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write("1")
        
def read_action_path_name():
    file_name = "action_name.txt"
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as file:
            text = file.read()
    content = str(text)
    return content

def delete_action():
    name_path = read_action_path_name()
    name_path_int = int(name_path)
    folder_path = "/home/pi/Desktop/110進/2024Project/Robot1_Aciton/"

    files = os.listdir(folder_path)

    files_to_delete = [f for f in files if f.endswith(".d6a") and f[:-4].isdigit() and 2 <= int(f[:-4]) <= name_path_int]
    for f in files_to_delete:
        f = f.strip()
        if f.endswith(".d6a") and f[:-4].isdigit() and 2 <= int(f[:-4]) <= name_path_int:
            print(f"符合條件: {f}")
            
    for file_name in files_to_delete:
        file_path = os.path.join(folder_path, file_name)
        try:
            os.remove(file_path)
            print(f"已刪除: {file_path}")
        except Exception as e:
            print(f"刪除失敗: {file_path}，錯誤: {e}")
            
delete_action()
write_action_path_name()
