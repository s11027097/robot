import os
from datetime import datetime
def modle_path():
    folder_path = os.path.join(os.getcwd(), "action_model")
    files = os.listdir(folder_path)
    latest_file = None
    latest_number = -1  

    for file in files:
        filename = os.path.splitext(file)[0]  
        file_number = int(filename)  
        if file_number > latest_number:
            latest_number = file_number
            latest_file = file 

    return latest_file