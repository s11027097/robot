import os
import shutil
log_dir = 'log'
action_model_dir = 'action_model'
action_train_savetype_dir = 'action_train_savetype'

if os.path.exists(log_dir) and os.path.isdir(log_dir):
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
else:
    print("資料夾不存在。")

if os.path.exists(action_model_dir) and os.path.isdir(action_model_dir):
    for filename in os.listdir(action_model_dir):
        file_path = os.path.join(action_model_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
else:
    print("資料夾不存在。")

if os.path.exists(action_train_savetype_dir) and os.path.isdir(action_train_savetype_dir):
    for filename in os.listdir(action_train_savetype_dir):
        file_path = os.path.join(action_train_savetype_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
else:
    print("資料夾不存在。")