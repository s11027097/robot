import pygame  # 引入 Pygame 用於播放音頻
import os  # 引入 os 模組來操作檔案

def read_action_path_name():
    """
    讀取 action_name.txt 檔案內容，並返回作為音檔名稱。
    """
    file_name = "action_name.txt"
    if os.path.exists(file_name):  # 檢查檔案是否存在
        with open(file_name, "r", encoding="utf-8") as file:
            text = file.read()
    content = str(text)  # 轉換成字串
    return content  # 回傳讀取到的內容

def pygame_void(name_path):
    """
    使用 Pygame 播放音檔。
    如果 name_path == "1"，則播放固定的開場音檔，否則播放 action_name.txt 指定的音檔。
    """
    audio_path = read_action_path_name()  # 讀取音檔名稱
    
    # 初始化 Pygame 音頻模組，設定取樣頻率為 32000 Hz
    pygame.mixer.init(frequency=32000)
    
    if name_path == "1":
        # 加載預設的開場音檔
        pygame.mixer.music.load("/home/pi/Desktop/110進/2024Project/Audio/crosstalk_open1.wav")
    else:
        # 加載動態指定的音檔
        pygame.mixer.music.load("/home/pi/Desktop/110進/2024Project/Audio/" + audio_path + ".wav")
    
    # 播放音頻
    pygame.mixer.music.play()
    
    # 等待音頻播放結束
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # 控制迴圈間隔，避免佔用過多 CPU