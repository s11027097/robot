import socket

# 設定樹莓派接收端的IP和端口
host = '樹莓派的IP地址'  # 將此處的 '樹莓派的IP地址' 替換為實際的樹莓派IP
port = 12345  # 與接收端一致的端口號

# 創建socket對象
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 連接到樹莓派的服務器
client_socket.connect((host, port))

# 發送消息
message = "Hello from the computer!"
client_socket.send(message.encode())

# 關閉連接
client_socket.close()
print(f"Message sent: {message}")
