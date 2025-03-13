import socket

# 設定接收端的IP和端口
host = '0.0.0.0'  # 標註為0.0.0.0來讓它接受所有網絡接口的連接
port = 12345  # 隨便設一個端口號（確保端口未被占用）

# 創建socket對象
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 綁定IP和端口
server_socket.bind((host, port))

# 開始等待連接（最大等待佇列為1）
server_socket.listen(1)

print(f"Server started on {host}:{port}. Waiting for a connection...")

# 等待客戶端連接
client_socket, client_address = server_socket.accept()
print(f"Connection from {client_address} has been established!")

# 接收來自客戶端的訊息
message = client_socket.recv(1024)  # 接收最多1024個字節
print(f"Received message: {message.decode()}")

# 關閉連接
client_socket.close()
server_socket.close()
