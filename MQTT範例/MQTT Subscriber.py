import paho.mqtt.client as mqtt

# 設定MQTT broker的地址和端口
broker = "test.mosquitto.org"  # 可以使用公共的Mosquitto broker
port = 1883
topic = "home/test"  # 訂閱的主題

# 當連接上MQTT broker時的回呼函數
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    # 訂閱指定的主題
    client.subscribe(topic)

# 當接收到消息時的回呼函數
def on_message(client, userdata, msg):
    print(f"Received message '{msg.payload.decode()}' on topic '{msg.topic}'")

# 創建MQTT客戶端實例
client = mqtt.Client()

# 設定回呼函數
client.on_connect = on_connect
client.on_message = on_message

# 連接到MQTT broker
client.connect(broker, port, 60)

# 開始接收消息
client.loop_forever()
