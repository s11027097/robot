import paho.mqtt.client as mqtt

# 設定MQTT broker的地址和端口
broker = "test.mosquitto.org"  # 同樣使用公共的Mosquitto broker
port = 1883
topic = "home/test"  # 發送的主題

# 創建MQTT客戶端實例
client = mqtt.Client()

# 連接到MQTT broker
client.connect(broker, port, 60)

# 發送消息
message = "Hello from the computer!"
client.publish(topic, message)

print(f"Message '{message}' sent to topic '{topic}'")

# 斷開連接
client.disconnect()
