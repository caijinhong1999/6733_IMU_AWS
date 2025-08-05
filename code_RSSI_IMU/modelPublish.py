# === 第一套代码: 每6秒发送固定坐标到 /modelPublish ===

import time
import json
import paho.mqtt.client as mqtt

broker = "10.166.179.5"
port = 1883
topic = "/modelPublish"

client = mqtt.Client()
client.connect(broker, port)

payload = {
    # "cow1": [5, 2, 0],  # 3 fall
    # "cow1": [8, 3, 0] # 2 forward
    # "cow1": [8, 3, 0] # 1 sleep
    # "cow1": [2, 2, 0] # 0 drink
    "cow1": [1, 3, 0] # 2 forward outside
}

while True:
    client.publish(topic, json.dumps(payload))
    print("Sent:", payload)
    time.sleep(6)
