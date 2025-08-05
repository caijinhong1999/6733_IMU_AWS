import asyncio
import torch
import torch.nn as nn
import numpy as np
import csv
import json
import os
from bleak import BleakClient
import paho.mqtt.client as mqtt

# MQTT configuration
BROKER = "10.166.179.5"
PORT = 1883
SUB_TOPIC = "/modelPublish"
PUB_TOPIC = "/RSSI_IMU"

# BLE configuration
READ_UUID = "4A981236-1CC4-E7C1-C757-F1267DD021E8"
WRITE_UUID = "4A981235-1CC4-E7C1-C757-F1267DD021E8"
BLE_ADDRESS = "D2:26:F4:24:FE:5A"

# CSV save path
CSV_PATH = "imu_data.csv"

# Model label mapping
LABEL_MAP = {0: "drink", 1: "sleep", 2: "forward", 3: "fall"}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
STATUS_DESCRIPTION = {
    -1: "outside the fence",
     0: "drinking",
     1: "sleeping",
     2: "forward",
     3: "abnormal"
}

# BiLSTM + MultiHeadAttention Model Definition
class BiLSTMWithMultiHeadAttention(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, num_classes=4, num_heads=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        return self.fc(attn_out.mean(dim=1))

# loading model
model = BiLSTMWithMultiHeadAttention()
model.load_state_dict(torch.load("imu_model.pt", map_location="cpu"))
model.eval()

imu_buf = [None] * 6
saved_rows = 0
latest_coord = {}  # cow_id -> [x, y, is_out]

# BLE notification parsing
def parse_notification(txt):
    if len(txt) < 4: return None
    sensor, axis, value = txt[0], txt[1], txt[2:]
    try: return sensor, axis, int(value) / 1000.0
    except: return None

def handle_notification(_, data):
    global imu_buf, saved_rows
    txt = data.decode(errors="ignore").strip()
    parsed = parse_notification(txt)
    if not parsed: return
    sensor, axis, val = parsed
    i = {"a": 0, "g": 3}[sensor] + {"x": 0, "y": 1, "z": 2}[axis]
    imu_buf[i] = val
    if None not in imu_buf:
        print(f"Receiving {saved_rows + 1:02d} line data：{imu_buf}")
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow(imu_buf)
        saved_rows += 1
        imu_buf = [None] * 6

# IMU Reasoning function
def predict_behavior():
    with open(CSV_PATH, "r") as f:
        rows = list(csv.reader(f))[1:]  # Skip title line
    if len(rows) < 20:
        print("IMU data is less than 20 lines, return unknown")
        return "unknown"
    recent = rows[-20:]
    print(f"Current total number of data rows: {len(rows)}, The last 20 lines used for prediction:")
    for r in recent:
        print(r)
    data = np.array([[float(x) for x in row] for row in recent]).reshape(1, 20, 6).astype(np.float32)
    with torch.no_grad():
        input_tensor = torch.from_numpy(data)
        output = model(input_tensor)
        pred_label = torch.argmax(output, dim=1).item()
        return LABEL_MAP[pred_label]

# Determine whether it is a boundary
def is_boundary(x, y):
    return x in {1, 16} or y in {1, 8}  # Boundary determination for 8x16 area

# Main logic after receiving coordinates
def on_message(client, userdata, msg):
    global saved_rows
    payload = json.loads(msg.payload.decode())
    print("Received:", payload)

    for cow_id, (x, y, original_status) in payload.items():
        latest_coord[cow_id] = [x, y, original_status]

        async def collect_and_predict():
            global saved_rows
            saved_rows = 0

            async with BleakClient(BLE_ADDRESS) as ble_client:
                await ble_client.start_notify(READ_UUID, handle_notification)
                await ble_client.write_gatt_char(WRITE_UUID, b"s")
                while saved_rows < 20:
                    await asyncio.sleep(0.1)
                await ble_client.stop_notify(READ_UUID)

            action = predict_behavior()
            print(f"Predicted behavior: {action}")

            if action == "forward" and is_boundary(x, y):
                predicted_status = -1  # 出界
            elif action in REVERSE_LABEL_MAP:
                predicted_status = REVERSE_LABEL_MAP[action]
            else:
                predicted_status = -2

            description = STATUS_DESCRIPTION.get(predicted_status, "unknown state")
            print(f"The {cow_id} now is {description}")

            result = {cow_id: [x, y, original_status, predicted_status]}
            client.publish(PUB_TOPIC, json.dumps(result))
            print("Published:", result)

        asyncio.run(collect_and_predict())

# Start MQTT client
def main():
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    client.subscribe(SUB_TOPIC)
    print("Listening for /modelPublish messages...")
    client.loop_forever()

if __name__ == "__main__":
    main()
