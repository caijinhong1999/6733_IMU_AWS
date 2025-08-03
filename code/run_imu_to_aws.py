import asyncio
import torch
import torch.nn as nn
import numpy as np
import csv
import json
import os
from bleak import BleakClient, BleakScanner
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# ==== BLE UUID ====
READ_UUID = "4A981236-1CC4-E7C1-C757-F1267DD021E8"
WRITE_UUID = "4A981235-1CC4-E7C1-C757-F1267DD021E8"
TARGET_NAME = "z5542386"  # BLE device name

# ==== CSV ====
CSV_PATH = os.path.join(os.path.dirname(__file__), "imu_data.csv")
CSV_HEADER = ["a_x", "a_y", "a_z", "g_x", "g_y", "g_z"]

# ==== AWS IoT ====
def setup_aws_client():
    client = AWSIoTMQTTClient("testDevice")
    client.configureEndpoint("a2l45kiu0a7t0e-ats.iot.ap-southeast-2.amazonaws.com", 8883)
    client.configureCredentials(
        "AmazonRootCA1.pem",
        "7fe292cd3213d3e112d793449d320a5daa82e3949f8b5365b35dcb0f17deb96a-private.pem.key",
        "7fe292cd3213d3e112d793449d320a5daa82e3949f8b5365b35dcb0f17deb96a-certificate.pem.crt"
    )
    client.connect()
    return client

aws_client = setup_aws_client()


# ==== 数据增强 ====
def augment_data(data):
    # data shape: (20, 6)
    data = data.copy()

    # 添加高斯噪声（μ=0, σ=0.01）
    noise = np.random.normal(0, 0.01, size=data.shape)
    data += noise

    # 随机缩放（0.9~1.1）
    scale = np.random.uniform(0.9, 1.1)
    data *= scale

    # 掉轴（随机屏蔽某一轴为0，模拟数据丢失）
    if np.random.rand() < 0.3:  # 30%概率执行
        axis_to_zero = np.random.choice(6)
        data[:, axis_to_zero] = 0.0

    return data

# ==== 模型结构定义 ====v
class BiLSTMWithMultiHeadAttention(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, num_classes=4, num_heads=4):
        super(BiLSTMWithMultiHeadAttention, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        x = self.dropout(attn_out.mean(dim=1))
        return self.fc(x)

model = BiLSTMWithMultiHeadAttention()
model.load_state_dict(torch.load("imu_model.pt", map_location="cpu"))
model.eval()
LABEL_MAP = {0:"drink", 1:"sleep", 2:"forward", 3:"fall"}

# ==== 初始化 CSV ====
def init_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)
        print("imu_data.csv had been created")
    else:
        print("imu_data.csv already exists, will continue to add data")

# ==== BLE 接收处理 ====
imu_buf = [None] * 6
saved_rows = 0

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
    if sensor not in {"a", "g"}: return
    i = {"a": 0, "g": 3}[sensor] + {"x": 0, "y": 1, "z": 2}[axis]
    imu_buf[i] = val
    if None not in imu_buf:
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow(imu_buf)
        saved_rows += 1
        print(f"Receiving {saved_rows:02d} line data：{imu_buf}")
        imu_buf = [None] * 6

def predict_from_csv():
    with open(CSV_PATH, "r") as f:
        rows = list(csv.reader(f))[1:]
    if len(rows) < 20:
        print("Data is less than 20 rows, unable to make predictions")
        return "unknown"
    recent = rows[-20:]
    data = np.array([[float(x) for x in row] for row in recent]).reshape(1, 20, 6).astype(np.float32)
    with torch.no_grad():
        data = augment_data(data)
        input_tensor = torch.from_numpy(data)
        output = model(input_tensor)
        pred_label = torch.argmax(output, dim=1).item()
        return LABEL_MAP[pred_label]

# ==== 自动扫描 BLE ====
# ==== 主函数 ====
async def main():
    while True:
        global saved_rows
        saved_rows = 0
        init_csv()

        try:
            address = "D2:26:F4:24:FE:5A"  # 已知设备地址，跳过扫描
        except Exception as e:
            print(f"BLE scan failed: {e}")
            continue

        async with BleakClient(address) as client:
            await client.start_notify(READ_UUID, handle_notification)

            while True:
                cmd = input("\nEnter's' to start collection, enter 'x' to exit the program:").strip().lower()
                if cmd == "x":
                    print("exit")
                    aws_client.disconnect()
                    return
                elif cmd != "s":
                    print("Invalid input, please enter's' or 'x'")
                    continue

                print("Start receiving IMU data ..")
                await client.write_gatt_char(WRITE_UUID, b"s")

                while saved_rows < 20:
                    await asyncio.sleep(0.1)


                action = predict_from_csv()
                msg = f'The cow is {action}'
                print(f"Forecast results: {msg}")

                aws_client.publish("cow/imu/status", json.dumps({"state": msg}), 1)
                print("Sent to AWS")

                break  # 回到下一轮等待 's' 或 'x'

if __name__ == "__main__":
    asyncio.run(main())
