import time
from ubluepy import Service, Characteristic, UUID, Peripheral, constants
from machine import Pin, I2C
import imu
from time import sleep_ms

# 初始化 I2C
bus = I2C(1, scl=Pin(15), sda=Pin(14))
imu = imu.IMU(bus)

# 全局状态
current_sensor = 0  # 0: accel, 1: gyro
current_axis = 0    # 0: x, 1: y, 2: z
samples_count = 0   # 已采集样本计数
is_sampling = False # 采样状态标志
current_group = [0.0] * 6  # 缓存当前组6维数据：[a_x, a_y, a_z, g_x, g_y, g_z]

# 获取当前轴的数据并缓存，完成一组时打印
def get_imu():
    global current_sensor, current_axis, samples_count, current_group
    try:
        if current_sensor == 0:
            data = imu.accel()
            prefix = 'a'  # accel
        elif current_sensor == 1:
            data = imu.gyro()
            prefix = 'g'  # gyro
        else:
            return "error"
        
        # 获取当前轴原始浮点值（未乘以1000）
        raw_value = data[current_axis]
        # 转换为整数用于传输（乘以1000）
        value = int(raw_value * 1000)
        
        # 缓存数据到当前组（按顺序对应CSV格式）
        if current_sensor == 0:  # 加速度计：a_x→0, a_y→1, a_z→2
            current_group[current_axis] = raw_value
        else:  # 陀螺仪：g_x→3, g_y→4, g_z→5
            current_group[3 + current_axis] = raw_value
        
        # 格式化为传输字符串（如 "ax-324"）
        data_str = f"{prefix}{['x', 'y', 'z'][current_axis]}{value:+05d}"
        
        # 更新状态并检查是否完成一组
        if current_axis == 2:  # z轴发送完
            current_axis = 0    # 重置到x轴
            current_sensor = (current_sensor + 1) % 2  # 切换传感器
            
            # 当完成一组数据（加速度+陀螺仪的6个轴）时打印
            if current_sensor == 0:  # 确保是完整一组（切换回加速度计时代表组完成）
                samples_count += 1
                # 按目标格式打印（保留3位小数，逗号分隔）
                print(",".join([f"{v:.3f}" for v in current_group]))
                # 重置缓存准备下一组
                current_group = [0.0] * 6
        else:
            current_axis += 1  # 切换到下一轴
            
        return data_str
    except Exception as e:
        print(f"IMU error: {e}")
        return "error"

def event_handler(id, handle, data):
    global periph, custom_read_char, notif_enabled, is_sampling, samples_count
    
    # GAP 连接事件
    if id == constants.EVT_GAP_CONNECTED:
        pass
    
    # GAP 断开事件
    elif id == constants.EVT_GAP_DISCONNECTED:
        periph.advertise(device_name="z5542386")
    
    # GATT 写事件
    elif id == constants.EVT_GATTS_WRITE:
        if handle == 16:  # 写特征（接收开始命令）
            print("Received command:", data)
            if data == b's':
                # 开始采样时重置状态
                is_sampling = True
                samples_count = 0
                current_sensor = 0
                current_axis = 0
                current_group = [0.0] * 6  # 重置缓存
                print("Start sampling 20 sets of 6D data")
        elif handle == 19:  # CCCD（通知开关）
            if int(data[0]) == 1:
                notif_enabled = True
                print("Notifications enabled")
            else:
                notif_enabled = False
                print("Notifications disabled")

# BLE 服务和特征配置
notif_enabled = False
custom_svc_uuid = UUID("4A981234-1CC4-E7C1-C757-F1267DD021E8")
custom_wrt_char_uuid = UUID("4A981235-1CC4-E7C1-C757-F1267DD021E8")
custom_read_char_uuid = UUID("4A981236-1CC4-E7C1-C757-F1267DD021E8")

custom_svc = Service(custom_svc_uuid)
custom_wrt_char = Characteristic(custom_wrt_char_uuid, props=Characteristic.PROP_WRITE)
custom_read_char = Characteristic(custom_read_char_uuid, props=Characteristic.PROP_READ | Characteristic.PROP_NOTIFY, attrs=Characteristic.ATTR_CCCD)

custom_svc.addCharacteristic(custom_wrt_char)
custom_svc.addCharacteristic(custom_read_char)

periph = Peripheral()
periph.addService(custom_svc)
periph.setConnectionHandler(event_handler)
periph.advertise(device_name="z5542386")

# 主循环
while True:
    if notif_enabled and is_sampling:
        if samples_count < 20:
            imu_data = get_imu()
            if imu_data != "error":
                b_msg = imu_data.encode()
                try:
                    custom_read_char.write(b_msg)
                except OSError as e:
                    print(f"Write error: {e}")
        else:
            # 完成20组采样后停止
            is_sampling = False
            print("Sampling completed")
    sleep_ms(10)  # 控制采样间隔