"""
运行环境: mujoco_env
目标: 监听 UDP 端口获取 SO101 数据，计算空间增量，映射给 MuJoCo 中的 UR5e。
"""
import time
import numpy as np
import threading
import socket
import json

class SO101TeleopUDP:
    def __init__(self, init_ur5e_pos, scale_factor=2.0):
        self.target_pos = init_ur5e_pos.copy() 
        self.scale_factor = scale_factor 
        
        self.last_so101_pos = None
        self.is_clutch_pressed = False 
        
        # 最新的硬件数据缓存
        self.latest_hardware_data = None
        
        # 启动 UDP 监听后台线程
        self.running = True
        self.listen_thread = threading.Thread(target=self._udp_listener_loop, daemon=True)
        self.listen_thread.start()

    def _udp_listener_loop(self):
        """后台高频接收 UDP 数据的线程"""
        UDP_IP = "127.0.0.1"
        UDP_PORT = 5005
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        # 设置超时，防止线程死锁
        sock.settimeout(1.0)
        
        print(f"🎧 [接收端] 正在监听端口 {UDP_PORT}，等待 SO101 数据...")
        
        while self.running:
            try:
                data_bytes, _ = sock.recvfrom(1024) # 接收最大 1024 字节
                data_dict = json.loads(data_bytes.decode('utf-8'))
                self.latest_hardware_data = data_dict
            except socket.timeout:
                continue # 没有收到数据，继续等
            except Exception as e:
                print(f"UDP 接收出错: {e}")

    def _calculate_fk_simple(self, data):
        """
        【极其重要】：你需要在这里把 3 个角度转化为 XYZ 坐标。
        这只是一个简化的伪映射例子，你需要根据 SO101 真实运动方向调整！
        """
        # 将度数转为弧度
        pan_rad = np.radians(data['pan'])
        lift_rad = np.radians(data['lift'])
        
        # 粗略映射：
        # Pan (底座旋转) 对应空间中的 Y 轴左右平移
        # Lift (大臂抬起) 对应空间中的 Z 轴高度 和 X 轴的前后伸展
        x = np.cos(lift_rad) * 0.2  # 假定臂长 20cm
        y = np.sin(pan_rad) * 0.2
        z = np.sin(lift_rad) * 0.2
        
        return np.array([x, y, z])

    def get_target_pos(self):
        if self.latest_hardware_data is None:
            return self.target_pos.copy()
            
        data = self.latest_hardware_data
        
        # 1. 根据夹爪角度判断离合器 (假设夹爪数值 > 80 认为是闭合)
        # 根据你实际的硬件反馈数值进行调整！
        self.is_clutch_pressed = (data['gripper'] > 80.0) 
        
        # 2. 算出现实末端的 XYZ
        current_so101_pos = self._calculate_fk_simple(data)
        
        if self.last_so101_pos is None or not self.is_clutch_pressed:
            self.last_so101_pos = current_so101_pos
            return self.target_pos.copy()
            
        # 3. 计算增量并映射
        delta_pos = current_so101_pos - self.last_so101_pos
        
        # 这里可能需要根据坐标系对换轴或加负号
        delta_pos_mapped = np.array([delta_pos[0], delta_pos[1], delta_pos[2]]) 
        
        self.target_pos += delta_pos_mapped * self.scale_factor
        self.last_so101_pos = current_so101_pos
        
        # 边界保护
        self.target_pos[0] = np.clip(self.target_pos[0], 0.35, 0.75) 
        self.target_pos[1] = np.clip(self.target_pos[1], -0.3, 0.3)
        self.target_pos[2] = np.clip(self.target_pos[2], 0.435, 0.8) 
        
        return self.target_pos.copy()

    def close(self):
        self.running = False
        self.listen_thread.join()