"""
运行环境: mujoco_env
目标: 监听 SO101 的 UDP 广播，转化为空间 XYZ 增量，遥控 MuJoCo 中的 UR5e 并录制数据。
"""
import os
import time
import numpy as np
import threading
import socket
import json
import mujoco.viewer

from envs.ur5e_pusht_env import UR5ePushTEnv
from control.osc_controller import IKController

class SO101TeleopUDP:
    def __init__(self, init_ee_pos, scale_factor=0.003):
        self.target_pos = init_ee_pos.copy() 
        self.scale_factor = scale_factor 
        
        self.last_hardware_xyz = None
        self.is_clutch_pressed = False 
        self.latest_hardware_data = None
        
        # 录制相关
        self.is_recording = False
        self.current_episode = []
        self.recorded_episodes = 0
        
        # 启动 UDP 监听后台线程
        self.running = True
        self.listen_thread = threading.Thread(target=self._udp_listener_loop, daemon=True)
        self.listen_thread.start()

    def _udp_listener_loop(self):
        """后台极速接收 UDP 数据"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 5005))
        sock.settimeout(1.0)
        
        print("\n🎧 [接收端] 正在监听 UDP 端口 5005...")
        while self.running:
            try:
                data_bytes, _ = sock.recvfrom(1024)
                self.latest_hardware_data = json.loads(data_bytes.decode('utf-8'))
            except socket.timeout:
                continue 
            except Exception as e:
                pass

    def _angles_to_xyz(self, data):
        """
        【空间翻译官】：将 SO101 的角度映射为虚拟的 XYZ 坐标
        这里使用最直观的线性角度映射，方便你后续调整方向（加减号）
        """
        # 提取角度
        pan = data.get('pan', 0.0)      # 底座左右旋转
        lift = data.get('lift', 0.0)    # 大臂抬起/放下
        elbow = data.get('elbow', 0.0)  # 小臂伸缩
        
        # 映射逻辑 (核心调参区)：
        # X 轴 (前后): 由 Lift 和 Elbow 共同决定
        x = (lift * 0.5) + (elbow * 0.5) 
        # Y 轴 (左右): 直接由 Pan 决定 (如果方向反了，把 pan 前面加个负号)
        y = pan 
        # Z 轴 (上下): 由 Lift 决定 (向上抬起时 Z 变大)
        z = -lift 
        
        return np.array([x, y, z])

    def get_target_pos(self):
        """计算并返回 UR5e 的目标 XYZ"""
        if self.latest_hardware_data is None:
            return self.target_pos.copy()
            
        data = self.latest_hardware_data
        
        # 1. 离合器逻辑：根据你终端输出的 "离合=40.2"，你可以设定一个阈值
        # 比如夹爪数值 > 50 认为是按下离合器（允许 UR5e 移动）
        current_clutch = (data.get('gripper', 0.0) > 50.0)
        
        # 2. 计算当前 SO101 对应的虚拟 XYZ
        current_hardware_xyz = self._angles_to_xyz(data)
        
        # 3. 离合器状态切换检测
        if self.last_hardware_xyz is None or not current_clutch:
            # 没按离合器，或者刚启动：更新基准点，UR5e 保持不动
            self.last_hardware_xyz = current_hardware_xyz
            self.is_clutch_pressed = current_clutch
            return self.target_pos.copy()
            
        # 4. 按下离合器时，计算移动增量 (Delta)
        delta_xyz = current_hardware_xyz - self.last_hardware_xyz
        
        # 5. 将增量放大并叠加到 UR5e 上
        self.target_pos += delta_xyz * self.scale_factor
        
        # 更新基准点
        self.last_hardware_xyz = current_hardware_xyz
        self.is_clutch_pressed = current_clutch
        
        # 6. 安全工作空间限制 (防撞桌子)
        self.target_pos[0] = np.clip(self.target_pos[0], 0.35, 0.75)  # X限制
        self.target_pos[1] = np.clip(self.target_pos[1], -0.3, 0.3)   # Y限制
        self.target_pos[2] = np.clip(self.target_pos[2], 0.435, 0.8)  # Z限制
        
        return self.target_pos.copy()

def main():
    # 1. 初始化环境与控制器
    env = UR5ePushTEnv()
    obs, info = env.reset()
    ik_controller = IKController(env.model, env.data)
    ik_controller.sync_target_orientation()
    
    # 2. 初始化遥控器
    init_ee_pos = env.data.site_xpos[ik_controller.site_id].copy()
    teleop = SO101TeleopUDP(init_ee_pos, scale_factor=0.003) 
    
    cmd_qpos = env.data.qpos[:6].copy()
    
    print("="*60)
    print("🎮 [SO101 主从遥操控制台] 已启动！")
    print("操作提示：")
    print("  - 抓紧夹爪 (离合>50) : 激活遥控，UR5e 会跟随 SO101 移动")
    print("  - 松开夹爪 (离合<50) : 暂停遥控，可在此期间把 SO101 拿回中心位置")
    print("  - 键盘退格键 (Backspace) : 可在 MuJoCo 窗口内重置 T 型块")
    print("="*60)

    # 3. MuJoCo 物理循环
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # 获取目标坐标并平滑限制
            target_pos = teleop.get_target_pos()
            current_qpos = env.data.qpos[:6].copy()
            
            action = ik_controller.calculate_joint_targets(target_pos, cmd_qpos)
            max_drift = 0.05 
            cmd_qpos = np.clip(action, current_qpos - max_drift, current_qpos + max_drift)
            
            env.step(cmd_qpos)
            viewer.sync()
            
            # 维持 50Hz 刷新率
            time_until_next_step = 0.02 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()