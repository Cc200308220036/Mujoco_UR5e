"""
实现目标：键盘遥操作 UR5e 机械臂完成 Push-T 任务，带完美重置与绝对坐标同步。
"""
import os
import time
import numpy as np
import mujoco.viewer
from pynput import keyboard

from envs.ur5e_pusht_env import UR5ePushTEnv
from control.osc_controller import IKController

class KeyboardTeleop:
    def __init__(self, init_ee_pos, step_size=0.002):
        self.step_size = step_size
        self.cmd_vel = np.zeros(3)
        
        # 在遥控器内部维护绝对目标位置
        self.target_pos = init_ee_pos.copy() 
        
        self.is_recording = False
        self.need_reset = False
        self.recorded_episodes = []
        self.current_episode = []

        # 检测已有的 episode 数量，避免覆盖已有数据，实现断点续录
        save_dir = os.path.join("dataset", "raw")
        if os.path.exists(save_dir):
            existing_files = [f for f in os.listdir(save_dir) if f.startswith("episode_") and f.endswith(".npz")]
            if existing_files:
                indices = [int(f.split("_")[1].split(".")[0]) for f in existing_files]
                self.start_ep_idx = max(indices)
                print(f"📂 检测到已录制 {self.start_ep_idx} 组数据，新数据将从 episode_{(self.start_ep_idx + 1):04d}.npz 开始保存。")
            else:
                self.start_ep_idx = 0
        else:
            self.start_ep_idx = 0

        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def sync_target(self, current_ee_pos):
        """环境重置时调用，同步遥控器的目标点到机械臂新出生点"""
        self.target_pos = current_ee_pos.copy()
        self.cmd_vel = np.zeros(3)

    def on_press(self, key):
        if key == keyboard.Key.up: self.cmd_vel[0] = 1.0
        elif key == keyboard.Key.down: self.cmd_vel[0] = -1.0
        elif key == keyboard.Key.left: self.cmd_vel[1] = 1.0
        elif key == keyboard.Key.right: self.cmd_vel[1] = -1.0
        elif key == keyboard.Key.page_up: self.cmd_vel[2] = 1.0
        elif key == keyboard.Key.page_down: self.cmd_vel[2] = -1.0
        elif key == keyboard.Key.backspace: 
            self.need_reset = True
        elif key == keyboard.Key.enter: 
            self.toggle_recording()

    def on_release(self, key):
        if key in [keyboard.Key.up, keyboard.Key.down]: 
            self.cmd_vel[0] = 0.0
        elif key in [keyboard.Key.left, keyboard.Key.right]: 
            self.cmd_vel[1] = 0.0
        elif key in [keyboard.Key.page_up, keyboard.Key.page_down]: 
            self.cmd_vel[2] = 0.0

    def toggle_recording(self):
        if not self.is_recording:
            print("\n[🔴 录制开始] 请操作机械臂完成 Push-T 任务...")
            self.current_episode = []
            self.is_recording = True
        else:
            print(f"\n[⏹️ 录制结束] 本次采集了 {len(self.current_episode)} 帧数据。")
            if len(self.current_episode) > 50:
                self.recorded_episodes.append(self.current_episode)
                ep_idx = self.start_ep_idx + len(self.recorded_episodes)
                print(f"当前已成功录制 {ep_idx} 个有效 Episode！ (本阶段已录制 {len(self.recorded_episodes)} 个)")
                
                # 【核心新增】：调用保存函数，将内存数据实时写入硬盘
                self.save_episode(self.current_episode, ep_idx)
            else:
                print("数据帧数过少，已丢弃。")
            self.is_recording = False

    def save_episode(self, episode_data, ep_idx):
        """将单次 Episode 数据解包并保存为包含图像的 .npz 文件"""
        save_dir = os.path.join("dataset", "raw")
        os.makedirs(save_dir, exist_ok=True)

        obs_qpos_list = []
        obs_t_cube_list = []
        img_overhead_list = []
        img_wrist_list = []
        action_list = []

        # 遍历提取图像和状态
        for frame in episode_data:
            obs_qpos_list.append(frame['obs']['qpos'])
            obs_t_cube_list.append(frame['obs']['t_cube_pos'])
            img_overhead_list.append(frame['obs']['image_overhead']) # 提取全局图像
            img_wrist_list.append(frame['obs']['image_wrist'])       # 提取腕部图像
            action_list.append(frame['action'])

        # 转换为 Numpy 矩阵
        obs_qpos_array = np.array(obs_qpos_list, dtype=np.float32)
        obs_t_cube_array = np.array(obs_t_cube_list, dtype=np.float32)
        img_overhead_array = np.array(img_overhead_list, dtype=np.uint8)
        img_wrist_array = np.array(img_wrist_list, dtype=np.uint8)
        action_array = np.array(action_list, dtype=np.float32)

        # 保存为压缩格式 (因为包含大量图像，必须用 savez_compressed)
        file_path = os.path.join(save_dir, f"episode_{ep_idx:04d}.npz")
        np.savez_compressed(
            file_path, 
            qpos=obs_qpos_array, 
            t_cube_pos=obs_t_cube_array, 
            image_overhead=img_overhead_array,
            image_wrist=img_wrist_array,
            action=action_array
        )
        
        print(f"💾 视觉数据已成功落盘保存至: {file_path}")

    def get_target_pos(self):
        """直接更新并返回内部维护的 target_pos"""
        self.target_pos += self.cmd_vel * self.step_size
        self.target_pos[0] = np.clip(self.target_pos[0], 0.3, 0.9)
        self.target_pos[1] = np.clip(self.target_pos[1], -0.4, 0.4)
        # 上限放宽到 1.0，包容刚重置时的高空出生点
        self.target_pos[2] = np.clip(self.target_pos[2], 0.435, 1.0)
        return self.target_pos.copy()


def main():
    # 1. 初始化物理环境
    env = UR5ePushTEnv()
    obs, info = env.reset()
    
    # 2. 初始化 IK 控制器并同步初始朝向
    ik_controller = IKController(env.model, env.data)
    ik_controller.sync_target_orientation()
    
    # 3. 初始化后，获取此时机械臂末端绝对位置，传给键盘遥控器
    init_ee_pos = env.data.site_xpos[ik_controller.site_id].copy()
    teleop = KeyboardTeleop(init_ee_pos, step_size=0.003) 
    
    # 初始控制指令缓存
    cmd_qpos = env.data.qpos[:6].copy()
    
    print("="*60)
    print("🎮 [键盘遥操控制台] 完美重置版启动！")
    print("操作提示：")
    print("  - 方向键 ↑ / ↓ : 沿 X 轴 前进/后退")
    print("  - 方向键 ← / → : 沿 Y 轴 左移/右移")
    print("  - PageUp / PageDown : 沿 Z 轴 上升/下降")
    print("  - Enter (回车) : 开始 / 停止 录制当前 Episode")
    print("  - Backspace (退格键) : 完美纯净重置环境")
    print("="*60)

    record_hz = 10
    steps_per_record = int(50 / record_hz) 
    step_count = 0

    # 4. 仿真主循环
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # 【处理完美重置】
            if teleop.need_reset:
                teleop.cmd_vel = np.zeros(3) 
                
                obs, info = env.reset()
                ik_controller.sync_target_orientation()
                cmd_qpos = env.data.qpos[:6].copy()
                
                # 同步遥控器的目标点到新出生点
                new_ee_pos = env.data.site_xpos[ik_controller.site_id].copy()
                teleop.sync_target(new_ee_pos)
                
                teleop.is_recording = False
                teleop.need_reset = False
                
                print("✨ [重置成功] 环境已纯净刷新，键盘控制坐标已强制同步！")
                time.sleep(0.1) 
                continue
            
            # 【正常遥操控制流】
            target_pos = teleop.get_target_pos()
            current_qpos = env.data.qpos[:6].copy()
            
            action = ik_controller.calculate_joint_targets(target_pos, cmd_qpos)
            
            max_drift = 0.05 
            cmd_qpos = np.clip(action, current_qpos - max_drift, current_qpos + max_drift)
            
            obs, reward, terminated, truncated, info = env.step(cmd_qpos)
            viewer.sync()
            
            # 记录数据
            step_count += 1
            if teleop.is_recording and step_count % steps_per_record == 0:
                frame_data = {
                    'obs': obs.copy(), 
                    'action': cmd_qpos.copy().astype(np.float32)
                }
                teleop.current_episode.append(frame_data)
            
            time_until_next_step = 0.02 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()