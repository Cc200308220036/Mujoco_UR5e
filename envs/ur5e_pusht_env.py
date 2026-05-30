"""
实现目标：升级 Gymnasium 环境接口，支持双视角（全局+腕部）RGB 图像的渲染输出。
输入 (Action)：
    - 类型：np.ndarray，形状 (6,)，连续值。控制 6 个关节目标角度。
输出 (Returns of step/reset)：
    - obs (dict): 
        包含低维状态 ('qpos', 't_cube_pos') 
        以及高维图像 ('image_overhead', 'image_wrist')，格式为 numpy.uint8 数组。
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import transforms3d # 用于欧拉角转四元数，没有的话 pip install transforms3d
import random

class UR5ePushTEnv(gym.Env):
    # Diffusion Policy 通常使用较小分辨率以提升训练速度，这里默认设为 256x256
    def __init__(self, xml_path="envs/assets/ur5e_scene.xml", max_steps=300, render_size=(256, 256)):
        super().__init__()
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"找不到模型文件: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 【新增】：初始化离屏渲染器
        self.render_size = render_size
        self.renderer = mujoco.Renderer(self.model, height=render_size[0], width=render_size[1])
        
        self.max_steps = max_steps
        self.current_step = 0
        
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32)
        self.t_cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "t_cube_joint")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        
        # 1. 机械臂初始姿态 (加入微小随机扰动，提升模型鲁棒性)
        base_home_qpos = np.array([3.1415, -1.57, 1.57, -1.57, -1.57, 0.0])
        noise = np.random.uniform(-0.05, 0.05, size=6)
        home_qpos = base_home_qpos + noise
        self.data.qpos[:6] = home_qpos
        self.data.ctrl[:6] = home_qpos 
        
        # 2. 获取 T 型块状态向量的起始索引
        idx = self.model.jnt_qposadr[self.t_cube_joint_id]
        
        # 3. 彻底的领域随机化：生成随机的 X、Y 坐标与 360 度 Yaw 角
        t_x = np.random.uniform(0.45, 0.65)
        t_y = np.random.uniform(-0.2, 0.2)
        rand_yaw = np.random.uniform(-np.pi, np.pi) 
        
        # (调试专用：如果确认没问题了，这行 print 可以注释掉)
        print(f"\n🔄 [环境重置] T块新坐标 -> X: {t_x:.3f}, Y: {t_y:.3f}, Yaw: {(rand_yaw * 180 / np.pi):.1f}度")
        
        # 4. 核心修复：纯数学手解绕 Z 轴旋转的四元数，彻底杜绝坐标轴错位
        qw = np.cos(rand_yaw / 2.0)
        qx = 0.0
        qy = 0.0
        qz = np.sin(rand_yaw / 2.0)
        quat = [qw, qx, qy, qz]
        
        # 5. 将坐标和姿态写入物理引擎 (Z轴高度保持 0.45 避免穿模)
        self.data.qpos[idx : idx+3] = [t_x, t_y, 0.45]
        self.data.qpos[idx+3 : idx+7] = quat
        
        # 6. 速度和加速度清零，防止残留动能导致物体乱飞
        self.data.qvel[:] = 0.0 
        self.data.qacc[:] = 0.0 
        
        # 7. 强制物理引擎推演一步，让坐标立即生效并完成碰撞检测
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}
        
        return self._get_obs(), {}
    def step(self, action):
        self.data.ctrl[:6] = action
        for _ in range(25):
            mujoco.mj_step(self.model, self.data)
            
        self.current_step += 1
        obs = self._get_obs()
        terminated = False 
        truncated = self.current_step >= self.max_steps
        
        return obs, 0.0, terminated, truncated, {}

    def _get_obs(self):
        qpos = self.data.qpos[:6].copy()
        t_cube_qpos_idx = self.model.jnt_qposadr[self.t_cube_joint_id]
        t_cube_pos = self.data.qpos[t_cube_qpos_idx : t_cube_qpos_idx+3].copy()
        
        # 【新增】：渲染全局图像
        self.renderer.update_scene(self.data, camera="overhead_cam")
        img_overhead = self.renderer.render().copy() # Shape: (H, W, 3)
        
        # 【新增】：渲染腕部图像
        self.renderer.update_scene(self.data, camera="wrist_cam")
        img_wrist = self.renderer.render().copy()

        return {
            "qpos": qpos.astype(np.float32),
            "t_cube_pos": t_cube_pos.astype(np.float32),
            "image_overhead": img_overhead,
            "image_wrist": img_wrist
        }