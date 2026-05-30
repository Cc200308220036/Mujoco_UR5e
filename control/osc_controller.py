"""
实现目标：6自由度逆运动学控制器 (6D IK)。同时约束末端的位置 (XYZ) 和姿态 (Roll-Pitch-Yaw)，保证推杆始终垂直朝下。
输入：目标末端位置 (target_pos) 和当前关节角度 (current_qpos)。
输出：目标关节角度 (target_qpos)。
"""

import numpy as np
import mujoco

class IKController:
    def __init__(self, model, data, site_name="attachment_site", damping=0.05):
        self.model = model
        self.data = data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.damping = damping
        self.nv = self.model.nv
        
        self.jacp = np.zeros((3, self.nv))
        self.jacr = np.zeros((3, self.nv))
        
        # 记录机械臂初始（垂直朝下）时的基准旋转矩阵
        self.target_xmat = None

    def sync_target_orientation(self):
        """在每次 env.reset() 后调用，锁死当前推杆的完美朝下姿态"""
        self.target_xmat = self.data.site_xmat[self.site_id].copy().reshape(3, 3)

    def calculate_joint_targets(self, target_pos, current_qpos):
        # 1. 获取当前位置和姿态
        current_pos = self.data.site_xpos[self.site_id]
        current_xmat = self.data.site_xmat[self.site_id].reshape(3, 3)
        
        # 2. 计算位置误差 (3D)
        error_pos = target_pos - current_pos
        
        # 3. 计算姿态误差 (3D) - 使用轴向叉乘法逼近角速度误差
        if self.target_xmat is not None:
            error_rot = 0.5 * (np.cross(current_xmat[:, 0], self.target_xmat[:, 0]) +
                               np.cross(current_xmat[:, 1], self.target_xmat[:, 1]) +
                               np.cross(current_xmat[:, 2], self.target_xmat[:, 2]))
        else:
            error_rot = np.zeros(3)
            
        # 将位置误差和姿态误差合并为 6D 误差向量
        error = np.concatenate([error_pos, error_rot])
        
        # 4. 获取完整的 6D 雅可比矩阵
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site_id)
        J_pos = self.jacp[:, :6]
        J_rot = self.jacr[:, :6]
        J = np.vstack([J_pos, J_rot]) # 形状: (6, 6)
        
        # 5. DLS 求解 6D IK
        lambda_sq = self.damping ** 2
        J_Jt = J @ J.T
        I = np.eye(6)
        
        J_inv_damped = J.T @ np.linalg.inv(J_Jt + lambda_sq * I)
        delta_q = J_inv_damped @ error
        
        return current_qpos[:6] + delta_q