"""
UR5e 机械臂双摄 Diffusion Policy 部署测试脚本 (终极完整版)
"""
import torch
import numpy as np
import mujoco.viewer
import time
import sys

# 引入官方库路径
sys.path.append('../diffusion_policy') 
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace
from envs.ur5e_pusht_env import UR5ePushTEnv

def main():
    # 1. 配置模型路径
    ckpt_path = "checkpoints/latest.ckpt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"正在加载模型: {ckpt_path}")
    payload = torch.load(open(ckpt_path, 'rb'), map_location=device)
    cfg = payload['cfg']
    
    # 2. 恢复网络结构并加载权重
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    policy.eval().to(device)
    print("✅ 模型加载成功！")

    # 3. 初始化环境
    env = UR5ePushTEnv()
    
    # 🌟 报错的元凶就是漏了这一行：必须重置环境并获取第一帧真实画面！
    current_obs, info = env.reset()
    
    n_obs_steps = cfg.n_obs_steps
    obs_history = {
        'img_overhead': [],
        'img_wrist': [],
        'state': []
    }

    print("🚀 启动 MuJoCo 物理引擎，神经网络即将接管...")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step_count = 0
        while viewer.is_running():
            # ---------------------------
            # A. 提取真实图像与状态 
            # ---------------------------
            img_overhead_raw = current_obs['image_overhead']
            img_wrist_raw = current_obs['image_wrist']
            current_state = current_obs['qpos'][:6].copy()

            # ---------------------------
            # B. 预处理数据 (转为模型需要的 [C, H, W] 并归一化)
            # ---------------------------
            img_o = np.moveaxis(img_overhead_raw, -1, 0).astype(np.float32) / 255.0
            img_w = np.moveaxis(img_wrist_raw, -1, 0).astype(np.float32) / 255.0
            
            obs_history['img_overhead'].append(img_o)
            obs_history['img_wrist'].append(img_w)
            obs_history['state'].append(current_state)
            
            if len(obs_history['state']) > n_obs_steps:
                obs_history['img_overhead'].pop(0)
                obs_history['img_wrist'].pop(0)
                obs_history['state'].pop(0)
            
            if len(obs_history['state']) == n_obs_steps:
                # 拼接 Batch
                obs_dict = {
                    'img_overhead': torch.from_numpy(np.stack(obs_history['img_overhead'])).unsqueeze(0).to(device),
                    'img_wrist': torch.from_numpy(np.stack(obs_history['img_wrist'])).unsqueeze(0).to(device),
                    'state': torch.from_numpy(np.stack(obs_history['state'])).unsqueeze(0).to(device).float()
                }

                # ---------------------------
                # C. 神经网络推理 (生成动作块)
                # ---------------------------
                with torch.no_grad():
                    # action_pred 的 shape 是 [1, horizon, 6] (例如 [1, 16, 6])
                    action_pred = policy.predict_action(obs_dict)['action']
                
                # 提取这一整块预测轨迹: shape [horizon, 6]
                trajectory = action_pred[0].cpu().numpy()
                
                del obs_dict
                del action_pred
                torch.cuda.empty_cache()
                
                # ---------------------------
                # D. 官方策略：动作分块执行 (Action Chunking)
                # ---------------------------
                # 官方推荐配置：预测 16 步，执行前 8 步 (n_action_steps)
                n_action_steps = cfg.n_action_steps # 通常是 8
                
                print(f"Step {step_count} | 规划完成，开始丝滑执行接下来 {n_action_steps} 步动作...")
                
                # 连续执行这 n_action_steps 步，期间不进行神经网络推理
                for step_i in range(n_action_steps):
                    target_action = trajectory[step_i]
                    current_qpos = env.data.qpos[:6].copy()
                    
                    # 依然保留底层的物理平滑 (控制频率10Hz -> 物理频率500Hz)
                    physics_steps = 50 
                    for p_i in range(physics_steps):
                        alpha = (p_i + 1) / physics_steps
                        interpolated_action = (1 - alpha) * current_qpos + alpha * target_action
                        
                        env.data.ctrl[:6] = interpolated_action
                        mujoco.mj_step(env.model, env.data)
                        viewer.sync()
                        
                    # 此时环境已经往前推进了 0.1 秒
                    # 我们需要更新 current_obs，但不需要喂给神经网络，仅仅是为了保持环境状态更新
                    if step_i < n_action_steps - 1:
                        # 盲走阶段：不重新计算图像观测，纯推进物理
                        pass 
                    else:
                        # Chunk 的最后一步走完，睁开眼睛，获取最新图像用于下一次大脑推理
                        current_obs, _, _, _, _ = env.step(target_action)
                        
                step_count += 1

if __name__ == "__main__":
    main()