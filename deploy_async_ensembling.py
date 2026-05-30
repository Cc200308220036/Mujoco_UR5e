"""
实现目标：带终端监控的异步多线程部署框架。50Hz 物理控制 + 10Hz 视觉推理。
"""
import torch
import numpy as np
import mujoco.viewer
import time
import threading
import collections
import sys

sys.path.append('../diffusion_policy') 
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace
from envs.ur5e_pusht_env import UR5ePushTEnv

# ==========================================
# 1. 全局共享内存 (线程通信枢纽)
# ==========================================
shared_data_lock = threading.Lock()
shared_latest_obs = None       
shared_new_trajectory = None   
is_running = True              

# ==========================================
# 2. 异步大脑：后台推理守护线程 (10Hz)
# ==========================================
def inference_worker(policy, device, n_obs_steps):
    global shared_latest_obs, shared_new_trajectory, is_running
    
    obs_history = {
        'img_overhead': collections.deque(maxlen=n_obs_steps),
        'img_wrist': collections.deque(maxlen=n_obs_steps),
        'state': collections.deque(maxlen=n_obs_steps)
    }
    
    infer_count = 0
    print("\n🧠 [后台大脑] 神经推理线程已成功点火，等待视觉信号...\n")
    
    while is_running:
        step_start_time = time.time()
        current_obs = None
        
        # 安全抓取最新画面
        with shared_data_lock:
            if shared_latest_obs is not None:
                current_obs = shared_latest_obs
                shared_latest_obs = None # 抓取后清空
        
        if current_obs is not None:
            # 预处理
            img_o = np.moveaxis(current_obs['image_overhead'], -1, 0).astype(np.float32) / 255.0
            img_w = np.moveaxis(current_obs['image_wrist'], -1, 0).astype(np.float32) / 255.0
            state = current_obs['qpos'][:6].copy().astype(np.float32)
            
            obs_history['img_overhead'].append(img_o)
            obs_history['img_wrist'].append(img_w)
            obs_history['state'].append(state)
            
            # 推理条件满足
            if len(obs_history['state']) == n_obs_steps:
                obs_dict = {
                    'img_overhead': torch.from_numpy(np.stack(obs_history['img_overhead'])).unsqueeze(0).to(device),
                    'img_wrist': torch.from_numpy(np.stack(obs_history['img_wrist'])).unsqueeze(0).to(device),
                    'state': torch.from_numpy(np.stack(obs_history['state'])).unsqueeze(0).to(device)
                }
                
                # 执行张量计算
                with torch.no_grad():
                    action_pred = policy.predict_action(obs_dict)['action']
                
                trajectory = action_pred[0].cpu().numpy()
                target_qpos = trajectory[0] # 取出轨迹第一步用于打印
                
                # 【终端监控 1】：让大脑大声喊出它的计算结果！
                infer_count += 1
                calc_time = (time.time() - step_start_time) * 1000
                print(f"🧠 [大脑] 第 {infer_count} 次思考完成 (耗时 {calc_time:.1f}ms) | 预测前三个关节: [{target_qpos[0]:.2f}, {target_qpos[1]:.2f}, {target_qpos[2]:.2f}]")
                
                with shared_data_lock:
                    shared_new_trajectory = trajectory
                    
                del obs_dict
                del action_pred
                torch.cuda.empty_cache()
                
        # 释放 CPU 资源
        time.sleep(0.01) 

# ==========================================
# 3. 小脑中枢：主程序与物理仿真循环 (50Hz)
# ==========================================
def main():
    global shared_latest_obs, shared_new_trajectory, is_running
    
    ckpt_path = "checkpoints/latest.ckpt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    payload = torch.load(open(ckpt_path, 'rb'), map_location=device)
    cfg = payload['cfg']
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    policy.eval().to(device)

    env = UR5ePushTEnv()
    current_obs, _ = env.reset()
    
    # 启动后台线程
    inference_thread = threading.Thread(
        target=inference_worker, 
        args=(policy, device, cfg.n_obs_steps),
        daemon=True
    )
    inference_thread.start()

    ensemble_alpha = 0.6  
    current_smoothed_target = current_obs['qpos'][:6].copy()

    print("🚀 [主线程] 启动 MuJoCo 物理引擎 (维持 50Hz 刷新率)...")
    
    control_hz = 50
    step_duration = 1.0 / control_hz
    env_step_count = 0
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # ---------------------------
            # A. 视神经同步 (每 5 步 = 10Hz 给大脑发一张新照片)
            # ---------------------------
            if env_step_count % 5 == 0:
                with shared_data_lock:
                    shared_latest_obs = current_obs
            
            # ---------------------------
            # B. 接收大脑信号
            # ---------------------------
            new_traj = None
            with shared_data_lock:
                if shared_new_trajectory is not None:
                    new_traj = shared_new_trajectory
                    shared_new_trajectory = None 
                    
            if new_traj is not None:
                raw_target = new_traj[0]
                current_smoothed_target = (1 - ensemble_alpha) * current_smoothed_target + ensemble_alpha * raw_target
                # 【终端监控 2】：小脑接收到信号并执行平滑
                print(f"   ⚙️ [小脑] 接收到新指令，正在平滑驱动电机...")
                
            # ---------------------------
            # C. 高频物理推进 (使用 env.step 确保相机渲染)
            # ---------------------------
            current_qpos = env.data.qpos[:6].copy()
            
            # 动作插值：当前位置 80% + 目标位置 20%
            interpolated_action = 0.8 * current_qpos + 0.2 * current_smoothed_target
            
            # 🌟 核心修复：直接调用 env.step，它会帮你把图渲染出来！
            current_obs, _, _, _, _ = env.step(interpolated_action)
            viewer.sync()
            
            env_step_count += 1
            
            # 精确控制主循环在 50Hz
            time_until_next = step_duration - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)
                
    is_running = False
    inference_thread.join()

if __name__ == "__main__":
    main()