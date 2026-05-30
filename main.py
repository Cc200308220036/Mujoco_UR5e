"""
实现目标：视觉驱动的全自动专家数据采集器。采集双相机图像与物理状态，并按回合实时压缩落盘，防止内存溢出。
输入：无（自动运行状态机）。
输出：按回合生成压缩的 npz 文件，保存在 dataset/raw/ 目录下。
"""

import os
import time
import numpy as np
import mujoco.viewer

from envs.ur5e_pusht_env import UR5ePushTEnv
from control.osc_controller import IKController
from data_collection.scripted_expert import PushTExpert

def save_episode_to_disk(episode_data, ep_idx):
    """将单回合包含图像的混合数据压缩保存至硬盘"""
    save_dir = os.path.join("dataset", "raw")
    os.makedirs(save_dir, exist_ok=True)

    obs_qpos, obs_t_cube = [], []
    img_overhead, img_wrist = [], []
    actions = []

    for frame in episode_data:
        obs_qpos.append(frame['obs']['qpos'])
        obs_t_cube.append(frame['obs']['t_cube_pos'])
        img_overhead.append(frame['obs']['image_overhead'])
        img_wrist.append(frame['obs']['image_wrist'])
        actions.append(frame['action'])

    # 使用 compressed 节约磁盘读写速度和空间
    file_path = os.path.join(save_dir, f"episode_{ep_idx:04d}.npz")
    np.savez_compressed(
        file_path,
        qpos=np.array(obs_qpos, dtype=np.float32),
        t_cube_pos=np.array(obs_t_cube, dtype=np.float32),
        image_overhead=np.array(img_overhead, dtype=np.uint8),
        image_wrist=np.array(img_wrist, dtype=np.uint8),
        action=np.array(actions, dtype=np.float32)
    )
    print(f"💾 视觉数据已压缩落盘: {file_path}")

def main():
    env = UR5ePushTEnv()
    obs, info = env.reset()
    
    ik_controller = IKController(env.model, env.data)
    ik_controller.sync_target_orientation()
    expert = PushTExpert()
    cmd_qpos = env.data.qpos[:6].copy()
    
    total_episodes_to_collect = 2   
    record_hz = 10                   
    env_hz = 50                      
    steps_per_record = int(env_hz / record_hz) 
    
    current_episode_data = []
    step_count = 0
    episode_count = 0
    
    print("="*60)
    print("🚀 [视觉数据采集器] 启动！")
    print("注意：正在高频渲染双摄图像，后台可能会稍有卡顿，属正常现象。")
    print("="*60)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running() and episode_count < total_episodes_to_collect:
            step_start = time.time()
            step_count += 1
            
            current_ee_pos = env.data.site_xpos[ik_controller.site_id].copy()
            target_pos = expert.get_action(obs, current_ee_pos)
            
            current_qpos = env.data.qpos[:6].copy()
            action = ik_controller.calculate_joint_targets(target_pos, cmd_qpos)
            cmd_qpos = np.clip(action, current_qpos - 0.05, current_qpos + 0.05)
            
            obs, reward, terminated, truncated, info = env.step(cmd_qpos)
            viewer.sync()
            
            if step_count % steps_per_record == 0:
                current_episode_data.append({
                    'obs': obs.copy(),
                    'action': target_pos.copy() 
                })
            
            if expert.state == 'DONE':
                episode_count += 1
                print(f"✅ Episode {episode_count} 成功完成! (共 {len(current_episode_data)} 帧)")
                
                # 【核心】：立刻落盘并清空内存
                save_episode_to_disk(current_episode_data, episode_count)
                current_episode_data = [] 
                
                if episode_count >= total_episodes_to_collect:
                    print("\n🎉 视觉数据收集达成！")
                    break
                
                obs, info = env.reset()                
                expert.reset()                         
                ik_controller.sync_target_orientation()
                cmd_qpos = env.data.qpos[:6].copy()    
                step_count = 0                         
                time.sleep(0.5) 
            
            time_until_next_step = 0.02 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()