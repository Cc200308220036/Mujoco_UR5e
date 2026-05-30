"""
实现目标：将录制的零散 .npz 专家数据，打包为 Diffusion Policy 官方原生支持的 .zarr 格式。
输出结构：
    pusht_ur5e.zarr/
        ├── data/
        │   ├── state  (N, D_state)
        │   └── action (N, D_action)
        └── meta/
            └── episode_ends (num_episodes,)
"""

import os
import glob
import numpy as np
import zarr

def create_zarr_dataset(raw_dir="dataset/raw", zarr_path="dataset/pusht_ur5e.zarr"):
    # 1. 查找所有 .npz 文件并排序
    npz_files = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))
    if not npz_files:
        print("❌ 未找到任何 .npz 文件！请先运行采集脚本。")
        return

    # 准备容器
    all_qpos = []
    all_t_pos = []
    all_actions = []
    episode_ends = []
    
    current_end = 0
    
    # 2. 遍历加载数据
    print(f"正在处理 {len(npz_files)} 个原始文件...")
    for f in npz_files:
        data = np.load(f)
        all_qpos.append(data['qpos'])
        all_t_pos.append(data['t_cube_pos'])
        all_actions.append(data['action'])
        
        # 记录每个 episode 在全局数组中的结束索引 (Diffusion Policy 的特殊要求)
        current_end += len(data['action'])
        episode_ends.append(current_end)
        
    # 3. 拼接所有数据
    qpos_arr = np.concatenate(all_qpos, axis=0)       # (Total_frames, 6)
    t_pos_arr = np.concatenate(all_t_pos, axis=0)     # (Total_frames, 3)
    action_arr = np.concatenate(all_actions, axis=0)  # (Total_frames, 3)
    
    # 对于纯低维状态的策略模型，通常将机械臂状态和物体状态拼接作为输入 State
    state_arr = np.concatenate([qpos_arr, t_pos_arr], axis=1) # (Total_frames, 9)
    episode_ends_arr = np.array(episode_ends)
    
    # 4. 写入 Zarr 文件
    print(f"正在打包存入 Zarr: {zarr_path} ...")
    # 如果已存在则覆盖 ('w' 模式)
    root = zarr.open(zarr_path, mode='w')
    
    # 创建 data 和 meta 组
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')
    
    # 写入具体数据集，并开启分块 (chunks) 以加速后续 PyTorch 读取
    data_group.create_dataset('state', data=state_arr, chunks=(100, state_arr.shape[1]))
    data_group.create_dataset('action', data=action_arr, chunks=(100, action_arr.shape[1]))
    meta_group.create_dataset('episode_ends', data=episode_ends_arr)
    
    print("="*50)
    print("✅ Zarr 数据集构建完成！")
    print(f"总 Episode 数: {len(npz_files)}")
    print(f"总数据帧数: {current_end}")
    print(f"State 维度: {state_arr.shape}")
    print(f"Action 维度: {action_arr.shape}")
    print("="*50)

if __name__ == "__main__":
    create_zarr_dataset()