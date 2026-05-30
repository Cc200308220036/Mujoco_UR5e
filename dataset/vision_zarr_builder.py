"""
实现目标：将包含双摄像头 RGB 图像的零散 .npz 专家数据，打包为 Vision-based Diffusion Policy 官方原生支持的 .zarr 格式。
输入：dataset/raw/ 目录下的多个 episode_xxxx.npz 文件。
输出：
    pusht_ur5e_vision.zarr/
        ├── data/
        │   ├── img_overhead (N, H, W, C) - uint8
        │   ├── img_wrist    (N, H, W, C) - uint8
        │   ├── state        (N, D_state) - float32 (包含机械臂本体状态)
        │   └── action       (N, D_action)- float32
        └── meta/
            └── episode_ends (num_episodes,) - int64
"""

import os
import glob
import numpy as np
import zarr

def create_vision_zarr_dataset(raw_dir="dataset/raw", zarr_path="dataset/pusht_ur5e_vision.zarr"):
    npz_files = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))
    if not npz_files:
        print("❌ 未找到任何 .npz 文件！请先用键盘遥控采集几组数据。")
        return

    # 准备容器
    all_overhead = []
    all_wrist = []
    all_qpos = []
    all_t_pos = []
    all_actions = []
    episode_ends = []
    
    current_end = 0
    
    print(f"正在读取 {len(npz_files)} 个视觉原始文件...")
    for f in npz_files:
        data = np.load(f)
        all_overhead.append(data['image_overhead'])
        all_wrist.append(data['image_wrist'])
        all_qpos.append(data['qpos'])
        all_t_pos.append(data['t_cube_pos'])
        all_actions.append(data['action'])
        
        current_end += len(data['action'])
        episode_ends.append(current_end)
        
    # 拼接所有数据
    print("正在拼接图像与状态张量 (如果数据量大可能需要一点时间)...")
    overhead_arr = np.concatenate(all_overhead, axis=0) # (Total, 256, 256, 3)
    wrist_arr = np.concatenate(all_wrist, axis=0)       # (Total, 256, 256, 3)
    qpos_arr = np.concatenate(all_qpos, axis=0)         # (Total, 6)
    t_pos_arr = np.concatenate(all_t_pos, axis=0)       # (Total, 3)
    action_arr = np.concatenate(all_actions, axis=0)    # (Total, 3)
    
    # 状态通常只包含机械臂本体的 proprioception (如关节角)
    # T型块的位置不应放入 state，因为视觉模型应该自己从图像中“看”出 T型块的位置！
    # 但为了方便，我们把末端当前位姿或者 qpos 作为 state
    state_arr = qpos_arr 
    episode_ends_arr = np.array(episode_ends, dtype=np.int64)
    
    # 写入 Zarr 文件
    print(f"正在打包存入 Zarr: {zarr_path} ...")
    root = zarr.open(zarr_path, mode='w')
    
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')
    
    # 注意：图像数据非常大，必须开启 chunks (分块) 否则训练读取时内存会爆炸
    # chunks=(100, 256, 256, 3) 意味着每次从硬盘读 100 张图片
    data_group.create_dataset('img_overhead', data=overhead_arr, chunks=(100, overhead_arr.shape[1], overhead_arr.shape[2], 3), dtype='uint8')
    data_group.create_dataset('img_wrist', data=wrist_arr, chunks=(100, wrist_arr.shape[1], wrist_arr.shape[2], 3), dtype='uint8')
    data_group.create_dataset('state', data=state_arr, chunks=(100, state_arr.shape[1]), dtype='float32')
    data_group.create_dataset('action', data=action_arr, chunks=(100, action_arr.shape[1]), dtype='float32')
    
    meta_group.create_dataset('episode_ends', data=episode_ends_arr, dtype='int64')
    
    print("="*50)
    print("✅ 视觉 Zarr 数据集构建完成！")
    print(f"总 Episode 数: {len(npz_files)}")
    print(f"总数据帧数: {current_end}")
    print(f"Overhead 图像维度: {overhead_arr.shape}")
    print(f"Wrist 图像维度: {wrist_arr.shape}")
    print(f"State 维度: {state_arr.shape}")
    print(f"Action 维度: {action_arr.shape}")
    print("="*50)

if __name__ == "__main__":
    create_vision_zarr_dataset()