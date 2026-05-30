"""
实现目标：读取并可视化打包好的 Zarr 视觉数据集，验证双摄像头视角与数据连续性。
输入：dataset/pusht_ur5e_vision.zarr
输出：弹出 OpenCV 视频窗口，以 10Hz 的频率同步并排播放 Overhead 和 Wrist 相机的画面。
"""

import zarr
import cv2
import numpy as np

def visualize_zarr(zarr_path="dataset/pusht_ur5e_vision.zarr"):
    print(f"正在加载数据集: {zarr_path}")
    try:
        root = zarr.open(zarr_path, mode='r')
        img_overhead = root['data/img_overhead'][:]
        img_wrist = root['data/img_wrist'][:]
        episode_ends = root['meta/episode_ends'][:]
    except Exception as e:
        print(f"❌ 加载失败，请检查路径: {e}")
        return
        
    total_frames = img_overhead.shape[0]
    print(f"✅ 成功加载！共包含 {len(episode_ends)} 个 Episode, {total_frames} 帧图像。")
    print("-----------------------------------------")
    print("🎮 播放控制说明：")
    print(" - 按 '空格键 (Space)' : 暂停 / 继续")
    print(" - 按 'Q' 键          : 退出播放")
    print("-----------------------------------------")
    
    current_ep = 1
    ep_end_idx = episode_ends[current_ep - 1]
    
    for i in range(total_frames):
        # 1. 获取当前帧图像
        # 提示：MuJoCo 渲染出来的是 RGB 格式，而 OpenCV 默认显示需要 BGR 格式，所以需要转换
        overhead_frame = cv2.cvtColor(img_overhead[i], cv2.COLOR_RGB2BGR)
        wrist_frame = cv2.cvtColor(img_wrist[i], cv2.COLOR_RGB2BGR)
        
        # 2. 将两张图片水平拼接在一起 (256x256 -> 512x256)
        combined_frame = np.hstack((overhead_frame, wrist_frame))
        
        # 3. 添加文本标注 (绿字)
        cv2.putText(combined_frame, "Overhead Cam", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined_frame, "Wrist Cam", (256 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加帧率和进度信息 (白字)
        progress_text = f"Ep: {current_ep}/{len(episode_ends)} | Frame: {i+1}/{total_frames}"
        cv2.putText(combined_frame, progress_text, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 4. 播放画面 (为了看得更清楚，我们把画面放大 1.5 倍)
        display_frame = cv2.resize(combined_frame, (768, 384))
        cv2.imshow("Dataset Visualization", display_frame)
        
        # 5. 控制帧率 (我们的数据是 10Hz 录制的，所以每帧停留 100 毫秒)
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  
            # 如果按下空格键，就进入死循环等待，直到再按一次空格
            print("⏸️ 已暂停，再按空格键继续...")
            while True:
                if cv2.waitKey(10) & 0xFF == ord(' '):
                    print("▶️ 继续播放")
                    break
            
        # 6. Episode 切换逻辑
        if i >= ep_end_idx - 1 and current_ep < len(episode_ends):
            current_ep += 1
            ep_end_idx = episode_ends[current_ep - 1]
            # 每个 Episode 结束时，画面停顿 1 秒，方便你观察最终状态
            cv2.waitKey(1000) 

    cv2.destroyAllWindows()
    print("🎬 播放结束。")

if __name__ == "__main__":
    visualize_zarr()