"""
实现目标：全交互式相机调参台。修复了按键冲突，并修正了 FOV 标量打印报错。
"""

import time
import numpy as np
import mujoco.viewer
from pynput import keyboard
from envs.ur5e_pusht_env import UR5ePushTEnv
import mujoco

# 全局变量控制平滑移动
delta_pos = np.zeros(3)
delta_fovy = 0.0
trigger_print = False

def on_press(key):
    global delta_pos, delta_fovy, trigger_print
    step = 0.002
    try:
        # 【修复2：替换为减号和等号，避开右中括号】
        if key.char == '-': delta_fovy = -0.5
        elif key.char == '=': delta_fovy = 0.5
    except AttributeError:
        if key == keyboard.Key.up: delta_pos[1] = step          
        elif key == keyboard.Key.down: delta_pos[1] = -step     
        elif key == keyboard.Key.left: delta_pos[0] = -step     
        elif key == keyboard.Key.right: delta_pos[0] = step     
        elif key == keyboard.Key.page_up: delta_pos[2] = step   
        elif key == keyboard.Key.page_down: delta_pos[2] = -step
        elif key == keyboard.Key.enter: trigger_print = True

def on_release(key):
    global delta_pos, delta_fovy
    try:
        if key.char in ['-', '=']: delta_fovy = 0.0
    except AttributeError:
        if key in [keyboard.Key.up, keyboard.Key.down]: delta_pos[1] = 0.0
        elif key in [keyboard.Key.left, keyboard.Key.right]: delta_pos[0] = 0.0
        elif key in [keyboard.Key.page_up, keyboard.Key.page_down]: delta_pos[2] = 0.0

def main():
    global trigger_print
    
    env = UR5ePushTEnv()
    env.reset()
    
    test_qpos = [3.14, -1.2, 1.5, -1.8, -1.57, 0.0]
    env.data.qpos[:6] = test_qpos
    env.data.ctrl[:6] = test_qpos
    
    idx = env.model.jnt_qposadr[env.t_cube_joint_id]
    env.data.qpos[idx : idx+7] = [0.65, 0.0, 0.45, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(env.model, env.data)

    cam_name = "wrist_cam"
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id == -1:
        print(f"❌ 找不到名为 {cam_name} 的相机")
        return

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("="*60)
    print("📸 [交互式相机调参台] 修复版已启动！")
    print("请点击弹出的 MuJoCo 窗口，按 `]` (仅按一次) 切换到 wrist_cam 视角。")
    print("================ 控制面板 ================")
    print(" [PageUp] / [PageDown] : 调节 Z 轴 (控制倾斜角大小)")
    print(" [↑] / [↓]             : 调节 Y 轴 (控制距离远近)")
    print(" [←] / [→]             : 调节 X 轴 (控制左右偏移)")
    print(" [-] (减号)            : 减小 FOV (画面放大)")
    print(" [=] (等号)            : 增大 FOV (画面变广)")
    print(" [Enter]               : 在终端打印最终 XML 代码")
    print("="*60)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            if np.any(delta_pos != 0):
                env.model.cam_pos[cam_id] += delta_pos
            if delta_fovy != 0:
                env.model.cam_fovy[cam_id] = np.clip(env.model.cam_fovy[cam_id] + delta_fovy, 10, 150)
            
            mujoco.mj_forward(env.model, env.data)
            viewer.sync()
            
            if trigger_print:
                pos = env.model.cam_pos[cam_id]
                fovy = env.model.cam_fovy[cam_id]
                print("\n✨ 调试完成！请将 ur5e.xml 中的 wrist_cam 替换为以下代码：")
                print("-" * 60)
                # 【修复1：移除了 fovy 的 [0] 索引，直接格式化标量】
                print(f'<camera name="wrist_cam" pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}" target="pusher_tip_body" fovy="{fovy:.1f}"/>')
                print("-" * 60)
                trigger_print = False
            
            time.sleep(0.02)

if __name__ == "__main__":
    main()