import mujoco
import mujoco.viewer
import time
import os

def test_environment():
    # 指向我们刚刚编写的主场景 XML
    xml_path = os.path.join("envs", "assets", "ur5e_scene.xml")
    
    # 加载模型和数据
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 启动交互式渲染窗口
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 物理仿真步进循环
        while viewer.is_running():
            step_start = time.time()
            
            # 步进物理引擎
            mujoco.mj_step(model, data)
            
            # 同步渲染状态
            viewer.sync()
            
            # 保持仿真时间与现实时间同步 (基于 xml 中设定的 timestep)
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    test_environment()