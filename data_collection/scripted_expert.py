"""
实现目标：多段式闭环推击专家 (Multi-Stroke FSM)。通过不断重新评估位置，克服物块旋转滑移，实现精准的 XY 坐标追踪。
"""

import numpy as np

class PushTExpert:
    def __init__(self, target_pos=np.array([0.7, 0.2])):
        self.target_pos = target_pos
        self.state = 'LIFT'
        
        self.hover_z = 0.55  
        self.push_z = 0.435  
        
        self.start_xy = None
        self.end_xy = None
        self.push_count = 0  # 记录当前是第几次推击

    def reset(self):
        self.state = 'LIFT'
        self.start_xy = None
        self.end_xy = None
        self.push_count = 0

    def get_action(self, obs, current_ee_pos):
        t_cube_xy = obs['t_cube_pos'][:2]
        
        # 计算当前物块与目标的距离
        dist_to_target = np.linalg.norm(self.target_pos - t_cube_xy)

        # 每次进入 APPROACH 状态且未计算起点时，重新规划轨迹
        if self.state == 'APPROACH' and self.start_xy is None:
            push_vec = self.target_pos - t_cube_xy
            dist = np.linalg.norm(push_vec)
            if dist > 0.001:
                push_dir = push_vec / dist
            else:
                push_dir = np.array([1.0, 0.0])
                
            # 起点：物块后方 6cm
            self.start_xy = t_cube_xy - push_dir * 0.06
            
            # 终点：每次最多只往前推一段距离 (Stroke Length，最大 5cm)，防止失控滑走
            stroke_len = min(dist + 0.02, 0.05)
            self.end_xy = t_cube_xy + push_dir * stroke_len
            
            self.push_count += 1
            print(f"[专家规划] 第{self.push_count}次修正推击! 起点: {np.round(self.start_xy, 3)}, 距离目标: {dist:.3f}m")

        speed_fast = 0.005 
        speed_push = 0.003
        
        # ============ 闭环多段状态机 ============
        if self.state == 'LIFT':
            target = np.array([current_ee_pos[0], current_ee_pos[1], self.hover_z])
            if np.abs(target[2] - current_ee_pos[2]) < 0.005:
                # 抬起后进行裁判：如果距离目标小于 2cm，说明任务完成！
                if dist_to_target < 0.02:
                    self.state = 'DONE'
                else:
                    self.state = 'APPROACH' # 否则进入下一轮靠近
            return self._move_towards(current_ee_pos, target, speed_fast)

        elif self.state == 'APPROACH':
            target = np.array([self.start_xy[0], self.start_xy[1], self.hover_z])
            if np.linalg.norm(target - current_ee_pos) < 0.01:
                self.state = 'DESCEND'
            return self._move_towards(current_ee_pos, target, speed_fast)

        elif self.state == 'DESCEND':
            target = np.array([self.start_xy[0], self.start_xy[1], self.push_z])
            if np.linalg.norm(target - current_ee_pos) < 0.01:
                self.state = 'PUSH'
            return self._move_towards(current_ee_pos, target, speed_push)

        elif self.state == 'PUSH':
            target = np.array([self.end_xy[0], self.end_xy[1], self.push_z])
            if np.linalg.norm(target[:2] - current_ee_pos[:2]) < 0.01:
                # 【核心修改】：单次推击完成后，不清空环境，而是回到 LIFT 状态，并解锁轨迹
                self.start_xy = None 
                self.state = 'LIFT'
            return self._move_towards(current_ee_pos, target, speed_push)

        elif self.state == 'DONE':
            # 保持在安全高度，等待 main.py 发出重置指令
            target = np.array([current_ee_pos[0], current_ee_pos[1], self.hover_z])
            return self._move_towards(current_ee_pos, target, speed_fast)

    def _move_towards(self, current, target, step_size):
        diff = target - current
        dist = np.linalg.norm(diff)
        if dist <= step_size:
            return target
        return current + (diff / dist) * step_size