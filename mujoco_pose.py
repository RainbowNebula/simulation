import mujoco
import numpy as np
import pinocchio
from numpy.linalg import norm, solve
import os
import threading
import mujoco.viewer

# 定义机械臂关节角度范围（Franka Panda的典型关节范围，单位：弧度）
JOINT_LIMITS = [
    (-2.8973, 2.8973),    # 关节1
    (-1.7628, 1.7628),    # 关节2
    (-2.8973, 2.8973),    # 关节3
    (-3.0718, -0.0698),   # 关节4（注意范围是负数）
    (-2.8973, 2.8973),    # 关节5
    (-0.0175, 3.7525),    # 关节6
    (-2.8973, 2.8973)     # 关节7
]

# 定义工作空间范围（末端执行器可达区域，单位：米）
WORKSPACE_LIMITS = {
    'x': (0.2, 0.8),    # x轴范围（前后）
    'y': (-0.5, 0.5),   # y轴范围（左右）
    'z': (0.1, 1.0)     # z轴范围（上下，避免低于地面）
}

def inverse_kinematics(current_q, target_dir, target_pos, model, data):
    # 先检查目标位置是否在工作空间内
    if not is_position_in_workspace(target_pos):
        print(f"警告：目标位置 {target_pos} 超出工作空间，已调整到最近边界")
        target_pos = clamp_position_to_workspace(target_pos)
    
    # 指定要控制的关节 ID
    JOINT_ID = 7
    oMdes = pinocchio.SE3(target_dir, np.array(target_pos))

    q = current_q.copy()
    eps = 1e-4
    IT_MAX = 1000
    DT = 1e-2
    damp = 1e-12

    i = 0
    while True:
        pinocchio.forwardKinematics(model, data, q)
        iMd = data.oMi[JOINT_ID].actInv(oMdes)
        err = pinocchio.log(iMd).vector

        if norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break

        J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)
        J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
        v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pinocchio.integrate(model, q, v * DT)

        # 每次迭代后限制关节角度在安全范围内
        q = clamp_joint_angles(q)
        
        i += 1

    return q.flatten(), success, norm(err)

def clamp_joint_angles(q):
    """将关节角度限制在安全范围内"""
    clamped_q = []
    for i in range(7):
        min_angle, max_angle = JOINT_LIMITS[i]
        clamped_q.append(np.clip(q[i], min_angle, max_angle))
    return np.array(clamped_q)

def is_position_in_workspace(pos):
    """检查位置是否在工作空间内"""
    x, y, z = pos
    return (WORKSPACE_LIMITS['x'][0] <= x <= WORKSPACE_LIMITS['x'][1] and
            WORKSPACE_LIMITS['y'][0] <= y <= WORKSPACE_LIMITS['y'][1] and
            WORKSPACE_LIMITS['z'][0] <= z <= WORKSPACE_LIMITS['z'][1])

def clamp_position_to_workspace(pos):
    """将位置限制在工作空间内"""
    x, y, z = pos
    x_clamped = np.clip(x, WORKSPACE_LIMITS['x'][0], WORKSPACE_LIMITS['x'][1])
    y_clamped = np.clip(y, WORKSPACE_LIMITS['y'][0], WORKSPACE_LIMITS['y'][1])
    z_clamped = np.clip(z, WORKSPACE_LIMITS['z'][0], WORKSPACE_LIMITS['z'][1])
    return np.array([x_clamped, y_clamped, z_clamped])

def limit_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

# 加载机器人模型
model = mujoco.MjModel.from_xml_path('/home/haichao/workspace/real2sim/third_party/mujoco-learning/model/franka_emika_panda/scene.xml')
data = mujoco.MjData(model)

# 加载Pinocchio模型
def load_pinocchio_model(urdf_path):
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    model = pinocchio.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data

urdf_paths = [
    '/home/haichao/workspace/real2sim/third_party/mujoco-learning/model/franka_panda_description/robots/panda_arm.urdf',
    '/home/haichao/workspace/real2sim/third_party/mujoco-learning/model/franka_emika_panda/panda_arm.urdf'
]

pinocchio_model, pinocchio_data = None, None
for path in urdf_paths:
    try:
        pinocchio_model, pinocchio_data = load_pinocchio_model(path)
        print(f"成功加载URDF模型: {path}")
        break
    except:
        continue

if pinocchio_model is None:
    raise Exception("无法加载URDF模型，请检查路径是否正确")

class CustomViewer:
    def __init__(self, model, data, pinocchio_model, pinocchio_data):
        self.handle = mujoco.viewer.launch_passive(model, data)
        self.model = model
        self.data = data
        self.pinocchio_model = pinocchio_model
        self.pinocchio_data = pinocchio_data
        
        # 确认末端执行器ID（关键！错误会导致运动异常）
        self.end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'panda_hand')
        print(f"末端执行器ID: {self.end_effector_id}")
        if self.end_effector_id == -1:
            print("警告: 找不到'panda_hand'，尝试'hand'")
            self.end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
            if self.end_effector_id == -1:
                print("警告: 找不到'hand'，尝试'left_finger'")
                self.end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_finger')
        print(f"最终使用的末端执行器ID: {self.end_effector_id}")

        # 初始关节角度（限制在安全范围）
        self.initial_q = clamp_joint_angles(data.qpos[:7].copy())
        print(f"初始关节位置（已限位）: {self.initial_q}")
        
        # 加载抓取姿态
        self.grasp_poses = self.load_grasp_poses()
        self.current_pose_idx = 0
        print(f"成功加载 {len(self.grasp_poses)} 个抓取姿态")
        
        # 动画参数（减慢速度，避免剧烈运动）
        self.animation_speed = 0.005  # 减慢过渡速度
        self.transition_progress = 0.0
        self.current_q = self.initial_q.copy()
        self.target_q = self.initial_q.copy()
        
        # 控制标志
        self.next_pose = False
        self.running = True
        
        # 启动输入监听线程
        self.input_thread = threading.Thread(target=self.listen_for_input)
        self.input_thread.daemon = True
        self.input_thread.start()

    def load_grasp_poses(self, file_path='/mnt/nas/liuqipeng/workspace/GraspGen/valid_pose.npy'):
        try:
            if not os.path.exists(file_path):
                print(f"未找到抓取姿态文件 {file_path}，生成示例数据")
                num_poses = 5
                poses = []
                for i in range(num_poses):
                    pose = np.eye(4)
                    # 生成工作空间内的安全位置（避免过低）
                    pose[0, 3] = 0.4 + i * 0.05  # x在0.4-0.6
                    pose[1, 3] = 0.0 + np.sin(i * 0.5) * 0.1  # y在-0.1-0.1
                    pose[2, 3] = 0.7 + np.cos(i * 0.5) * 0.1  # z在0.6-0.8（远离地面）
                    theta = i * 0.3
                    pose[0, 0] = np.cos(theta)
                    pose[0, 1] = -np.sin(theta)
                    pose[1, 0] = np.sin(theta)
                    pose[1, 1] = np.cos(theta)
                    poses.append(pose)
                return np.array(poses)
            
            poses = np.load(file_path)
            if len(poses.shape) != 3 or poses.shape[1:] != (4, 4):
                raise ValueError(f"抓取姿态数据格式应为 (N,4,4)，实际为 {poses.shape}")
            
            # 过滤并修正姿态中的位置，确保在工作空间内
            valid_poses = []
            for pose in poses:
                pos = pose[:3, 3]
                if not is_position_in_workspace(pos):
                    print(f"修正姿态位置 {pos} 到工作空间内")
                    pose[:3, 3] = clamp_position_to_workspace(pos)
                valid_poses.append(pose)
            return np.array(valid_poses)
        
        except Exception as e:
            print(f"加载抓取姿态出错: {e}")
            return np.array([np.eye(4)])

    def get_current_grasp_pose(self):
        if len(self.grasp_poses) == 0:
            return np.eye(4)
        return self.grasp_poses[self.current_pose_idx % len(self.grasp_poses)]

    def listen_for_input(self):
        print("请输入 'n' 切换到下一个姿态，输入 'q' 退出程序")
        while self.running:
            user_input = input().strip().lower()
            if user_input == 'n':
                self.next_pose = True
            elif user_input == 'q':
                self.running = False
                self.handle.close()
            else:
                print("无效输入，请输入 'n' 切换到下一个姿态，输入 'q' 退出程序")

    def compute_target_joints(self):
        pose = self.get_current_grasp_pose()
        target_pos = pose[:3, 3]
        target_dir = pose[:3, :3]
        
        print(f"计算姿态 {self.current_pose_idx % len(self.grasp_poses) + 1} 的逆解:")
        print(f"目标位置: {target_pos}")
        
        target_q, success, error = inverse_kinematics(
            self.current_q, 
            target_dir, 
            target_pos,
            self.pinocchio_model,
            self.pinocchio_data
        )
        
        # 再次确认关节角度在安全范围
        target_q = clamp_joint_angles(target_q)
        
        if success:
            print(f"逆解计算成功，误差: {error:.6f}")
            self.target_q = target_q
        else:
            print(f"警告: 逆解计算未收敛，误差: {error:.6f}")
            self.target_q = target_q

    def transition_joints(self):
        """更平滑的过渡算法，避免剧烈运动"""
        if self.transition_progress < 1.0:
            self.transition_progress += self.animation_speed
            self.transition_progress = min(1.0, self.transition_progress)
            # 使用缓动函数（ease-in-out）使运动更平滑
            t = self.transition_progress
            ease_t = t * t * (3 - 2 * t)  # 缓动公式
            self.current_q = self.current_q + (self.target_q - self.current_q) * ease_t
        return self.transition_progress >= 1.0

    def is_running(self):
        return self.running and self.handle.is_running()

    def sync(self):
        self.handle.sync()

    @property
    def cam(self):
        return self.handle.cam

    @property
    def viewport(self):
        return self.handle.viewport
    
    def run_loop(self):
        self.compute_target_joints()
        
        while self.is_running():
            mujoco.mj_forward(self.model, self.data)
            
            if self.next_pose:
                self.current_pose_idx += 1
                self.transition_progress = 0.0
                self.compute_target_joints()
                print(f"切换到姿态 {self.current_pose_idx % len(self.grasp_poses) + 1}/{len(self.grasp_poses)}")
                self.next_pose = False
            
            self.transition_joints()
            
            # 最终确认关节角度（双重保险）
            self.data.qpos[:7] = clamp_joint_angles(self.current_q)
            
            mujoco.mj_step(self.model, self.data)
            self.sync()

if __name__ == "__main__":
    viewer = CustomViewer(model, data, pinocchio_model, pinocchio_data)
    
    # 调整相机位置，更好地观察机械臂运动
    viewer.cam.distance = 2.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20
    viewer.cam.lookat = [0.5, 0, 0.5]  # 聚焦工作空间中心
    
    viewer.run_loop()
