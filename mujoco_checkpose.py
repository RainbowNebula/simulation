import mujoco
import numpy as np
import glfw
import mujoco.viewer
import pinocchio
from numpy.linalg import norm, solve
import os
import threading

def inverse_kinematics(current_q, target_dir, target_pos, model, data):
    # 指定要控制的关节 ID
    JOINT_ID = 7
    # 定义期望的位姿，使用目标姿态的旋转矩阵和目标位置创建 SE3 对象
    oMdes = pinocchio.SE3(target_dir, np.array(target_pos))

    # 将当前关节角度赋值给变量 q，作为迭代的初始值
    q = current_q.copy()
    # 定义收敛阈值，当误差小于该值时认为算法收敛
    eps = 1e-4
    # 定义最大迭代次数，防止算法陷入无限循环
    IT_MAX = 1000
    # 定义积分步长，用于更新关节角度
    DT = 1e-2
    # 定义阻尼因子，用于避免矩阵奇异
    damp = 1e-12

    # 初始化迭代次数为 0
    i = 0
    while True:
        # 进行正运动学计算，得到当前关节角度下机器人各关节的位置和姿态
        pinocchio.forwardKinematics(model, data, q)
        # 计算目标位姿到当前位姿之间的变换
        iMd = data.oMi[JOINT_ID].actInv(oMdes)
        # 通过李群对数映射将变换矩阵转换为 6 维误差向量
        err = pinocchio.log(iMd).vector

        # 判断误差是否小于收敛阈值，如果是则认为算法收敛
        if norm(err) < eps:
            success = True
            break
        # 判断迭代次数是否超过最大迭代次数，如果是则认为算法未收敛
        if i >= IT_MAX:
            success = False
            break

        # 计算当前关节角度下的雅可比矩阵
        J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)
        # 对雅可比矩阵进行变换
        J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
        # 使用阻尼最小二乘法求解关节速度
        v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        # 根据关节速度更新关节角度
        q = pinocchio.integrate(model, q, v * DT)

        i += 1

    return q.flatten(), success, norm(err)

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

# 尝试不同的URDF路径（根据实际情况调整）
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
        
        # 找到末端执行器的 body id
        self.end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
        print(f"末端执行器ID: {self.end_effector_id}")
        if self.end_effector_id == -1:
            print("警告: 找不到指定名称的末端执行器，尝试使用'left_finger'")
            self.end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_finger')
            print(f"尝试使用末端执行器ID: {self.end_effector_id}")

        # 初始关节角度
        self.initial_q = data.qpos[:7].copy()
        print(f"初始关节位置: {self.initial_q}")
        
        # 加载抓取姿态
        self.grasp_poses = self.load_grasp_poses()
        self.current_pose_idx = 0
        print(f"成功加载 {len(self.grasp_poses)} 个抓取姿态")
        
        # 用于控制动画
        self.animation_speed = 0.01
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
        """加载npy文件中的抓取姿态"""
        try:
            if not os.path.exists(file_path):
                print(f"未找到抓取姿态文件 {file_path}，生成示例数据")
                # 生成示例数据
                num_poses = 5
                poses = []
                for i in range(num_poses):
                    # 生成一个简单的变换矩阵
                    pose = np.eye(4)
                    # 平移部分 - 生成一些在工作空间内的点
                    pose[0, 3] = 0.3 + i * 0.05  # x
                    pose[1, 3] = 0.2 + np.sin(i * 0.5) * 0.1  # y
                    pose[2, 3] = 0.6 + np.cos(i * 0.5) * 0.1  # z
                    # 旋转部分 - 简单的绕z轴旋转
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
            return poses
        except Exception as e:
            print(f"加载抓取姿态出错: {e}")
            # 返回默认姿态
            return np.array([np.eye(4)])

    def get_current_grasp_pose(self):
        """获取当前抓取姿态"""
        if len(self.grasp_poses) == 0:
            return np.eye(4)
        return self.grasp_poses[self.current_pose_idx % len(self.grasp_poses)]

    def listen_for_input(self):
        """监听命令行输入，输入'n'切换到下一个姿态"""
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
        """计算目标姿态对应的关节角度"""
        pose = self.get_current_grasp_pose()
        target_pos = pose[:3, 3]  # 平移向量
        target_dir = pose[:3, :3]  # 旋转矩阵
        
        print(f"计算姿态 {self.current_pose_idx % len(self.grasp_poses) + 1} 的逆解:")
        print(f"目标位置: {target_pos}")
        
        # 计算逆运动学
        target_q, success, error = inverse_kinematics(
            self.current_q, 
            target_dir, 
            target_pos,
            self.pinocchio_model,
            self.pinocchio_data
        )
        
        if success:
            print(f"逆解计算成功，误差: {error:.6f}")
            self.target_q = target_q
        else:
            print(f"警告: 逆解计算未收敛，误差: {error:.6f}")
            # 仍然使用计算结果，尽管未完全收敛
            self.target_q = target_q

    def transition_joints(self):
        """平滑过渡到目标关节角度"""
        if self.transition_progress < 1.0:
            self.transition_progress += self.animation_speed
            self.transition_progress = min(1.0, self.transition_progress)
            # 线性插值平滑过渡
            self.current_q = self.current_q + (self.target_q - self.current_q) * self.animation_speed / (1.0 - self.transition_progress + 1e-6)
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
        # 初始计算一次目标关节
        self.compute_target_joints()
        
        while self.is_running():
            mujoco.mj_forward(self.model, self.data)
            
            # 检查是否需要切换到下一个姿态
            if self.next_pose:
                self.current_pose_idx += 1
                self.transition_progress = 0.0
                self.compute_target_joints()
                print(f"切换到姿态 {self.current_pose_idx % len(self.grasp_poses) + 1}/{len(self.grasp_poses)}")
                self.next_pose = False
            
            # 平滑过渡到目标关节角度
            self.transition_joints()
            
            # 更新关节位置
            self.data.qpos[:7] = self.current_q
            
            mujoco.mj_step(self.model, self.data)
            self.sync()

if __name__ == "__main__":
    # 创建Viewer并运行
    viewer = CustomViewer(model, data, pinocchio_model, pinocchio_data)
    
    # 设置相机位置
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 90  # 方位角
    viewer.cam.elevation = -30  # 仰角
    viewer.cam.lookat = [0.5, 0, 0.5]  # 相机注视点
    
    viewer.run_loop()