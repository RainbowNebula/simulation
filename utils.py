import numpy as np
import transforms3d as t3d  # 确保已导入transforms3d库
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Slerp

import random
import torch


def augment_grasps_with_interpolation(grasp_poses, num_interpolations=5):
    """
    处理4x4变换矩阵形式的抓取位姿，通过插值生成新位姿
    
    参数:
        grasp_poses: 原始抓取位姿列表，每个位姿是4x4变换矩阵
        num_interpolations: 插值步数
        
    返回:
        扩展后的抓取位姿列表（原始位姿+插值位姿）
    """
    augmented_grasps = list(grasp_poses)  # 保留原始位姿
    
    if len(grasp_poses) < 2:
        return augmented_grasps  # 不足两个位姿时直接返回原始列表
    
    # 提取所有位姿的位置信息（变换矩阵的平移部分）并找到差异最大的两个位姿
    positions = []
    for pose in grasp_poses:
        # 从4x4变换矩阵中提取位置 (最后一列的前三个元素)
        pos = pose[:3, 3]
        positions.append(pos)
    
    # 计算位置距离矩阵并找到最大距离的位姿对
    positions_np = np.array(positions)
    distances = cdist(positions_np, positions_np, metric='euclidean')
    max_idx = np.unravel_index(np.argmax(distances), distances.shape)
    pose1, pose2 = grasp_poses[max_idx[0]], grasp_poses[max_idx[1]]
    
    # 从变换矩阵中提取旋转矩阵和平移向量
    rot_mat1, pos1 = pose1[:3, :3], pose1[:3, 3]
    rot_mat2, pos2 = pose2[:3, :3], pose2[:3, 3]
    
    # 将旋转矩阵转换为四元数以便进行SLERP插值
    quat1 = R.from_matrix(rot_mat1).as_quat()
    quat2 = R.from_matrix(rot_mat2).as_quat()
    
    # 生成插值参数（注意：Slerp需要关键帧的参数值）
    key_times = [0, 1]  # 两个关键帧的时间点
    t_values = np.linspace(0, 1, num_interpolations)  # 插值点
    
    # 四元数球面插值 - 修复Slerp初始化错误
    # 创建旋转对象列表和对应的关键时间点
    rotations = R.from_quat([quat1, quat2])
    slerp = Slerp(key_times, rotations)  # 正确的Slerp初始化方式
    slerp_rots = slerp(t_values)  # 计算插值结果
    
    # 位置线性插值
    interpolated_positions = np.array([pos1 + (pos2 - pos1) * t for t in t_values])
    
    # 创建并添加插值得到的新抓取位姿（排除起点和终点避免重复）
    for i in range(1, num_interpolations - 1):  # 跳过第一个和最后一个，避免重复
        # 获取当前插值的位置和旋转
        pos = interpolated_positions[i]
        rot_matrix = slerp_rots[i].as_matrix()
        
        # 构造4x4变换矩阵
        new_pose = np.eye(4)  # 初始化4x4单位矩阵
        new_pose[:3, :3] = rot_matrix  # 填充旋转部分
        new_pose[:3, 3] = pos  # 填充平移部分
        
        augmented_grasps.append(new_pose)
    
    return augmented_grasps

def quat_euler_converter(quaternion, euler_angles_deg, return_type='quat'):
    """
    将四元数与角度制欧拉角组合，返回旋转矩阵或四元数
    
    参数:
        quaternion: 四元数，格式为[x, y, z, w]
        euler_angles_deg: 角度制XYZ欧拉角，格式为[rx_deg, ry_deg, rz_deg]
        return_type: 返回类型，'matrix'返回旋转矩阵，'quaternion'返回四元数[x, y, z, w]
    
    返回:
        3x3旋转矩阵或四元数列表
    """
    # 从四元数创建旋转对象
    rot_quat = R.from_quat(quaternion)
    
    # 将角度制欧拉角转换为弧度
    euler_angles_rad = np.radians(euler_angles_deg)
    
    # 从欧拉角创建旋转对象 (xyz顺序)
    rot_euler = R.from_euler('xyz', euler_angles_rad)
    
    # 组合两个旋转: 先应用四元数旋转，再应用欧拉角旋转
    total_rot = rot_quat * rot_euler 
    
    # 根据返回类型选择输出
    if return_type == 'matrix':
        return total_rot.as_matrix()
    elif return_type == 'quat':
        return total_rot.as_quat()[[3, 0, 1, 2]] # wxyz
    else:
        raise ValueError("return_type必须是'matrix'或'quaternion'")


def decompose_grasppose(grasppose):
    """
    使用库函数将4x4 grasppose变换矩阵分解为四元数、旋转角和平移向量
    
    参数:
        matrix: 4x4 numpy数组，表示变换矩阵
        
    返回:
        元组: (四元数(x,y,z,w), 旋转角(roll,pitch,yaw,弧度), 平移向量(x,y,z))
    """
    # 验证输入
    if grasppose.shape != (4, 4):
        raise ValueError("输入必须是4x4矩阵")
    
    # 提取平移向量
    position = grasppose[:3, 3]
    
    # 提取旋转矩阵并转换
    rot_mat = grasppose[:3, :3]
    rotation = R.from_matrix(rot_mat)
    
    # 四元数 (x,y,z,w)
    quaternion = rotation.as_quat(scalar_first=True) # scalar_first=False
    # 欧拉角 (roll, pitch, yaw)，使用ZYX顺序
    euler_angles = rotation.as_euler('xyz', degrees=True)  # 转换为roll, pitch, yaw顺序
    
    return position, quaternion, euler_angles

def rand_pose(
    xlim: np.ndarray,
    ylim: np.ndarray,
    zlim: np.ndarray,
    ylim_prop=False,
    rotate_rand=False,
    rotate_lim=[0, 0, 0],  # 改为角度制（范围如 [-30, 30] 表示±30度）
    quat=[1, 0, 0, 0],     # 初始四元数（默认无旋转）
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 处理输入范围，确保格式正确
    if len(xlim) < 2 or xlim[1] < xlim[0]:
        xlim = np.array([xlim[0], xlim[0]])
    if len(ylim) < 2 or ylim[1] < ylim[0]:
        ylim = np.array([ylim[0], ylim[0]])
    if len(zlim) < 2 or zlim[1] < zlim[0]:
        zlim = np.array([zlim[0], zlim[0]])
    
    # 随机生成位置坐标
    x = np.random.uniform(xlim[0], xlim[1])
    y = np.random.uniform(ylim[0], ylim[1])

    # 根据条件调整y坐标
    while ylim_prop and abs(x) < 0.15 and y > 0:
        y = np.random.uniform(ylim[0], 0)
        
    z = np.random.uniform(zlim[0], zlim[1])
    position = np.array([x, y, z])  # 位置的numpy数组

    # 处理旋转
    rotate_quat = np.array(quat, dtype=np.float64)  # 初始四元数（修正原代码笔误 qpos→quat）
    angles_rad = np.zeros(3)  # 初始化旋转角（弧度）

    if rotate_rand:
        # 生成随机旋转角（先按角度制生成，再转换为弧度）
        angles_deg = np.array([
            np.random.uniform(-rotate_lim[i], rotate_lim[i]) 
            for i in range(3)
        ])
        angles_rad = np.deg2rad(angles_deg)  # 角度转弧度（核心修改）
        
        # 转换为四元数（transforms3d 要求输入为弧度）
        rand_quat = t3d.euler.euler2quat(
            angles_rad[0], angles_rad[1], angles_rad[2]
        )
        # 与初始四元数相乘（叠加旋转）
        rotate_quat = t3d.quaternions.qmult(rotate_quat, rand_quat)

    # 旋转角的角度制输出（保持与原返回值一致）
    angles_deg = np.rad2deg(angles_rad)

    return position, rotate_quat, angles_deg
    


def seed_everything(seed):
    # if seed >= 10000:
    #     raise ValueError("seed number should be less than 10000")
    # seed = (rank * 100000) + seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
