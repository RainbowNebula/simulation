import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_to_rotation_matrix(roll, pitch, yaw, order='xyz'):
    """
    将欧拉角转换为旋转矩阵
    参数:
        roll, pitch, yaw: 欧拉角（弧度）
        order: 旋转顺序，默认为'xyz'
    返回:
        3x3 旋转矩阵
    """
    r = R.from_euler(order, [roll, pitch, yaw])
    return r.as_matrix()

def quat_to_rotation_matrix(quat, scalar_first=False):
    """
    将四元转换为旋转矩阵
    参数:
        quat: 四元数xyzw
    返回:
        3x3 旋转矩阵
    """
    r = R.from_quat(quat, scalar_first=scalar_first)
    return r.as_matrix()

def create_transform_matrix(rotation_matrix, translation):
    """
    创建4x4齐次变换矩阵
    参数:
        rotation_matrix: 3x3旋转矩阵
        translation: 3元素平移向量 [x, y, z]
    返回:
        4x4 齐次变换矩阵
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform

def transform_grasp_pose_euler(original_pose, rot_angles=[0, 0, 90]):
    """
    将抓取姿态从X轴闭合方向转换为Y轴闭合方向
    
    参数:
        original_pose: 4x4 numpy数组，表示原始抓取姿态的变换矩阵
        rot_angles: 列表，包含[roll, pitch, yaw]旋转角度（度），默认绕Z轴旋转90度
        
    返回:
        transformed_pose: 4x4 numpy数组，表示转换后的抓取姿态的变换矩阵
    """
    # 检查输入矩阵是否为4x4
    if original_pose.shape != (4, 4):
        raise ValueError("输入必须是4x4的变换矩阵")
    
    # 检查旋转角度参数是否有效
    if len(rot_angles) != 3:
        raise ValueError("rot_angles必须是包含3个元素的列表：[roll, pitch, yaw]")
    
    # 根据旋转角度创建旋转矩阵
    rotation_transform = R.from_euler('xyz', rot_angles, degrees=True)
    rotation_matrix = rotation_transform.as_matrix()
    
    # 转换为4x4矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    
    # 应用旋转变换: 新姿态 = 原始姿态 × 旋转矩阵
    transformed_pose = np.dot(original_pose, transform_matrix)
    
    return transformed_pose

def transform_grasp_poses(base_position, base_rpy, grasp_poses, rotation_order='xyz'):
    """
    转换抓取位姿，考虑机械臂基座相对于原点的位姿
    
    参数:
        base_position: 基座相对于原点的位置 [x, y, z]
        base_rpy: 基座相对于原点的旋转角（弧度）[roll, pitch, yaw]
        grasp_poses: (N, 4, 4) 数组，原始的抓取位姿
        rotation_order: 旋转顺序，默认为'xyz'
    
    返回:
        transformed_poses: (N, 4, 4) 数组，转换后的抓取位姿
    """
    # 计算基座的旋转矩阵
    base_rotation = euler_to_rotation_matrix(
        base_rpy[0], base_rpy[1], base_rpy[2], 
        order=rotation_order
    )
    
    # 创建基座相对于原点的变换矩阵
    base_transform = create_transform_matrix(base_rotation, base_position)
    
    # 对每个抓取位姿应用变换
    num_poses = grasp_poses.shape[0]
    transformed_poses = np.zeros_like(grasp_poses)
    
    for i in range(num_poses):
        # 左乘基座变换矩阵，得到新的位姿
        transformed_poses[i] = np.dot(base_transform, grasp_poses[i])
    
    return transformed_poses

# 示例用法
if __name__ == "__main__":
    # 示例参数
    base_position = [0.0, 0.0, 0.0]  # 基座位置
    base_rpy = [0, np.radians(0), np.radians(90)]  # 基座旋转角 (弧度)
    
    # 创建示例抓取位姿 (2个位姿)
    # 第一个位姿: 原点，无旋转
    grasp_pose1 = np.eye(4)
    # 第二个位姿: 有平移和旋转
    rot = euler_to_rotation_matrix(np.radians(10), np.radians(20), np.radians(30))
    grasp_pose2 = create_transform_matrix(rot, [0.5, 0.6, 0.7])
    
    # 组合成(N, 4, 4)数组
    grasp_poses = np.stack([grasp_pose1, grasp_pose2])
    
    # 转换抓取位姿
    transformed_poses = transform_grasp_poses(
        base_position, base_rpy, grasp_poses
    )
    
    # 打印结果
    print("原始抓取位姿 1:\n", grasp_poses[0])
    print("\n原始抓取位姿 2:\n", grasp_poses[1])
    print("\n转换后的抓取位姿 1:\n", transformed_poses[0])
    print("\n转换后的抓取位姿 2:\n", transformed_poses[1])
