import numpy as np
from scipy.spatial.transform import Rotation as R

def calculate_lookat_point(cam_position, rotation_angles_deg, near=0.1):
    """
    计算相机lookat点，默认朝向为世界坐标系x轴正方向
    
    参数:
        cam_position: 相机位置，格式为[x, y, z]
        rotation_angles_deg: 旋转角（角度制），格式为[rx, ry, rz]，
                             分别对应绕x轴、y轴、z轴的旋转角度（单位：度）
        near: 相机近平面距离
        
    返回:
        lookat_point: 相机看向的目标点
        world_forward: 世界坐标系中的前向向量
    """
    # 将角度（度）转换为弧度（旋转矩阵计算需要弧度）
    rotation_rad = np.radians(rotation_angles_deg)
    
    # 将旋转角转换为旋转矩阵（ZYX顺序）
    rot = R.from_euler('zyx', rotation_rad, degrees=False)
    rotation_matrix = rot.as_matrix()
    
    # 相机的前向向量（默认指向世界坐标系x轴正方向）
    forward = np.array([1, 0, 0])
    
    # 将前向向量转换到世界坐标系
    world_forward = rotation_matrix @ forward
    
    # 计算距离（近平面的2倍）
    distance = near * 2
    
    # 计算lookat点
    lookat_point = cam_position + world_forward * distance
    
    return lookat_point, world_forward

def main():
    # 相机参数
    cam_position = [0.0, 0.0, 1.0]  # 相机位置
    
    # 基础旋转角度（角度制，单位：度）
    # [0, 0, 0] 表示无旋转
    base_rotation_deg = [0, 0, 0]   # 明确标注为角度制（度）
    near = 0.05
    
    # 计算无旋转时的lookat点
    base_lookat, base_forward = calculate_lookat_point(
        cam_position, base_rotation_deg, near
    )
    
    # 旋转角度（角度制，单位：度）
    rotated_angles_deg = [0, 65, 0]  # 绕y轴旋转90度（角度制）
    rotated_lookat, rotated_forward = calculate_lookat_point(
        cam_position, rotated_angles_deg, near
    )
    
    print(f"相机位置: {cam_position}")
    print("\n无旋转时（旋转角为角度制）:")
    print(f"  旋转角度: {base_rotation_deg} 度")
    print(f"  前向向量: {base_forward}（世界坐标系x轴正方向）")
    print(f"  lookat点: {base_lookat}")
    
    print("\n旋转后（旋转角为角度制）:")
    print(f"  旋转角度: {rotated_angles_deg} 度")
    print(f"  前向向量: {rotated_forward}")
    print(f"  lookat点: {rotated_lookat}")

if __name__ == "__main__":
    main()
