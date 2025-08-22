import numpy as np
from scipy.spatial.transform import Rotation as R

def calculate_lookat_point(cam_position, rotation_angles_deg):
    """
    计算相机看向z=0平面的lookat点（相机默认朝向为x轴正向）
    
    参数:
        cam_position: 相机位置，格式为[x, y, z]（list/tuple/np.ndarray均可）
        rotation_angles_deg: 旋转角（角度制），格式为[rx, ry, rz]，
                             分别对应绕x轴（俯仰）、y轴（偏航）、z轴（滚转）的旋转，
                             旋转顺序为Z-Y-X（先绕z轴、再绕y轴、最后绕x轴）
        
    返回:
        lookat_point: 相机看向z=0平面的目标点，z坐标严格为0（np.ndarray）
        world_forward: 旋转后相机在世界坐标系中的前向向量（仍沿x轴正向衍生，np.ndarray）
    """
    # 1. 统一输入格式为numpy数组，避免类型兼容问题
    cam_pos = np.asarray(cam_position, dtype=np.float32)
    rot_angles_deg = np.asarray(rotation_angles_deg, dtype=np.float32)
    
    # 2. 角度转弧度（scipy的from_euler默认接收弧度）
    rot_rad = np.radians(rot_angles_deg)
    
    # 3. 欧拉角（Z-Y-X顺序）转旋转矩阵
    # 旋转顺序说明：先绕z轴转rz，再绕y轴转ry，最后绕x轴转rx（符合相机姿态控制常用逻辑）
    rot = R.from_euler(seq='zyx', angles=rot_rad, degrees=False)
    rotation_matrix = rot.as_matrix()  # 3x3旋转矩阵（相机坐标系→世界坐标系）
    
    # 4. 定义相机默认前向向量（朝向x轴正向）
    cam_forward_local = np.array([1, 0, 0], dtype=np.float32)  # 相机本地坐标系的前向（x轴正）
    
    # 5. 将本地前向向量转换到世界坐标系
    world_forward = rotation_matrix @ cam_forward_local  # 旋转后的世界坐标系前向向量
    
    # 6. 计算射线与z=0平面的交点（核心：确保lookat点z=0）
    # 射线方程：P(t) = cam_pos + t × world_forward（t为非负参数，代表沿前向的距离）
    # z=0平面求交：cam_pos[2] + t × world_forward[2] = 0 → 解t
    cam_z = cam_pos[2]
    forward_z = world_forward[2]
    
    # 特殊情况：前向向量z分量接近0（相机水平看向x轴，无上下倾斜）
    if abs(forward_z) < 1e-10:
        # 直接取z=0平面上与相机y坐标一致、x轴沿前向延伸的点（t取1避免lookat与相机位置重合）
        lookat_point = cam_pos + world_forward * 1.0  # t=1确保有一定距离
        lookat_point[2] = 0.0  # 强制z=0，消除浮点误差
    else:
        # 正常情况：解t值（t为负时表示前向朝向z>0方向，仍取交但确保逻辑正确）
        t = -cam_z / forward_z
        # 计算lookat点（t<0时仍有效，代表相机向后看z=0平面）
        lookat_point = cam_pos + t * world_forward
        lookat_point[2] = 0.0  # 强制z=0，避免浮点计算残留
    
    return lookat_point, world_forward

def main():
    # ----------------------
    # 测试用例1：无旋转（默认朝x轴正向）
    # ----------------------
    cam_pos1 = [0.0, 0.0, 1.0]  # 相机在(0,0,2)，z=2
    rot1 = [0, 70, 0]  # 无旋转
    lookat1, forward1 = calculate_lookat_point(cam_pos1, rot1)
    print("=== 测试用例1：无旋转（默认朝x轴正向）===")
    print(f"相机位置：{cam_pos1}")
    print(f"旋转角度：{rot1} 度")
    print(f"世界坐标系前向向量：{forward1.round(4)}")
    print(f"lookat点（z=0平面）：{lookat1.round(4)}（z={lookat1[2]}）\n")

    # ----------------------
    # 测试用例2：绕y轴旋转90度（朝向z轴正向）
    # ----------------------
    cam_pos2 = [0.0, 0.0, 2.0]  # 相机位置不变
    rot2 = [0, 90, 0]  # 绕y轴转90度（偏航），原x轴正向→z轴正向
    lookat2, forward2 = calculate_lookat_point(cam_pos2, rot2)
    print("=== 测试用例2：绕y轴旋转90度（朝z轴正向）===")
    print(f"相机位置：{cam_pos2}")
    print(f"旋转角度：{rot2} 度")
    print(f"世界坐标系前向向量：{forward2.round(4)}")
    print(f"lookat点（z=0平面）：{lookat2.round(4)}（z={lookat2[2]}）\n")

    # ----------------------
    # 测试用例3：绕x轴旋转-30度（向下倾斜，朝向x轴下方）
    # ----------------------
    cam_pos3 = [5.0, 3.0, 4.0]  # 相机在(5,3,4)，z=4
    rot3 = [-30, 0, 0]  # 绕x轴转-30度（俯仰向下）
    lookat3, forward3 = calculate_lookat_point(cam_pos3, rot3)
    print("=== 测试用例3：绕x轴旋转-30度（向下倾斜）===")
    print(f"相机位置：{cam_pos3}")
    print(f"旋转角度：{rot3} 度")
    print(f"世界坐标系前向向量：{forward3.round(4)}")
    print(f"lookat点（z=0平面）：{lookat3.round(4)}（z={lookat3[2]}）")

if __name__ == "__main__":
    main()