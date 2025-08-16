import os
import cv2
import numpy as np
import json
import pickle
import argparse
from tqdm import tqdm

def parse_arguments():
    """解析命令行参数，支持视频和图像输入"""
    parser = argparse.ArgumentParser(description='将RGB图像的颜色映射到左右IR图像/视频')
    parser.add_argument('--rgb_input', required=False, 
                        default="/home/haichao/workspace/real2sim/simulation/data/20250709-1614/images/rgb_333422301209_1752048895.png",
                        help='RGB图像或视频文件路径')
    parser.add_argument('--left_ir_input', required=False,
                        default="/home/haichao/workspace/real2sim/simulation/data/20250709-1614/images/ir_left_333422301209_1752048895.png",
                        help='左IR图像或视频文件路径')
    parser.add_argument('--right_ir_input', required=False, 
                        default="/home/haichao/workspace/real2sim/simulation/data/20250709-1614/images/ir_right_333422301209_1752048895.png",
                        help='右IR图像或视频文件路径')
    parser.add_argument('--config_file', required=False, 
                        default="/home/haichao/workspace/real2sim/simulation/data/20250709-1614/video/intrinsics.pkl", 
                        help='相机内参和外参配置文件路径')
    parser.add_argument('--output_dir', required=False, 
                        default=None, help='输出目录，默认为输入文件所在目录')
    parser.add_argument('--is_image', action='store_true', default=True, 
                        help='指定输入为图像文件而非视频')
    return parser.parse_args()

def load_camera_parameters(config_file):
    """加载相机内参和外参"""
    try:
        with open(config_file, 'rb') as f:
            params = pickle.load(f)
        return params
    except Exception as e:
        print(f"无法加载配置文件: {e}")
        return None

def undistort_image(image, K, dist_coeffs):
    """去畸变"""
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(image, K, dist_coeffs, None, new_camera_matrix)
    return undistorted, new_camera_matrix

def align(
    intrinsics_ir, intrinsics_rgb, extrinsics_rgb2ir, ir_image, rgb_image_shape
) -> np.ndarray:
    """
    使用真实深度值将IR图像投影到RGB坐标系
    参数:
        intrinsics_ir: IR相机内参矩阵 (3x3)
        intrinsics_rgb: RGB相机内参矩阵 (3x3)
        extrinsics_rgb2ir: 从RGB到IR的外参矩阵 (4x4)
        ir_image: IR图像（作为深度图使用）
        rgb_image_shape: RGB图像尺寸 (width, height)
    返回:
        投影到RGB坐标系的深度图
    """
    width, height = rgb_image_shape
    # 初始化输出为NaN（无效值）
    out = np.full((height, width), np.nan)
    
    # 生成IR图像的像素坐标网格
    y, x = np.meshgrid(
        np.arange(ir_image.shape[0]), np.arange(ir_image.shape[1]), indexing="ij"
    )

    if len(ir_image.shape) == 3:
        ir_gray = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
    else:
        ir_gray = ir_image

    # 将坐标转换为一维数组（避免复杂的索引操作）
    x = x.flatten()  # 形状为 (N,)
    y = y.flatten()
    z = ir_gray.flatten()  # 同样转为一维数组
    
    # 过滤无效深度（<=0的深度值）
    valid_depth_mask = (z > 0)
    x = x[valid_depth_mask]
    y = y[valid_depth_mask]
    z = z[valid_depth_mask]
    
    if x.size == 0:
        print("警告：IR图像中没有有效深度值")
        return out
    
    # 将IR像素坐标转换为相机坐标系3D点
    x_cam = (x - intrinsics_ir[0, 2]) / intrinsics_ir[0, 0]
    y_cam = (y - intrinsics_ir[1, 2]) / intrinsics_ir[1, 1]
    pts_ir = np.vstack((x_cam * z, y_cam * z, z))  # 3×N的3D点
    
    # 转换到RGB相机坐标系（外参应用）
    R = extrinsics_rgb2ir[:3, :3]  # 旋转矩阵
    t = extrinsics_rgb2ir[:3, 3:].reshape(3, 1)  # 平移向量
    pts_rgb = R @ pts_ir + t  # 3×N的3D点
    
    # 投影到RGB图像平面
    pts_proj = intrinsics_rgb @ pts_rgb
    px = np.round(pts_proj[0, :] / pts_proj[2, :]).astype(int)  # RGB图像x坐标
    py = np.round(pts_proj[1, :] / pts_proj[2, :]).astype(int)  # RGB图像y坐标
    
    # 过滤在RGB图像范围内的点
    mask = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    
    # 赋值有效深度到输出
    out[py[mask], px[mask]] = pts_proj[2, mask]
    return out

def colorize_ir_with_real_depth(rgb_frame, ir_frame, params, ir_side='left'):
    """使用真实深度值的align函数将RGB颜色映射到IR帧"""
    # 获取相机参数
    K_rgb = np.array(params['K_matrix'])
    dist_coeffs_rgb = np.array(params['coeffs'])
    
    if ir_side == 'left':
        K_ir = np.array(params['ir_left_K'])
        dist_coeffs_ir = np.array(params['ir_left_coeffs'])
        rgb_to_ir_extrinsics = params['extrin_color_to_ir_left']
    else:
        K_ir = np.array(params['ir_right_K'])
        dist_coeffs_ir = np.array(params['ir_right_coeffs'])
        rgb_to_ir_extrinsics = params['extrin_color_to_ir_right']
    
    # 对RGB图像进行去畸变处理
    rgb_undistorted, new_K_rgb = undistort_image(rgb_frame, K_rgb, dist_coeffs_rgb)
    rgb_image_shape = (int(new_K_rgb[0, 2] * 2), int(new_K_rgb[1, 2] * 2))  # 假设图像中心在(px, py)
    
    # 提取外参矩阵（转换为4x4齐次矩阵）
    R = np.array(rgb_to_ir_extrinsics['rotation']).reshape(3, 3)
    t = np.array(rgb_to_ir_extrinsics['translation']).reshape(3, 1)
    extrinsics_matrix = np.eye(4)
    extrinsics_matrix[:3, :3] = R
    extrinsics_matrix[:3, 3:] = t
    
    # 使用改进的align函数计算深度投影
    depth_map = align(
        K_ir, new_K_rgb, extrinsics_matrix, ir_frame, rgb_image_shape
    )
    
    # 将深度映射转换为彩色IR图像
    h, w = rgb_image_shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    valid = ~np.isnan(depth_map)
    x_valid = x[valid]
    y_valid = y[valid]
    depth_valid = depth_map[valid]
    
    # 创建从IR到RGB的映射
    ir_h, ir_w = ir_frame.shape[:2]
    ir_y, ir_x = np.meshgrid(np.arange(ir_h), np.arange(ir_w), indexing='ij')
    ir_x = ir_x.reshape(-1)
    ir_y = ir_y.reshape(-1)
    
    # 映射RGB颜色到IR
    colorized_ir = np.zeros_like(ir_frame)
    for i in range(len(ir_x)):
        if 0 <= ir_x[i] < ir_w and 0 <= ir_y[i] < ir_h:
            # 找到最近的有效RGB坐标
            if np.any(valid):
                dist = (x_valid - ir_x[i])**2 + (y_valid - ir_y[i])** 2
                closest = np.argmin(dist)
                colorized_ir[ir_y[i], ir_x[i]] = rgb_undistorted[y_valid[closest], x_valid[closest]]
    
    return colorized_ir

def align_ir_to_rgb(rgb_image, ir_image, rgb_intrinsics, ir_intrinsics, rgb_to_ir_extrinsics):
    # 提取内参和外参
    K_rgb = np.array(rgb_intrinsics)
    K_ir = np.array(ir_intrinsics)
    R = np.array(rgb_to_ir_extrinsics['rotation']).reshape(3, 3)
    t = np.array(rgb_to_ir_extrinsics['translation']).reshape(3, 1)
    
    # 创建从RGB到IR的变换矩阵
    T_rgb_to_ir = np.eye(4)
    T_rgb_to_ir[:3, :3] = R
    T_rgb_to_ir[:3, 3] = t.flatten()
    
    # 生成IR图像的像素网格
    h_ir, w_ir = ir_image.shape[:2]
    y, x = np.meshgrid(np.arange(h_ir), np.arange(w_ir), indexing='ij')
    x = x.reshape(-1)
    y = y.reshape(-1)
    
    # 计算IR像素对应的3D点（假设深度为1米，用于投影）
    z = np.ones_like(x)  
    pts_ir = np.vstack([
        (x - K_ir[0, 2]) * z / K_ir[0, 0],
        (y - K_ir[1, 2]) * z / K_ir[1, 1],
        z
    ])
    
    # 将3D点从IR坐标系转换到RGB坐标系
    pts_rgb_homogeneous = np.vstack([pts_ir, np.ones((1, pts_ir.shape[1]))])
    pts_rgb_homogeneous = np.linalg.inv(T_rgb_to_ir) @ pts_rgb_homogeneous
    pts_rgb = pts_rgb_homogeneous[:3, :] / pts_rgb_homogeneous[3, :]
    
    # 将3D点投影到RGB图像平面
    pts_rgb_proj = K_rgb @ pts_rgb
    pts_rgb_proj = pts_rgb_proj / pts_rgb_proj[2, :]
    u_rgb = pts_rgb_proj[0, :].reshape(h_ir, w_ir).astype(np.float32)
    v_rgb = pts_rgb_proj[1, :].reshape(h_ir, w_ir).astype(np.float32)
    
    # 使用反向映射将RGB颜色映射到IR图像
    rgb_aligned = cv2.remap(
        rgb_image, 
        u_rgb, 
        v_rgb, 
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    return rgb_aligned

def colorize_ir_frame_improved(rgb_frame, ir_frame, params, ir_side='left'):
    """使用改进的对齐方法将RGB颜色映射到IR帧，包括去畸变和3D对齐"""
    # 获取相机参数
    K_rgb = np.array(params['K_matrix'])
    dist_coeffs_rgb = np.array(params['coeffs'])
    
    if ir_side == 'left':
        K_ir = np.array(params['ir_left_K'])
        dist_coeffs_ir = np.array(params['ir_left_coeffs'])
        rgb_to_ir_extrinsics = params['extrin_color_to_ir_left']
    else:
        K_ir = np.array(params['ir_right_K'])
        dist_coeffs_ir = np.array(params['ir_right_coeffs'])
        rgb_to_ir_extrinsics = params['extrin_color_to_ir_right']
    
    # 对RGB图像进行去畸变处理
    rgb_undistorted, _ = undistort_image(rgb_frame, K_rgb, dist_coeffs_rgb)
    
    # 使用3D对齐方法将RGB颜色映射到IR
    rgb_aligned = align_ir_to_rgb(
        rgb_undistorted, 
        ir_frame, 
        K_rgb, 
        K_ir, 
        rgb_to_ir_extrinsics
    )
    
    # IR掩码
    if len(ir_frame.shape) == 3:
        ir_gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
    else:
        ir_gray = ir_frame
    
    mask = ir_gray > 5  # 过滤掉过暗的像素
    
    # 确保掩码和rgb_aligned尺寸匹配
    if mask.shape != rgb_aligned.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (rgb_aligned.shape[1], rgb_aligned.shape[0])).astype(bool)
    
    # 应用掩码到对齐后的RGB图像
    colorized_ir = np.zeros_like(rgb_aligned)
    colorized_ir[mask] = rgb_aligned[mask]
    
    return colorized_ir

def process_image(rgb_path, left_ir_path, right_ir_path, params, output_dir):
    """处理单张图像并生成彩色化IR图像"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取图像文件
    rgb_image = cv2.imread(rgb_path)
    left_ir_image = cv2.imread(left_ir_path)
    right_ir_image = cv2.imread(right_ir_path)
    
    if rgb_image is None:
        raise FileNotFoundError(f"无法读取RGB图像: {rgb_path}")
    if left_ir_image is None:
        raise FileNotFoundError(f"无法读取左IR图像: {left_ir_path}")
    if right_ir_image is None:
        raise FileNotFoundError(f"无法读取右IR图像: {right_ir_path}")
    
    # 使用改进的方法进行彩色化处理
    left_colorized = colorize_ir_with_real_depth(rgb_image, left_ir_image, params, ir_side='left')
    right_colorized = colorize_ir_with_real_depth(rgb_image, right_ir_image, params, ir_side='right')
    
    # 生成输出文件名
    rgb_filename = os.path.basename(rgb_path)
    left_output_path = os.path.join(output_dir, f"left_ir_colorized_real_{rgb_filename}")
    right_output_path = os.path.join(output_dir, f"right_ir_colorized_real_{rgb_filename}")
    
    # 保存彩色化IR图像
    cv2.imwrite(left_output_path, left_colorized)
    cv2.imwrite(right_output_path, right_colorized)
    
    print(f"已保存改进的彩色化左IR图像到: {left_output_path}")
    print(f"已保存改进的彩色化右IR图像到: {right_output_path}")

def process_videos_improved(rgb_path, left_ir_path, right_ir_path, params, output_dir):
    """使用改进的方法处理视频并生成彩色化IR视频"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    rgb_cap = cv2.VideoCapture(rgb_path)
    left_ir_cap = cv2.VideoCapture(left_ir_path)
    right_ir_cap = cv2.VideoCapture(right_ir_path)
    
    # 检查视频是否成功打开
    if not rgb_cap.isOpened():
        raise ValueError(f"无法打开RGB视频: {rgb_path}")
    if not left_ir_cap.isOpened():
        raise ValueError(f"无法打开左IR视频: {left_ir_path}")
    if not right_ir_cap.isOpened():
        raise ValueError(f"无法打开右IR视频: {right_ir_path}")
    
    # 获取视频参数
    fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 确定输出视频尺寸
    width = int(params['ir_left_width'])
    height = int(params['ir_left_height'])
    
    # 创建输出视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    left_output_path = os.path.join(output_dir, "left_ir_colorized_real_improved.mp4")
    right_output_path = os.path.join(output_dir, "right_ir_colorized_real_improved.mp4")
    left_writer = cv2.VideoWriter(left_output_path, fourcc, fps, (width, height))
    right_writer = cv2.VideoWriter(right_output_path, fourcc, fps, (width, height))
    
    # 处理视频帧
    progress_bar = tqdm(total=frame_count, desc="处理视频")
    
    while True:
        ret_rgb, rgb_frame = rgb_cap.read()
        ret_left, left_ir_frame = left_ir_cap.read()
        ret_right, right_ir_frame = right_ir_cap.read()
        
        if not ret_rgb or not ret_left or not ret_right:
            break
        
        # 确保IR帧为三通道
        if len(left_ir_frame.shape) == 2:
            left_ir_frame = cv2.cvtColor(left_ir_frame, cv2.COLOR_GRAY2BGR)
        if len(right_ir_frame.shape) == 2:
            right_ir_frame = cv2.cvtColor(right_ir_frame, cv2.COLOR_GRAY2BGR)
        
        # 使用改进的方法进行彩色化处理
        left_colorized = colorize_ir_frame_improved(rgb_frame, left_ir_frame, params, ir_side='left')
        right_colorized = colorize_ir_frame_improved(rgb_frame, right_ir_frame, params, ir_side='right')
        
        # 写入输出视频
        left_writer.write(left_colorized)
        right_writer.write(right_colorized)
        
        progress_bar.update(1)
    
    progress_bar.close()
    rgb_cap.release()
    left_ir_cap.release()
    right_ir_cap.release()
    left_writer.release()
    right_writer.release()
    
    print(f"已保存改进的彩色化左IR视频到: {left_output_path}")
    print(f"已保存改进的彩色化右IR视频到: {right_output_path}")

def main():
    args = parse_arguments()
    
    # 加载相机参数
    params = load_camera_parameters(args.config_file)
    if params is None:
        print("无法继续处理，缺少相机参数")
        return
    
    # 检查外参
    if 'extrin_color_to_ir_left' not in params or 'extrin_color_to_ir_right' not in params:
        print("配置文件缺少必要的外参信息")
        print("请确保配置文件中包含extrin_color_to_ir_left和extrin_color_to_ir_right")
        return
    
    # 确定输出目录
    if args.output_dir is None:
        output_dir = os.path.dirname(args.rgb_input)
    else:
        output_dir = args.output_dir
    
    # try:
    if args.is_image:
        # 处理单张图像
        process_image(
            args.rgb_input, 
            args.left_ir_input, 
            args.right_ir_input, 
            params, 
            output_dir
        )
    else:
        # 处理视频
        process_videos_improved(
            args.rgb_input, 
            args.left_ir_input, 
            args.right_ir_input, 
            params, 
            output_dir
        )
    # except Exception as e:
    #     print(f"处理错误: {e}")

if __name__ == "__main__":
    main()