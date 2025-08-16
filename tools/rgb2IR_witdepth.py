import os
import cv2
import numpy as np
import pickle
import argparse
from tqdm import tqdm


def parse_arguments():
    """解析命令行参数，包含真实深度路径和人工微调参数"""
    parser = argparse.ArgumentParser(description='结合真实深度的IR与RGB对齐工具')
    # 输入文件路径
    parser.add_argument('--rgb_path', required=True, help='RGB视频/图像路径')
    parser.add_argument('--left_ir_path', required=True, help='左IR视频/图像路径')
    parser.add_argument('--right_ir_path', required=False, help='右IR视频/图像路径（可选）')
    parser.add_argument('--left_ir_depth', required=False, help='左IR对应的深度视频/图像路径（提升左IR精度）')
    parser.add_argument('--config_file', required=True, help='相机内参外参配置文件（pkl格式）')
    
    # 处理模式
    parser.add_argument('--is_image', action='store_true', help='输入为图像而非视频')
    
    # 增加人工微调参数（修正偏移）
    parser.add_argument('--x_offset', type=int, default=0, help='水平偏移（正值右移，修正左偏）')
    parser.add_argument('--y_offset', type=int, default=0, help='垂直偏移（正值下移）')
    
    # 优化选项
    parser.add_argument('--subpixel_refine', action='store_true', default=True, help='启用亚像素级对齐')
    parser.add_argument('--vis_intermediate', action='store_true', help='可视化中间结果（调试用）')
    parser.add_argument('--output_dir', default=None, help='输出目录（默认与RGB文件同目录）')
    
    return parser.parse_args()


def load_camera_parameters(config_file):
    """加载并优化相机参数（旋转矩阵正交化）"""
    try:
        with open(config_file, 'rb') as f:
            params = pickle.load(f)
        
        # 旋转矩阵正交化（消除数值误差）
        for key in ['extrin_color_to_ir_left', 'extrin_color_to_ir_right']:
            if key in params:
                R = np.array(params[key]['rotation']).reshape(3, 3)
                U, _, Vt = np.linalg.svd(R)
                R_ortho = U @ Vt
                params[key]['rotation'] = R_ortho.flatten().tolist()
        return params
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return None


def undistort_image(image, K, dist_coeffs):
    """图像去畸变处理"""
    h, w = image.shape[:2]
    new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
    return cv2.undistort(image, K, dist_coeffs, None, new_cam_mat), new_cam_mat


def preprocess_depth(depth_image, min_depth=0.3, max_depth=5.0):
    """深度图预处理（单位转换、滤波、填洞）"""
    # 转换为米（假设输入为毫米）
    depth_m = depth_image.astype(np.float32) / 1000.0
    
    # 过滤无效深度
    mask = (depth_m >= min_depth) & (depth_m <= max_depth)
    depth_m[~mask] = 0.0
    
    # 填充空洞
    if np.any(depth_m == 0):
        inpaint_mask = (depth_m == 0).astype(np.uint8) * 255
        depth_m = cv2.inpaint(depth_m, inpaint_mask, 3, cv2.INPAINT_TELEA)
    
    return depth_m


def align_ir_to_rgb_with_depth(rgb_img, ir_img, depth_img, rgb_K, ir_K, extrinsics, x_offset=0, y_offset=0):
    """使用真实深度将左IR对齐到RGB"""
    h, w = ir_img.shape[:2]
    R = np.array(extrinsics['rotation']).reshape(3, 3)
    t = np.array(extrinsics['translation']).reshape(3, 1)
    T_ir_to_rgb = np.eye(4)
    T_ir_to_rgb[:3, :3] = R.T
    T_ir_to_rgb[:3, 3] = -R.T @ t

    # 深度图预处理
    depth_m = preprocess_depth(depth_img)
    
    # 生成IR像素网格
    y_idx, x_idx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    z_flat = depth_m.flatten()

    # 过滤无效深度
    valid = z_flat > 0
    x_valid = x_flat[valid]
    y_valid = y_flat[valid]
    z_valid = z_flat[valid]

    # 计算IR相机坐标系3D点
    pts_ir = np.vstack([
        (x_valid - ir_K[0, 2]) * z_valid / ir_K[0, 0],
        (y_valid - ir_K[1, 2]) * z_valid / ir_K[1, 1],
        z_valid
    ])

    # 转换到RGB相机坐标系
    pts_hom = np.vstack([pts_ir, np.ones((1, len(x_valid)))])
    pts_rgb_hom = T_ir_to_rgb @ pts_hom
    pts_rgb = pts_rgb_hom[:3, :] / pts_rgb_hom[3, :]

    # 投影到RGB图像
    pts_proj = rgb_K @ pts_rgb
    u_rgb = pts_proj[0, :] / pts_proj[2, :] + x_offset
    v_rgb = pts_proj[1, :] / pts_proj[2, :] + y_offset

    # 生成对齐后的RGB图像
    aligned_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_h, rgb_w = rgb_img.shape[:2]
    valid_uv = (u_rgb >= 0) & (u_rgb < rgb_w) & (v_rgb >= 0) & (v_rgb < rgb_h)

    # 亚像素精度映射
    x_ir = x_valid[valid_uv].astype(int)
    y_ir = y_valid[valid_uv].astype(int)
    u_rgb_valid = u_rgb[valid_uv].astype(np.float32)
    v_rgb_valid = v_rgb[valid_uv].astype(np.float32)

    for i in range(len(x_ir)):
        aligned_rgb[y_ir[i], x_ir[i]] = cv2.getRectSubPix(
            rgb_img, (1, 1), (u_rgb_valid[i], v_rgb_valid[i])
        ).flatten()

    return aligned_rgb


def align_ir_to_rgb_basic(rgb_img, ir_img, rgb_K, ir_K, extrinsics, x_offset=0, y_offset=0):
    """基础对齐方法（无深度时使用）"""
    h, w = ir_img.shape[:2]
    R = np.array(extrinsics['rotation']).reshape(3, 3)
    t = np.array(extrinsics['translation']).reshape(3, 1)
    T_ir_to_rgb = np.eye(4)
    T_ir_to_rgb[:3, :3] = R.T
    T_ir_to_rgb[:3, 3] = -R.T @ t

    # 生成像素网格（假设深度1米）
    y_idx, x_idx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    z_flat = np.ones_like(x_flat)  # 假设深度

    # 3D投影计算（同深度方法）
    pts_ir = np.vstack([
        (x_flat - ir_K[0, 2]) * z_flat / ir_K[0, 0],
        (y_flat - ir_K[1, 2]) * z_flat / ir_K[1, 1],
        z_flat
    ])

    pts_hom = np.vstack([pts_ir, np.ones((1, len(x_flat)))])
    pts_rgb_hom = T_ir_to_rgb @ pts_hom
    pts_rgb = pts_rgb_hom[:3, :] / pts_rgb_hom[3, :]

    pts_proj = rgb_K @ pts_rgb
    u_rgb = (pts_proj[0, :] / pts_proj[2, :] + x_offset).reshape(h, w).astype(np.float32)
    v_rgb = (pts_proj[1, :] / pts_proj[2, :] + y_offset).reshape(h, w).astype(np.float32)

    return cv2.remap(
        rgb_img, u_rgb, v_rgb, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )


def colorize_left_ir(rgb, ir, depth, params, args):
    """左IR彩色化主函数（支持深度/无深度两种模式）"""
    # 相机参数
    rgb_K = np.array(params['K_matrix'])
    rgb_dist = np.array(params['coeffs'])
    ir_K = np.array(params['ir_left_K'])
    ir_dist = np.array(params['ir_left_coeffs'])
    extrinsics = params['extrin_color_to_ir_left']

    # 去畸变
    rgb_undist, _ = undistort_image(rgb, rgb_K, rgb_dist)
    ir_undist, _ = undistort_image(ir, ir_K, ir_dist)

    # 对齐
    if depth is not None:
        depth_undist, _ = undistort_image(depth, ir_K, ir_dist)
        aligned = align_ir_to_rgb_with_depth(
            rgb_undist, ir_undist, depth_undist,
            rgb_K, ir_K, extrinsics,
            args.x_offset, args.y_offset
        )
    else:
        aligned = align_ir_to_rgb_basic(
            rgb_undist, ir_undist,
            rgb_K, ir_K, extrinsics,
            args.x_offset, args.y_offset
        )

    # 生成掩码
    ir_gray = cv2.cvtColor(ir_undist, cv2.COLOR_BGR2GRAY) if len(ir_undist.shape) == 3 else ir_undist
    mask = (ir_gray > 5).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3)))
    colorized = aligned * mask[..., None]

    # 可视化中间结果
    if args.vis_intermediate:
        cv2.imshow("IR Mask", mask * 255)
        cv2.imshow("Aligned RGB", aligned)
        cv2.imshow("Colorized Left IR", colorized)
        cv2.waitKey(1 if not args.is_image else 0)

    return colorized


def colorize_right_ir(rgb, ir, params, args):
    """右IR彩色化（无深度支持）"""
    rgb_K = np.array(params['K_matrix'])
    rgb_dist = np.array(params['coeffs'])
    ir_K = np.array(params['ir_right_K'])
    ir_dist = np.array(params['ir_right_coeffs'])
    extrinsics = params['extrin_color_to_ir_right']

    rgb_undist, _ = undistort_image(rgb, rgb_K, rgb_dist)
    ir_undist, _ = undistort_image(ir, ir_K, ir_dist)

    aligned = align_ir_to_rgb_basic(
        rgb_undist, ir_undist,
        rgb_K, ir_K, extrinsics
    )

    ir_gray = cv2.cvtColor(ir_undist, cv2.COLOR_BGR2GRAY) if len(ir_undist.shape) == 3 else ir_undist
    mask = (ir_gray > 5).astype(np.uint8)
    return aligned * mask[..., None]


def process_image(args, params, output_dir):
    """处理图像文件"""
    # 读取图像
    rgb = cv2.imread(args.rgb_path)
    left_ir = cv2.imread(args.left_ir_path, cv2.IMREAD_GRAYSCALE)
    right_ir = cv2.imread(args.right_ir_path, cv2.IMREAD_GRAYSCALE) if args.right_ir_path else None

    # 兼容npy
    if args.depth_path.endswith("mp4"):
        depth = cv2.imread(args.depth_path, cv2.IMREAD_ANYDEPTH) 
    else:
        depth = np.load(args.depth_path)

    # 处理左IR
    left_colorized = colorize_left_ir(rgb, left_ir, depth, params, args)
    left_out = os.path.join(output_dir, f"left_ir_colorized{os.path.splitext(args.rgb_path)[1]}")
    cv2.imwrite(left_out, left_colorized)
    print(f"左IR彩色化图像已保存: {left_out}")

    # 处理右IR（如果提供）
    if right_ir is not None:
        right_colorized = colorize_right_ir(rgb, right_ir, params, args)
        right_out = os.path.join(output_dir, f"right_ir_colorized{os.path.splitext(args.rgb_path)[1]}")
        cv2.imwrite(right_out, right_colorized)
        print(f"右IR彩色化图像已保存: {right_out}")

    if args.vis_intermediate:
        cv2.waitKey(0)


def process_video(args, params, output_dir):
    """处理视频文件"""
    # 打开视频流
    rgb_cap = cv2.VideoCapture(args.rgb_path)
    left_cap = cv2.VideoCapture(args.left_ir_path)
    right_cap = cv2.VideoCapture(args.right_ir_path) if args.right_ir_path else None
    depth_cap = cv2.VideoCapture(args.depth_path) if args.depth_path else None

    # 获取视频参数
    fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(params['ir_left_width'])
    height = int(params['ir_left_height'])

    # 输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    left_out = os.path.join(output_dir, "left_ir_colorized.mp4")
    left_writer = cv2.VideoWriter(left_out, fourcc, fps, (width, height))

    right_writer = None
    if right_cap is not None:
        right_out = os.path.join(output_dir, "right_ir_colorized.mp4")
        right_writer = cv2.VideoWriter(right_out, fourcc, fps, (width, height))

    # 逐帧处理
    pbar = tqdm(total=frame_count, desc="处理视频")
    while True:
        ret_rgb, rgb = rgb_cap.read()
        ret_left, left_ir = left_cap.read()
        if not ret_rgb or not ret_left:
            break

        # 读取深度帧
        depth = None
        if depth_cap is not None:
            ret_depth, depth = depth_cap.read()
            if not ret_depth:
                depth = None

        # 处理左IR
        left_colorized = colorize_left_ir(rgb, left_ir, depth, params, args)
        left_writer.write(left_colorized)

        # 处理右IR
        if right_cap is not None:
            ret_right, right_ir = right_cap.read()
            if ret_right:
                right_colorized = colorize_right_ir(rgb, right_ir, params, args)
                right_writer.write(right_colorized)

        pbar.update(1)

    # 释放资源
    pbar.close()
    rgb_cap.release()
    left_cap.release()
    left_writer.release()
    print(f"左IR彩色化视频已保存: {left_out}")

    if right_cap is not None:
        right_cap.release()
        right_writer.release()
        print(f"右IR彩色化视频已保存: {right_out}")
    if depth_cap is not None:
        depth_cap.release()


def main():
    args = parse_arguments()
    params = load_camera_parameters(args.config_file)
    if not params:
        return

    # 输出目录
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.rgb_path)
    os.makedirs(output_dir, exist_ok=True)

    # 处理图像/视频
    if args.is_image:
        process_image(args, params, output_dir)
    else:
        process_video(args, params, output_dir)

    cv2.destroyAllWindows()
    print("处理完成")


if __name__ == "__main__":
    main()