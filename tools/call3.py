import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import json
from datetime import datetime
import os
from scipy.spatial.transform import Rotation as R  # 用于旋转向量转欧拉角


# --- 标定所需的最少图像数 ---
MIN_IMAGES = 15

# --- ArUco字典定义 ---
CHARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)


def rotation_vector_to_matrix(rvec):
    """将旋转向量转换为旋转矩阵"""
    rmat, _ = cv2.Rodrigues(rvec)
    return rmat


def rotation_matrix_to_euler(rmat):
    """将旋转矩阵转换为ZYX顺序的欧拉角（弧度）"""
    return R.from_matrix(rmat).as_euler('zyx')


# 用于将NumPy数据类型转换为JSON兼容类型的辅助类
class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理Numpy数据类型"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Numpy数组→列表
        if isinstance(obj, np.integer):
            return int(obj)      # Numpy整数→Python整数
        if isinstance(obj, np.floating):
            return float(obj)    # Numpy浮点数→Python浮点数
        if isinstance(obj, np.bool_):
            return bool(obj)     # Numpy布尔值→Python布尔值
        return super(NumpyEncoder, self).default(obj)


def setup_arg_parser():
    """设置命令行参数解析（输入格子数）"""
    parser = argparse.ArgumentParser(description="RealSense Camera Calibration with Parametric Patterns")
    
    # --- 通用参数 ---
    parser.add_argument(
        '--type',
        type=str,
        required=True,
        choices=['chessboard', 'charuco'],
        help="标定板类型 ('chessboard' 或 'charuco')"
    )

    parser.add_argument(
        '--output_file',
        type=str,
        default='realsense_calibration.json',
        help="保存标定结果的JSON文件路径"
    )
    
    parser.add_argument(
        '--show_undistorted',
        action='store_true',
        help="标定后显示去畸变前后的图像对比"
    )

    # --- 棋盘格参数（输入格子数） ---
    parser.add_argument('--grid_width', type=int, default=14, help="棋盘格横向格子总数（如15表示15列格子）")
    parser.add_argument('--grid_height', type=int, default=9, help="棋盘格纵向格子总数（如10表示10行格子）")
    parser.add_argument('--square_size', type=float, default=0.02, help="棋盘格方块的边长（单位：米）")

    # --- ChArUco板参数 ---
    parser.add_argument('--charuco_squares_x', type=int, default=11, help="ChArUco板水平方向的格子数")
    parser.add_argument('--charuco_squares_y', type=int, default=8, help="ChArUco板垂直方向的格子数")
    parser.add_argument('--charuco_square_length', type=float, default=0.04, help="ChArUco板格子边长（单位：米）")
    parser.add_argument('--charuco_marker_length', type=float, default=0.02, help="ChArUco板ArUco标记边长（单位：米）")

    return parser.parse_args()


def setup_realsense():
    """初始化并配置RealSense相机"""
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"启动RealSense相机时出错: {e}")
        print("请确认相机已连接且未被其他程序占用。")
        return None, None
        
    return pipeline, profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()


def create_board_object_points(args):
    """根据传入的参数动态创建标定板的3D世界坐标（修复形状不匹配问题）"""
    if args.type == 'chessboard':
        # 1. 计算角点数（格子数-1）
        corner_width = args.grid_width - 1  # 横向角点数
        corner_height = args.grid_height - 1  # 纵向角点数
        checkerboard_size = (corner_width, corner_height)
        
        # 2. 创建3D世界坐标（基于角点数生成网格，避免形状不匹配）
        objp = np.zeros((corner_width * corner_height, 3), np.float32)
        # 关键修复：用角点数范围生成网格（0到corner_width，而非grid_width）
        objp[:, :2] = np.mgrid[0:corner_width, 0:corner_height].T.reshape(-1, 2)
        objp *= args.square_size  # 缩放为实际物理尺寸
        return objp, None, checkerboard_size
        
    elif args.type == 'charuco':
        # 保持不变
        board = cv2.aruco.CharucoBoard(
            (args.charuco_squares_x, args.charuco_squares_y),
            args.charuco_square_length,
            args.charuco_marker_length,
            CHARUCO_DICT
        )
        return board.getChessboardCorners(), board, None
    
    return None, None, None


def main():
    """主函数，执行标定流程"""
    args = setup_arg_parser()
    pipeline, rs_intrinsics = setup_realsense()
    if pipeline is None: return

    # 打印格子数与角点数的对应关系，提示用户
    if args.type == 'chessboard':
        print(f"检测配置：{args.grid_width}x{args.grid_height}个格子 → 对应{args.grid_width-1}x{args.grid_height-1}个角点")
    print("请将相机对准标定板。按 's' 保存稳定帧，按 'q' 退出并开始标定。")

    objp, board, checkerboard_size = create_board_object_points(args)
    obj_points, img_points = [], []
    all_charuco_corners, all_charuco_ids = [], []
    last_captured_image = None
    gray = None
    
    # 新增：创建以当前时间命名的图片保存文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 时间格式：年-月-日_时-分-秒
    image_save_dir = os.path.join("images", timestamp)  # 完整路径：根目录/时间戳
    os.makedirs(image_save_dir, exist_ok=True)  # 确保文件夹存在
    print(f"标定图片将保存至: {image_save_dir}")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue

            image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯降噪
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 局部对比度增强
            gray = clahe.apply(gray)
            display_image = image.copy()
            
            found_pattern, corners = False, None
            
            if args.type == 'chessboard':
                # 优化：检测参数增加FILTER_QUADS
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | \
                      cv2.CALIB_CB_FAST_CHECK | \
                      cv2.CALIB_CB_NORMALIZE_IMAGE | \
                      cv2.CALIB_CB_FILTER_QUADS  # 过滤非四边形角点
                ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None, flags=flags)
                # 优化检测鲁棒性：添加自适应阈值和快速检查
                # ret, corners = cv2.findChessboardCorners(
                #     gray, 
                #     checkerboard_size, 
                #     None,
                #     flags=cv2.CALIB_CB_ADAPTIVE_THRESH  # 适应光照变化
                #     | cv2.CALIB_CB_FAST_CHECK          # 快速排除无棋盘格的帧
                #     | cv2.CALIB_CB_NORMALIZE_IMAGE     # 归一化对比度
                # )
                if ret:
                    found_pattern = True
                    # 亚像素级角点优化
                    corners = cv2.cornerSubPix(
                        gray, 
                        corners, 
                        (11, 11), 
                        (-1, -1), 
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    cv2.drawChessboardCorners(display_image, checkerboard_size, corners, ret)

            elif args.type == 'charuco':
                detector = cv2.aruco.ArucoDetector(CHARUCO_DICT, cv2.aruco.DetectorParameters())
                marker_corners, marker_ids, _ = detector.detectMarkers(gray)
                
                if marker_ids is not None and len(marker_ids) > 4:
                    num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        marker_corners, marker_ids, gray, board
                    )
                    if num_corners > 4:
                        found_pattern, corners = True, charuco_corners
                        cv2.aruco.drawDetectedCornersCharuco(display_image, charuco_corners, charuco_ids, (0, 255, 0))

            # 绘制坐标轴（如果检测到）
            if found_pattern:
                h, w = gray.shape
                initial_cam_matrix = np.array([
                    [rs_intrinsics.fx, 0, rs_intrinsics.ppx], 
                    [0, rs_intrinsics.fy, rs_intrinsics.ppy], 
                    [0, 0, 1]
                ], dtype=np.float32)
                initial_dist_coeffs = np.zeros(5)
                rvec, tvec = None, None
                if args.type == 'chessboard':
                    _, rvec, tvec = cv2.solvePnP(objp, corners, initial_cam_matrix, initial_dist_coeffs)
                elif args.type == 'charuco' and corners is not None:
                    _, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        corners, charuco_ids, board, initial_cam_matrix, initial_dist_coeffs
                    )
                if rvec is not None and tvec is not None:
                    cv2.drawFrameAxes(display_image, initial_cam_matrix, initial_dist_coeffs, rvec, tvec, 0.1)

            # 显示已保存帧数
            saved_count = len(obj_points) if args.type == 'chessboard' else len(all_charuco_corners)
            progress_text = f"已保存: {saved_count} / {MIN_IMAGES}"
            cv2.putText(display_image, progress_text, (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('RealSense Calibration', display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("退出捕获模式...")
                break
            if key == ord('s') and found_pattern:
                saved_count += 1
                print(f"捕获帧. 总数: {saved_count}")
                if args.type == 'chessboard':
                    obj_points.append(objp)
                    img_points.append(corners)
                elif args.type == 'charuco':
                    all_charuco_corners.append(corners)
                    all_charuco_ids.append(charuco_ids)
                last_captured_image = image.copy()
                cv2.imwrite(f"{image_save_dir}/call_{saved_count}.png", last_captured_image)
                print(f"已保存: {image_save_dir}/call_{saved_count}.png")
            elif key == ord('s') and not found_pattern:
                 print("未检测到标定板，无法保存。请调整角度或光照。")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    saved_count = len(obj_points) if args.type == 'chessboard' else len(all_charuco_corners)
    if saved_count < MIN_IMAGES:
        print(f"\n标定需要至少 {MIN_IMAGES} 张有效图像，当前只有 {saved_count} 张。终止标定。")
        return

    print("\n开始相机标定...")
    h, w = gray.shape
    
    ret, mtx, dist, rvecs, tvecs = None, None, None, None, None
    # 新增：处理外参（旋转向量→旋转矩阵→欧拉角）
    extrinsics = []
    for i in range(len(rvecs)):
        # 旋转向量→旋转矩阵
        rmat, _ = cv2.Rodrigues(rvecs[i])
        # 旋转矩阵→欧拉角（ZYX顺序，单位：度）
        euler = rotation_matrix_to_euler(rmat) * 180 / np.pi  # 转为角度
        extrinsics.append({
            "rotation_vector": rvecs[i].flatten().tolist(),  # 旋转向量（弧度）
            "rotation_matrix": rmat.tolist(),                # 旋转矩阵
            "euler_angles": euler.tolist(),                   # 欧拉角（度，yaw-pitch-roll）
            "translation_vector": tvecs[i].flatten().tolist() # 平移向量（米）
        })

    # 打印标定结果（包含外参）
    print("\n--- 标定结果 ---")
    print(f"相机内参矩阵:\n{mtx}")
    print(f"畸变系数:\n{dist}")
    reprojection_error = np.mean([
        cv2.norm(imgp, cv2.projectPoints(objp, rvec, tvec, mtx, dist)[0], cv2.NORM_L2)/len(imgp) 
        for objp, imgp, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs)
    ])
    print(f"重投影误差: {reprojection_error:.4f} 像素")
    print(f"\n外参数量: {len(extrinsics)}（与有效帧数一致）")
    print("第一帧外参示例:")
    print(f"  旋转矩阵:\n{np.array(extrinsics[0]['rotation_matrix'])}")
    print(f"  欧拉角 (yaw, pitch, roll 度): {extrinsics[0]['euler_angles']}")
    print(f"  平移向量 (x, y, z 米): {extrinsics[0]['translation_vector']}")

    # 保存标定数据（包含外参）
    calibration_data = {
        'camera_matrix': mtx.tolist(),
        'distortion_coefficients': dist.flatten().tolist(),
        'reprojection_error': reprojection_error,
        'image_size': (w, h),
        'calibration_type': args.type,
        'calibration_time': timestamp,
        'image_save_directory': image_save_dir,
        'extrinsics': extrinsics  # 新增：外参列表（每帧对应一个外参）
    }
    
    try:
        with open(args.output_file, 'w') as f:
            json.dump(calibration_data, f, cls=NumpyEncoder, indent=4)
        print(f"\n标定数据（含外参）已保存至 {args.output_file}")
    except Exception as e:
        print(f"保存文件失败: {e}")
    try:
        with open(args.output_file, 'w') as f:
            json.dump(calibration_data, f, cls=NumpyEncoder, indent=4)
        print(f"标定数据已保存至 {args.output_file}")
    except Exception as e:
        print(f"保存文件失败: {e}")

    # 显示去畸变对比
    if args.show_undistorted and last_captured_image is not None:
        print("\n显示原始图像与去畸变图像对比（按任意键退出）。")
        h_img, w_img = last_captured_image.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w_img, h_img), 1, (w_img, h_img))
        dst = cv2.undistort(last_captured_image, mtx, dist, None, new_camera_mtx)
        x, y, w_roi, h_roi = roi
        dst = dst[y:y+h_roi, x:x+w_roi]
        cv2.imshow('Original vs Undistorted', np.hstack((last_captured_image, dst)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()