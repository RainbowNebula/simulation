import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import json
from datetime import datetime
import os
from scipy.spatial.transform import Rotation as R

# 真正使用的
# --- 标定所需的最少图像数 ---
MIN_IMAGES = 15

# --- ArUco字典定义 ---
CHARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)


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


def rotation_vector_to_matrix(rvec):
    """将旋转向量转换为旋转矩阵"""
    rmat, _ = cv2.Rodrigues(rvec)
    return rmat


def rotation_matrix_to_euler(rmat):
    """将旋转矩阵转换为ZYX顺序的欧拉角（弧度）"""
    return R.from_matrix(rmat).as_euler('zyx')


def setup_arg_parser():
    """设置命令行参数解析"""
    parser = argparse.ArgumentParser(description="RealSense Camera Calibration with Extrinsics")
    
    # 通用参数
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

    # 棋盘格参数
    parser.add_argument('--grid_width', type=int, default=14, help="棋盘格横向格子总数")
    parser.add_argument('--grid_height', type=int, default=9, help="棋盘格纵向格子总数")
    parser.add_argument('--square_size', type=float, default=0.02, help="棋盘格方块边长（米）")

    # ChArUco板参数
    parser.add_argument('--charuco_squares_x', type=int, default=11, help="ChArUco板水平格子数")
    parser.add_argument('--charuco_squares_y', type=int, default=8, help="ChArUco板垂直格子数")
    parser.add_argument('--charuco_square_length', type=float, default=0.04, help="ChArUco格子边长（米）")
    parser.add_argument('--charuco_marker_length', type=float, default=0.02, help="ChArUco标记边长（米）")

    return parser.parse_args()


def setup_realsense():
    """初始化并配置RealSense相机，返回相机及原始内参"""
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    # 启用彩色流
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"启动RealSense相机时出错: {e}")
        print("请确认相机已连接且未被其他程序占用。")
        return None, None
        
    # 获取相机原始内参（出厂校准参数）
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    
    # 打印原始内参和畸变系数
    print("\n--- RealSense相机原始内参（出厂默认） ---")
    print(f"图像尺寸: {intrinsics.width}x{intrinsics.height}")
    print(f"内参矩阵:")
    print(f"  fx: {intrinsics.fx:.2f} (x方向焦距)")
    print(f"  fy: {intrinsics.fy:.2f} (y方向焦距)")
    print(f"  cx: {intrinsics.ppx:.2f} (主点x坐标)")
    print(f"  cy: {intrinsics.ppy:.2f} (主点y坐标)")
    print(f"畸变系数: {intrinsics.coeffs} (k1, k2, p1, p2, k3)")
    print("----------------------------------------\n")
    
    return pipeline, intrinsics


def create_board_object_points(args):
    """创建标定板的3D世界坐标"""
    if args.type == 'chessboard':
        corner_width = args.grid_width - 1
        corner_height = args.grid_height - 1
        checkerboard_size = (corner_width, corner_height)
        
        objp = np.zeros((corner_width * corner_height, 3), np.float32)
        objp[:, :2] = np.mgrid[0:corner_width, 0:corner_height].T.reshape(-1, 2)
        objp *= args.square_size
        return objp, None, checkerboard_size
        
    elif args.type == 'charuco':
        board = cv2.aruco.CharucoBoard(
            (args.charuco_squares_x, args.charuco_squares_y),
            args.charuco_square_length,
            args.charuco_marker_length,
            CHARUCO_DICT
        )
        return board.getChessboardCorners(), board, None
    
    return None, None, None


def main():
    """主函数，执行标定流程（包含外参处理和原始内参打印）"""
    args = setup_arg_parser()
    pipeline, rs_intrinsics = setup_realsense()
    if pipeline is None: return

    if args.type == 'chessboard':
        print(f"检测配置：{args.grid_width}x{args.grid_height}个格子 → 对应{args.grid_width-1}x{args.grid_height-1}个角点")
    print("请将相机对准标定板。按 's' 保存稳定帧，按 'q' 退出并开始标定。")

    objp, board, checkerboard_size = create_board_object_points(args)
    obj_points, img_points = [], []
    all_charuco_corners, all_charuco_ids = [], []
    all_rvecs, all_tvecs = [], []  # 外参存储
    last_captured_image = None
    gray = None
    
    # 创建图片保存文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_save_dir = os.path.join("images", timestamp)
    os.makedirs(image_save_dir, exist_ok=True)
    print(f"标定图片将保存至: {image_save_dir}")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue

            image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            display_image = image.copy()
            
            found_pattern, corners = False, None
            rvec, tvec = None, None  # 当前帧的外参
            
            if args.type == 'chessboard':
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | \
                      cv2.CALIB_CB_FAST_CHECK | \
                      cv2.CALIB_CB_NORMALIZE_IMAGE | \
                      cv2.CALIB_CB_FILTER_QUADS
                ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None, flags=flags)
                if ret:
                    found_pattern = True
                    corners = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    cv2.drawChessboardCorners(display_image, checkerboard_size, corners, ret)
                    # 计算当前帧外参
                    _, rvec, tvec = cv2.solvePnP(objp, corners, 
                                               np.array([[rs_intrinsics.fx, 0, rs_intrinsics.ppx], 
                                                        [0, rs_intrinsics.fy, rs_intrinsics.ppy], 
                                                        [0, 0, 1]], dtype=np.float32), 
                                               np.zeros(5))

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
                        # 计算当前帧外参
                        _, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                            corners, charuco_ids, board,
                            np.array([[rs_intrinsics.fx, 0, rs_intrinsics.ppx], 
                                     [0, rs_intrinsics.fy, rs_intrinsics.ppy], 
                                     [0, 0, 1]], dtype=np.float32), 
                            np.zeros(5)
                        )

            # 绘制坐标轴
            if found_pattern and rvec is not None and tvec is not None:
                cv2.drawFrameAxes(display_image, 
                                 np.array([[rs_intrinsics.fx, 0, rs_intrinsics.ppx], 
                                          [0, rs_intrinsics.fy, rs_intrinsics.ppy], 
                                          [0, 0, 1]], dtype=np.float32), 
                                 np.zeros(5), 
                                 rvec, tvec, 0.1)

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
            if key == ord('s') and found_pattern and rvec is not None and tvec is not None:
                saved_count += 1
                print(f"捕获帧. 总数: {saved_count}")
                if args.type == 'chessboard':
                    obj_points.append(objp)
                    img_points.append(corners)
                else:
                    all_charuco_corners.append(corners)
                    all_charuco_ids.append(charuco_ids)
                all_rvecs.append(rvec)
                all_tvecs.append(tvec)
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
    
    # 计算内参和外参
    ret, mtx, dist, rvecs, tvecs = None, None, None, None, None
    if args.type == 'chessboard':
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, (w, h), None, None
        )
    else:
        calib_obj_points = [board.getChessboardCorners()[ids.flatten()] for ids in all_charuco_ids]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            calib_obj_points, all_charuco_corners, (w, h), None, None
        )

    if not ret:
        print("标定失败！")
        return
        
    # 处理外参
    extrinsics = []
    for i in range(len(rvecs)):
        rmat, _ = cv2.Rodrigues(rvecs[i])
        euler = rotation_matrix_to_euler(rmat) * 180 / np.pi  # 转为角度
        extrinsics.append({
            "rotation_vector": rvecs[i].flatten().tolist(),
            "rotation_matrix": rmat.tolist(),
            "euler_angles": euler.tolist(),
            "translation_vector": tvecs[i].flatten().tolist()
        })

    # 打印标定结果（与原始内参对比）
    print("\n--- 标定后内参结果 ---")
    print(f"图像尺寸: {w}x{h}")
    print(f"内参矩阵:")
    print(f"  fx: {mtx[0,0]:.2f} (原始: {rs_intrinsics.fx:.2f})")
    print(f"  fy: {mtx[1,1]:.2f} (原始: {rs_intrinsics.fy:.2f})")
    print(f"  cx: {mtx[0,2]:.2f} (原始: {rs_intrinsics.ppx:.2f})")
    print(f"  cy: {mtx[1,2]:.2f} (原始: {rs_intrinsics.ppy:.2f})")
    print(f"畸变系数 (k1, k2, p1, p2, k3):")
    print(f"  标定后: {dist.flatten()}")
    print(f"  原始:    {rs_intrinsics.coeffs}")
    print(f"重投影误差: {np.mean([cv2.norm(imgp, cv2.projectPoints(objp, rvec, tvec, mtx, dist)[0], cv2.NORM_L2)/len(imgp) for objp, imgp, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs)]):.4f} 像素")

    # 保存标定数据
    calibration_data = {
        'camera_matrix': mtx.tolist(),
        'distortion_coefficients': dist.flatten().tolist(),
        'reprojection_error': np.mean([cv2.norm(imgp, cv2.projectPoints(objp, rvec, tvec, mtx, dist)[0], cv2.NORM_L2)/len(imgp) for objp, imgp, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs)]),
        'image_size': (w, h),
        'calibration_type': args.type,
        'calibration_time': timestamp,
        'image_save_directory': image_save_dir,
        'extrinsics': extrinsics,
        'original_intrinsics': {  # 新增：保存原始内参
            'fx': rs_intrinsics.fx,
            'fy': rs_intrinsics.fy,
            'cx': rs_intrinsics.ppx,
            'cy': rs_intrinsics.ppy,
            'dist_coeffs': rs_intrinsics.coeffs,
            'width': rs_intrinsics.width,
            'height': rs_intrinsics.height
        }
    }
    
    try:
        with open(args.output_file, 'w') as f:
            json.dump(calibration_data, f, cls=NumpyEncoder, indent=4)
        print(f"\n标定数据（含原始内参和外参）已保存至 {args.output_file}")
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