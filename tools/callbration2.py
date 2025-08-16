import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import json # 导入json库
import os

# --- 标定所需的最少图像数 ---
MIN_IMAGES = 15

# --- ArUco字典定义 ---
CHARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)


# 用于将NumPy数据类型转换为JSON兼容类型的辅助类
class NumpyEncoder(json.JSONEncoder):
    """
    自定义JSON编码器，用于处理Numpy数据类型。
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # 将Numpy数组转换为列表
        if isinstance(obj, np.integer):
            return int(obj)      # 将Numpy整数转换为Python整数
        if isinstance(obj, np.floating):
            return float(obj)    # 将Numpy浮点数转换为Python浮点数
        if isinstance(obj, np.bool_):
            return bool(obj)     # 将Numpy布尔值转换为Python布尔值
        return super(NumpyEncoder, self).default(obj)


def setup_arg_parser():
    """设置命令行参数解析"""
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
        default='realsense_calibration.json', # <--- 默认文件名更新
        help="保存标定结果的JSON文件路径"
    )
    parser.add_argument(
        '--show_undistorted',
        action='store_true',
        help="标定后显示去畸变前后的图像对比"
    )

    # --- 棋盘格参数 ---
    parser.add_argument('--chessboard_width', type=int, default=13, help="棋盘格内部角点的宽度（角点数）")
    parser.add_argument('--chessboard_height', type=int, default=8, help="棋盘格内部角点的高度（角点数）")
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

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"启动RealSense相机时出错: {e}")
        print("请确认相机已连接且未被其他程序占用。")
        return None, None
        
    return pipeline, profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

def create_board_object_points(args):
    """根据传入的参数动态创建标定板的3D世界坐标"""
    if args.type == 'chessboard':
        checkerboard_size = (args.chessboard_width, args.chessboard_height)
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
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
    """主函数，执行标定流程"""
    args = setup_arg_parser()
    pipeline, rs_intrinsics = setup_realsense()
    if pipeline is None: return

    print("RealSense相机已初始化。请将相机对准标定板。")
    print("按 's' 保存稳定的一帧。按 'q' 退出并开始标定。")

    objp, board, checkerboard_size = create_board_object_points(args)
    obj_points, img_points = [], []
    all_charuco_corners, all_charuco_ids = [], []
    last_captured_image = None
    gray = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue

            image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            display_image = image.copy()
            
            found_pattern, corners = False, None
            
            if args.type == 'chessboard':
                ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
                if ret:
                    found_pattern = True
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    cv2.drawChessboardCorners(display_image, checkerboard_size, corners, ret)

            elif args.type == 'charuco':
                detector = cv2.aruco.ArucoDetector(CHARUCO_DICT, cv2.aruco.DetectorParameters())
                marker_corners, marker_ids, _ = detector.detectMarkers(gray)
                
                if marker_ids is not None and len(marker_ids) > 4:
                    num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
                    if num_corners > 4:
                        found_pattern, corners = True, charuco_corners
                        cv2.aruco.drawDetectedCornersCharuco(display_image, charuco_corners, charuco_ids, (0, 255, 0))

            if found_pattern:
                h, w = gray.shape
                initial_cam_matrix = np.array([[rs_intrinsics.fx, 0, rs_intrinsics.ppx], [0, rs_intrinsics.fy, rs_intrinsics.ppy], [0, 0, 1]], dtype=np.float32)
                initial_dist_coeffs = np.zeros(5)
                rvec, tvec = None, None
                if args.type == 'chessboard':
                    _, rvec, tvec = cv2.solvePnP(objp, corners, initial_cam_matrix, initial_dist_coeffs)
                elif args.type == 'charuco' and corners is not None and charuco_ids is not None:
                    _, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(corners, charuco_ids, board, initial_cam_matrix, initial_dist_coeffs)
                if rvec is not None and tvec is not None:
                    cv2.drawFrameAxes(display_image, initial_cam_matrix, initial_dist_coeffs, rvec, tvec, 0.1)

            saved_count = len(obj_points) if args.type == 'chessboard' else len(all_charuco_corners)
            progress_text = f"已保存: {saved_count} / {MIN_IMAGES}"
            cv2.putText(display_image, progress_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('RealSense Calibration', display_image)
            key = cv2.waitKey(1) & 0xFF

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
            elif key == ord('s') and not found_pattern:
                 print("未检测到标定板，无法保存。")
            elif key == ord('q'):
                print("退出捕获模式...")
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    saved_count = len(obj_points) if args.type == 'chessboard' else len(all_charuco_corners)
    if saved_count < MIN_IMAGES:
        print(f"\n标定需要至少 {MIN_IMAGES} 张有效图像，当前只有 {saved_count} 张。终止标定。")
        return

    print("\n开始相机标定... 请稍候。")
    h, w = gray.shape
    
    ret, mtx, dist, rvecs, tvecs = None, None, None, None, None
    if args.type == 'chessboard':
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    elif args.type == 'charuco':
        calib_obj_points = [board.getChessboardCorners()[ids.flatten()] for ids in all_charuco_ids]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(calib_obj_points, all_charuco_corners, (w, h), None, None)

    if not ret:
        print("标定失败！")
        return
        
    print("\n--- 标定结果 ---")
    print(f"相机内参矩阵 (fx, fy, cx, cy):\n{mtx}")
    print(f"\n畸变系数 (k1, k2, p1, p2, k3):\n{dist}")
    
    mean_error = 0
    calib_points = obj_points if args.type == 'chessboard' else calib_obj_points
    calib_corners = img_points if args.type == 'chessboard' else all_charuco_corners
    for i in range(len(calib_points)):
        imgpoints2, _ = cv2.projectPoints(calib_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(calib_corners[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    reprojection_error = mean_error / len(calib_points)
    print(f"\n重投影误差: {reprojection_error:.4f} 像素")
    print("好的标定结果，其误差应小于0.5。")

    # --- 将结果保存为JSON文件 ---
    print(f"\n保存标定数据到 '{args.output_file}'...")
    
    # 创建一个字典来存储所有数据
    calibration_data = {
        'camera_matrix': mtx,
        'distortion_coefficients': dist,
        'rotation_vectors': rvecs,
        'translation_vectors': tvecs,
        'reprojection_error': reprojection_error,
        'image_size': (w, h),
        'calibration_type': args.type
    }
    
    # 使用json.dump()和自定义的NumpyEncoder来写入文件
    try:
        with open(args.output_file, 'w') as f:
            json.dump(calibration_data, f, cls=NumpyEncoder, indent=4)
        print("完成。")
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")

    # (可选) 显示去畸变效果
    if args.show_undistorted and last_captured_image is not None:
        print("\n显示原始图像与去畸变图像的对比。按任意键退出。")
        h_img, w_img = last_captured_image.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w_img, h_img), 1, (w_img, h_img))
        dst = cv2.undistort(last_captured_image, mtx, dist, None, new_camera_mtx)
        x, y, w_roi, h_roi = roi
        dst = dst[y:y+h_roi, x:x+w_roi]
        comparison_image = np.hstack((last_captured_image, dst))
        cv2.imshow('Original vs. Undistorted', comparison_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()