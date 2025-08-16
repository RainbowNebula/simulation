import cv2
import numpy as np
import os
import argparse
import json
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# --- ArUco字典定义 ---
CHARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

def create_board_object_points(args):
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

class CameraCalibrator:
    def __init__(self, args):
        self.args = args
        self.camera_matrix = None  # 内参矩阵
        self.dist_coeffs = None    # 畸变系数
        self.rvecs = None          # 旋转向量（外参）
        self.tvecs = None          # 平移向量（外参）
        self.rotation_matrices = []# 旋转矩阵（外参）
        self.euler_angles = []     # 欧拉角（外参）
        self.reprojection_error = None
        self.image_size = None
        self.calibration_type = args.type
        self.obj_points = []
        self.img_points = []
        self.images = []
        self.image_paths = []
        
        if self.calibration_type == 'chessboard':
            self.pattern_size = (self.args.grid_width - 1, self.args.grid_height - 1)
            self.objp = np.zeros((np.prod(self.pattern_size), 3), np.float32)
            self.objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
            self.objp *= self.args.square_size
        elif self.calibration_type == 'charuco':
            self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            self.board = cv2.aruco.CharucoBoard(
                (self.args.charuco_squares_x, self.args.charuco_squares_y),
                self.args.charuco_square_length,
                self.args.charuco_marker_length,
                self.dictionary
            )
    
    def load_images(self, image_dir):
        print(f"正在加载目录中的图像: {image_dir}")
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(image_dir, ext)))
        
        if not image_files:
            raise ValueError(f"在目录 {image_dir} 中未找到图像文件")
        
        print(f"找到 {len(image_files)} 张图像")
        return image_files
    
    def detect_corners(self, image_files, args):
        print(f"开始检测 {self.calibration_type} 标定板角点...")
        valid_count = 0
        objp, board, checkerboard_size = create_board_object_points(args)
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取图像 {img_path}，跳过")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            if self.calibration_type == 'chessboard':
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | \
                      cv2.CALIB_CB_FAST_CHECK | \
                      cv2.CALIB_CB_NORMALIZE_IMAGE | \
                      cv2.CALIB_CB_FILTER_QUADS
                ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None, flags=flags)
                
                if ret:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    cv2.drawChessboardCorners(img, self.pattern_size, corners, ret)
                    self.obj_points.append(self.objp)
                    self.img_points.append(corners)
                    self.images.append(img)
                    self.image_paths.append(img_path)
                    valid_count += 1
                    print(f"已检测到角点: {img_path} ({valid_count}/{len(image_files)})")
            
            elif self.calibration_type == 'charuco':
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
                
                if ids is not None and len(ids) > 0:
                    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        corners, ids, gray, self.board
                    )
                    
                    if ret and len(charuco_ids) > 3:
                        self.obj_points.append(self.board.getChessboardCorners()[charuco_ids])
                        self.img_points.append(charuco_corners)
                        self.images.append(img)
                        self.image_paths.append(img_path)
                        valid_count += 1
                        print(f"已检测到角点: {img_path} ({valid_count}/{len(image_files)})")
        
        if valid_count < 3:
            raise ValueError(f"仅在 {valid_count} 张图像中检测到标定板角点，至少需要3张")
        
        self.image_size = gray.shape[::-1]
        print(f"成功在 {valid_count} 张图像中检测到标定板角点")
        return valid_count
    
    def calibrate_camera(self):
        print("\n开始相机标定...")
        
        if self.calibration_type == 'chessboard':
            ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
                self.obj_points, self.img_points, self.image_size, None, None
            )
        elif self.calibration_type == 'charuco':
            ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.aruco.calibrateCameraCharuco(
                self.img_points, [ids for ids in self.obj_points], self.board, self.image_size, None, None
            )
        
        if not ret:
            raise RuntimeError("相机标定失败")
        
        # 计算外参（旋转矩阵、欧拉角）
        self._compute_extrinsics()
        
        # 计算重投影误差
        self.reprojection_error = self._calculate_reprojection_error()
        
        # 强制打印所有参数
        self._print_all_parameters()
        
        return ret
    
    def _compute_extrinsics(self):
        for rvec in self.rvecs:
            rmat, _ = cv2.Rodrigues(rvec)
            self.rotation_matrices.append(rmat)
            euler = R.from_matrix(rmat).as_euler('zyx', degrees=True)
            self.euler_angles.append(euler)
    
    def _print_all_parameters(self):
        """强制打印内参、畸变参数、外参"""
        print("\n" + "="*60)
        print("|                     标定参数汇总                     |")
        print("="*60)
        
        # 1. 打印内参矩阵
        print("\n【1. 内参矩阵 (3x3)】")
        print("  内参矩阵描述相机的光学特性（焦距、主点）：")
        for row in self.camera_matrix:
            print("  " + ", ".join([f"{x:.6f}" for x in row]))
        print("  其中：")
        print(f"  fx = {self.camera_matrix[0,0]:.6f} (x方向焦距)")
        print(f"  fy = {self.camera_matrix[1,1]:.6f} (y方向焦距)")
        print(f"  cx = {self.camera_matrix[0,2]:.6f} (主点x坐标)")
        print(f"  cy = {self.camera_matrix[1,2]:.6f} (主点y坐标)")
        
        # 2. 打印畸变系数
        print("\n【2. 畸变系数】")
        print("  畸变系数描述镜头的非线性畸变（k1, k2, p1, p2, k3）：")
        print(f"  {self.dist_coeffs.flatten()}")
        
        # 3. 打印外参（每帧）
        print("\n【3. 外参（每帧的位姿）】")
        print("  外参描述相机相对于世界坐标系的位置和朝向：")
        for i in range(len(self.rvecs)):
            print(f"\n  第 {i+1} 帧：")
            print("  旋转矩阵 (3x3)：")
            for row in self.rotation_matrices[i]:
                print("    " + ", ".join([f"{x:.6f}" for x in row]))
            print(f"  平移向量 (米)：{self.tvecs[i].flatten()}")
            print(f"  欧拉角 (度, yaw-pitch-roll)：{self.euler_angles[i]}")
        
        # 4. 打印重投影误差
        print("\n【4. 重投影误差】")
        print(f"  平均重投影误差：{self.reprojection_error:.6f} 像素")
        print("  （误差越小，标定越精确，通常应 < 0.5 像素）")
        print("\n" + "="*60)
    
    def _calculate_reprojection_error(self):
        mean_error = 0
        for i in range(len(self.obj_points)):
            img_points2, _ = cv2.projectPoints(self.obj_points[i], self.rvecs[i], self.tvecs[i], 
                                            self.camera_matrix, self.dist_coeffs)
            error = cv2.norm(self.img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            mean_error += error
        return mean_error / len(self.obj_points)
    
    def save_results(self, output_file):
        print(f"\n保存标定结果到 {output_file}")
        
        extrinsics = []
        for i in range(len(self.rvecs)):
            extrinsics.append({
                "frame": i+1,
                "image_path": self.image_paths[i],
                "rotation_vector": self.rvecs[i].flatten().tolist(),
                "rotation_matrix": self.rotation_matrices[i].tolist(),
                "translation_vector": self.tvecs[i].flatten().tolist(),
                "euler_angles": self.euler_angles[i].tolist()
            })
        
        calibration_data = {
            'calibration_type': self.calibration_type,
            'image_size': self.image_size,
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coefficients': self.dist_coeffs.tolist(),
            'reprojection_error': self.reprojection_error,
            'extrinsics': extrinsics,
            'calibration_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        with open(output_file, 'w') as f:
            json.dump(calibration_data, f, indent=4)
    
    def visualize_results(self, output_dir=None):
        if not self.images:
            print("没有图像可用于可视化")
            return
        
        print("\n可视化标定结果...")
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for i, img in enumerate(self.images):
            img_name = os.path.basename(self.image_paths[i])
            cv2.imshow(f"角点检测结果 ({i+1}/{len(self.images)})", img)
            cv2.waitKey(500)
            if output_dir:
                cv2.imwrite(os.path.join(output_dir, f"corners_{img_name}"), img)
        
        cv2.destroyAllWindows()
        
        if output_dir:
            self._plot_reprojection_errors(output_dir)
    
    def _plot_reprojection_errors(self, output_dir):
        errors = []
        for i in range(len(self.obj_points)):
            img_points2, _ = cv2.projectPoints(self.obj_points[i], self.rvecs[i], self.tvecs[i], 
                                            self.camera_matrix, self.dist_coeffs)
            error = cv2.norm(self.img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            errors.append(error)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(errors) + 1), errors)
        plt.axhline(y=self.reprojection_error, color='r', linestyle='--', label=f'平均误差: {self.reprojection_error:.4f}')
        plt.xlabel('图像编号')
        plt.ylabel('重投影误差 (像素)')
        plt.title('每张图像的重投影误差')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'reprojection_errors.png'))
        print(f"已保存重投影误差图: {os.path.join(output_dir, 'reprojection_errors.png')}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='基于文件夹中的图片进行相机标定（强制打印内外参）')
    
    parser.add_argument('--image_dir', type=str, required=False, default="images/20250715_205537", help='包含标定图片的文件夹路径')
    parser.add_argument('--type', type=str, default='chessboard', choices=['chessboard', 'charuco'], 
                        help='标定板类型 (默认: chessboard)')
    parser.add_argument('--output_file', type=str, default='camera_calibration.json', 
                        help='保存标定结果的JSON文件路径')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='保存可视化结果的文件夹路径')
    
    # 棋盘格参数
    parser.add_argument('--grid_width', type=int, default=14, help="棋盘格横向格子总数")
    parser.add_argument('--grid_height', type=int, default=9, help="棋盘格纵向格子总数")
    parser.add_argument('--square_size', type=float, default=0.02, help="棋盘格方块边长（米）")
    
    # ChArUco板参数
    parser.add_argument('--charuco_squares_x', type=int, default=5, help='ChArUco板横向方块数量')
    parser.add_argument('--charuco_squares_y', type=int, default=7, help='ChArUco板纵向方块数量')
    parser.add_argument('--charuco_square_length', type=float, default=0.04, help='ChArUco格子边长（米）')
    parser.add_argument('--charuco_marker_length', type=float, default=0.03, help='ChArUco标记边长（米）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_dir):
        print(f"错误: 指定的图像目录不存在: {args.image_dir}")
        return
    
    try:
        calibrator = CameraCalibrator(args)
        image_files = calibrator.load_images(args.image_dir)
        valid_count = calibrator.detect_corners(image_files, args)
        calibrator.calibrate_camera()
        calibrator.save_results(args.output_file)
        
        if args.output_dir:
            calibrator.visualize_results(args.output_dir)
        
        print("\n相机标定完成! 所有参数已打印并保存至JSON文件")
        
    except Exception as e:
        print(f"标定过程中发生错误: {e}")

if __name__ == "__main__":
    main()