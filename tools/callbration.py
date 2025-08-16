#!/usr/bin/env python3
"""
实时 RealSense 标定脚本（棋盘格 / ArUco）
python realsense_calib_live.py --calibration_type chessboard
python realsense_calib_live.py --calibration_type aruco
"""

import argparse
import time
import yaml
import cv2
import numpy as np
import pyrealsense2 as rs

# ----------------- 可修改的参数 -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--calibration_type", choices=["chessboard", "aruco"], default="chessboard")
parser.add_argument("--rows", type=int, default=9, help="棋盘格内部行角点数")
parser.add_argument("--cols", type=int, default=6, help="棋盘格内部列角点数")
parser.add_argument("--square_size", type=float, default=0.025, help="棋盘格单格物理尺寸(m)")
parser.add_argument("--aruco_dict", type=str, default="DICT_6X6_250",
                    help="ArUco 字典名，见 cv2.aruco.DICT_xxx")
parser.add_argument("--min_frames", type=int, default=15, help="最少需要的标定帧数")
parser.add_argument("--stab_thresh", type=float, default=2.0, help="像素级稳定阈值")
parser.add_argument("--stab_time", type=float, default=1.0, help="稳定判定时间(s)")
parser.add_argument("--save_yaml", type=str, default="realsense_calib.yml")
args = parser.parse_args()

# ----------------- RealSense 初始化 -----------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 跳过前 30 帧自动曝光稳定
for _ in range(30):
    pipeline.wait_for_frames()

# ----------------- 标定准备 -----------------
obj_points_all = []   # 三维点
img_points_all = []   # 二维点
frame_idx = 0

if args.calibration_type == "chessboard":
    pattern_size = (args.cols, args.rows)
    # 生成棋盘格角点的 3D 坐标 (z=0)
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_size
else:   # aruco
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.aruco_dict))
    aruco_params = cv2.aruco.DetectorParameters_create()
    # 这里假设每个 tag 边长 0.038 m，可改
    tag_size = 0.038
    # 单 tag 的 3D 坐标
    objp = np.array([
        [-tag_size/2, -tag_size/2, 0],
        [ tag_size/2, -tag_size/2, 0],
        [ tag_size/2,  tag_size/2, 0],
        [-tag_size/2,  tag_size/2, 0]
    ], dtype=np.float32)

# ----------------- 稳定判定缓存 -----------------
last_tvec = None
stab_start = None

# ----------------- 主循环 -----------------
print("按 q 退出，坐标轴稳定时按 s 保存当前帧")
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    img = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()

    ok = False
    axis = np.float32([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]])

    if args.calibration_type == "chessboard":
        ret, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv2.CALIB_CB_FAST_CHECK +
                                                 cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                             criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            ok = True
            # 求解 PnP
            _, rvec, tvec = cv2.solvePnP(objp, corners, np.eye(3), None)
            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, np.eye(3), None)
            imgpts = imgpts.astype(int).reshape(-1, 2)
            cv2.drawFrameAxes(vis, np.eye(3), None, rvec, tvec, 0.05, 3)

    else:   # aruco
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        if ids is not None:
            for i, corner in enumerate(corners):
                # 每个 tag 独立求解
                _, rvec, tvec = cv2.solvePnP(objp, corner[0], np.eye(3), None)
                imgpts, _ = cv2.projectPoints(axis, rvec, tvec, np.eye(3), None)
                imgpts = imgpts.astype(int).reshape(-1, 2)
                cv2.drawFrameAxes(vis, np.eye(3), None, rvec, tvec, 0.05, 3)
            ok = True
            # 取第一个 tag 做稳定判定
            _, rvec, tvec = cv2.solvePnP(objp, corners[0][0], np.eye(3), None)

    # 稳定判定逻辑
    if ok:
        tvec_norm = np.linalg.norm(tvec)
        if last_tvec is None:
            last_tvec = tvec.copy()
            stab_start = time.time()

        dist = np.linalg.norm(tvec - last_tvec)
        if dist < args.stab_thresh:
            if time.time() - stab_start > args.stab_time:
                auto_save = True
                stab_start = time.time()  # 重置计时器
            else:
                auto_save = False
        else:
            last_tvec = tvec.copy()
            stab_start = time.time()
            auto_save = False
    else:
        auto_save = False
        last_tvec = None

    # 显示进度
    cv2.putText(vis, f"Captured: {frame_idx}/{args.min_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Calibration", vis)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if (key == ord('s') or auto_save) and ok:
        filename = f"calib_{args.calibration_type}_{frame_idx:03d}.png"
        cv2.imwrite(filename, img)
        print(f"Saved {filename}")
        obj_points_all.append(objp)
        if args.calibration_type == "chessboard":
            img_points_all.append(corners.reshape(-1, 2))
        else:
            img_points_all.append(corners[0][0])  # 仅取第一个 tag 的 4 个角点
        frame_idx += 1

cv2.destroyAllWindows()
pipeline.stop()

# ----------------- 标定计算 -----------------
if frame_idx < args.min_frames:
    print(f"仅采集到 {frame_idx} 帧，不足 {args.min_frames}，无法标定")
    exit()

print("\n开始计算相机标定 …")
# 注意：RealSense 彩色相机像素坐标即 640×480，因此先假设无畸变先验
h, w = 480, 640
camera_matrix_init = np.array([[640, 0, 320],
                               [0, 480, 240],
                               [0, 0, 1]], dtype=np.float64)

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points_all, img_points_all, (w, h), camera_matrix_init, None,
    flags=cv2.CALIB_FIX_PRINCIPAL_POINT)

print("\n重投影误差 RMS =", ret)
print("相机矩阵 K =\n", K)
print("畸变系数 [k1 k2 p1 p2 k3] =", dist.ravel())

# ----------------- 保存结果 -----------------
data = {
    "image_width": int(w),
    "image_height": int(h),
    "camera_matrix": {"rows": 3, "cols": 3, "data": K.flatten().tolist()},
    "distortion_coefficients": {"rows": 1, "cols": 5, "data": dist.ravel().tolist()},
    "reprojection_error": float(ret)
}
with open(args.save_yaml, "w") as f:
    yaml.dump(data, f, default_flow_style=None)
print(f"\n标定结果已保存至 {args.save_yaml}")



