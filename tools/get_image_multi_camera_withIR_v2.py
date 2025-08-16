import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from pynput import keyboard
import pickle
import json



# 配置参数统一控制
CONFIG = {
    "width": 1280,
    "height": 720,
    "fps": 30,
    "format_bgr": rs.format.bgr8,
    "format_z": rs.format.z16,
    "format_y": rs.format.y8,
}


class RecordingState:
    def __init__(self):
        self.recording = False
        self.rgb_writers = {}
        self.depth_writers = {}
        self.ir_left_writers = {}
        self.ir_right_writers = {}

        self.depth_pkl_files = {}
        self.depth_data_lists = {}
        self.ir_left_pkl_files = {}
        self.ir_left_data_lists = {}
        self.ir_right_pkl_files = {}
        self.ir_right_data_lists = {}
        self.start_time = None  


def configure_cameras():
    ctx = rs.context()
    devices = ctx.query_devices()

    pipelines = []
    configs = []
    serial_numbers = []
    stereo_baselines = {}  # 存储每个相机的基线

    for dev in devices:
        serial_number = dev.get_info(rs.camera_info.serial_number)
        serial_numbers.append(serial_number)

        advnc_mode = rs.rs400_advanced_mode(dev)
        if not advnc_mode.is_enabled():
            print(f"Enabling advanced mode for camera {serial_number}...")
            advnc_mode.toggle_advanced_mode(True)
            time.sleep(5)
            advnc_mode = rs.rs400_advanced_mode(dev)

        # 读取高级模式配置文件（JSON格式）
        config_json = advnc_mode.serialize_json()
        config_dict = json.loads(config_json)

        # 提取基线参数（不同型号字段可能不同，需确认）
        # D435i/D455通常使用"StereoBaseline"字段，单位：米

        # if "stero" in config_dict:
        #     stereo_baseline = config_dict["StereoBaseline"]
        #     stereo_baselines[serial_number] = stereo_baseline
        #     print(f"Camera {serial_number} 基线距离：{stereo_baseline:.6f} 米")
        # else:
        #     print(f"警告：相机 {serial_number} 配置中未找到StereoBaseline字段")
        #     stereo_baselines[serial_number] = None

        ae = rs.STAEControl()
        ae.meanIntensitySetPoint = 600
        advnc_mode.set_ae_control(ae)

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, CONFIG["width"], CONFIG["height"], CONFIG["format_bgr"], CONFIG["fps"])
        config.enable_stream(rs.stream.depth, CONFIG["width"], CONFIG["height"], CONFIG["format_z"], CONFIG["fps"])
        config.enable_stream(rs.stream.infrared, 1, CONFIG["width"], CONFIG["height"], CONFIG["format_y"], CONFIG["fps"])
        config.enable_stream(rs.stream.infrared, 2, CONFIG["width"], CONFIG["height"], CONFIG["format_y"], CONFIG["fps"])

        for sensor in dev.query_sensors():
            if sensor.is_depth_sensor():
                sensor.set_option(rs.option.emitter_enabled, 0.0) #控制激光强度的，最大300

            if sensor.supports(rs.option.confidence_threshold):
                sensor.set_option(rs.option.confidence_threshold, 3)
            else:
                print(f"警告：设备 {dev.get_info(rs.camera_info.serial_number)} 的深度传感器不支持置信度阈值选项")

            
            if sensor.get_info(rs.camera_info.name) == 'Stereo Module':
                # 读取基线（m）
                baseline = sensor.get_option(rs.option.stereo_baseline)
                stereo_baselines[serial_number] = baseline
                print(f"设备 {dev.get_info(rs.camera_info.serial_number)} 的Stereo baseline: {baseline:.3f} mm")
                break

        configs.append(config)

    return pipelines, configs, serial_numbers, stereo_baselines


def setup_directories(serial_numbers):
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    base_dir = os.path.join(script_dir, "data")
    os.makedirs(base_dir, exist_ok=True)

    now = time.localtime()
    time_dir = time.strftime("%Y%m%d-%H%M", now)

    # state.start_time = time_dir # 设置录制开始时间

    directories = {}
    for serial in serial_numbers:
        cam_base = os.path.join(base_dir,time_dir)
        images_dir = os.path.join(cam_base, "images")
        video_dir = os.path.join(cam_base, "video")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        directories[serial] = {"images": images_dir, "video": video_dir}
    return directories

def start_recording(directories, serial_numbers):
    timestamp = int(time.time())

    rgb_writers = {}
    depth_writers = {}
    ir_left_writers = {}
    ir_right_writers = {}

    depth_pkl_files = {}
    depth_data_lists = {}
    ir_left_pkl_files = {}
    ir_left_data_lists = {}
    ir_right_pkl_files = {}
    ir_right_data_lists = {}

    for serial in serial_numbers:

        video_dir = directories[serial]["video"]
        video_size = (CONFIG["width"], CONFIG["height"])

        rgb_writer = cv2.VideoWriter(
            f"{video_dir}/rgb_video_{serial}_{timestamp}.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'), CONFIG["fps"], video_size
        )

        depth_writer = cv2.VideoWriter(
            f"{video_dir}/depth_video_{serial}_{timestamp}.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'), CONFIG["fps"], video_size, isColor=False
        )

        ir_left_writer = cv2.VideoWriter(
            f"{video_dir}/ir_left_video_{serial}_{timestamp}.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'), CONFIG["fps"], video_size, isColor=False
        )

        ir_right_writer = cv2.VideoWriter(
            f"{video_dir}/ir_right_video_{serial}_{timestamp}.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'), CONFIG["fps"], video_size, isColor=False
        )

        rgb_writers[serial] = rgb_writer
        depth_writers[serial] = depth_writer
        ir_left_writers[serial] = ir_left_writer
        ir_right_writers[serial] = ir_right_writer

        depth_pkl_file = f"{video_dir}/depth_raw_{serial}_{timestamp}.pkl"
        ir_left_pkl_file = f"{video_dir}/ir_left_raw_{serial}_{timestamp}.pkl"
        ir_right_pkl_file = f"{video_dir}/ir_right_raw_{serial}_{timestamp}.pkl"

        depth_data_lists[serial] = []
        ir_left_data_lists[serial] = []
        ir_right_data_lists[serial] = []

        depth_pkl_files[serial] = depth_pkl_file
        ir_left_pkl_files[serial] = ir_left_pkl_file
        ir_right_pkl_files[serial] = ir_right_pkl_file

    return (
        rgb_writers, depth_writers, ir_left_writers, ir_right_writers,
        depth_pkl_files, depth_data_lists,
        ir_left_pkl_files, ir_left_data_lists,
        ir_right_pkl_files, ir_right_data_lists
    )


def stop_recording(state: RecordingState):
    for serial in state.rgb_writers.keys():
        state.rgb_writers[serial].release()
        state.depth_writers[serial].release()
        state.ir_left_writers[serial].release()
        state.ir_right_writers[serial].release()

        with open(state.depth_pkl_files[serial], 'wb') as f:
            pickle.dump(state.depth_data_lists[serial], f)
        with open(state.ir_left_pkl_files[serial], 'wb') as f:
            pickle.dump(state.ir_left_data_lists[serial], f)
        with open(state.ir_right_pkl_files[serial], 'wb') as f:
            pickle.dump(state.ir_right_data_lists[serial], f)

    print("Stopped recording and saved raw data")


def main():
    pipelines, configs, serial_numbers, stereo_baselines = configure_cameras()
    directories = setup_directories(serial_numbers)
     
    # https://support.intelrealsense.com/hc/en-us/community/posts/1500000444841-How-to-align-infrared-resolution-the-same-as-RGB-resolution
    # align_to = rs.stream.depth
    align_to = rs.stream.color
    colorizer = rs.colorizer()

    profiles = [pipeline.start(config) for pipeline, config in zip(pipelines, configs)]

    intrinsics = {}
    for serial, profile in zip(serial_numbers, profiles):
        # 获取 RGB 内参
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intr = color_stream.get_intrinsics()

        # 获取 IR 内参
        ir_left_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        ir_left_intr = ir_left_profile.get_intrinsics()
        ir_right_profile = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        ir_right_intr = ir_right_profile.get_intrinsics()

        # 获取左右 IR 外参（左 -> 右）
        extrin_ir_lr = ir_left_profile.get_extrinsics_to(ir_right_profile)

        # 新增：获取 RGB 到 左 IR 和 右 IR 的外参（RGB -> Left IR 和 RGB -> Right IR）
        extrin_color_to_ir_left = color_stream.get_extrinsics_to(ir_left_profile)
        extrin_color_to_ir_right = color_stream.get_extrinsics_to(ir_right_profile)


        K_matrix = [
            [color_intr.fx, 0, color_intr.ppx],
            [0, color_intr.fy, color_intr.ppy],
            [0, 0, 1]
        ]

        ir_left_K = [
            [ir_left_intr.fx, 0, ir_left_intr.ppx],
            [0, ir_left_intr.fy, ir_left_intr.ppy],
            [0, 0, 1]
        ]

        ir_right_K = [
            [ir_right_intr.fx, 0, ir_right_intr.ppx],
            [0, ir_right_intr.fy, ir_right_intr.ppy],
            [0, 0, 1]
        ]

        intrinsics[serial] = {
            "K_matrix": K_matrix,
            "width": color_intr.width,
            "height": color_intr.height,
            "model": str(color_intr.model),
            "coeffs": color_intr.coeffs,

            "ir_left_K": ir_left_K,
            "ir_right_K": ir_right_K,
            "ir_left_width": ir_left_intr.width,
            "ir_left_height": ir_left_intr.height,
            "ir_right_width": ir_right_intr.width,
            "ir_right_height": ir_right_intr.height,
            "ir_model": str(ir_left_intr.model),
            "ir_left_coeffs": ir_left_intr.coeffs,
            "ir_right_coeffs": ir_right_intr.coeffs,

            "extrin_ir_lr": {
                "rotation": list(extrin_ir_lr.rotation),
                "translation": list(extrin_ir_lr.translation)
            },
            "extrin_color_to_ir_right": {
                "rotation": list(extrin_color_to_ir_right.rotation),
                "translation": list(extrin_color_to_ir_right.translation)
            },
            "extrin_color_to_ir_left": {
                "rotation": list(extrin_color_to_ir_left.rotation),
                "translation": list(extrin_color_to_ir_left.translation)
            }
        }

        if serial in stereo_baselines:
            intrinsics[serial]["stereo_baseline"] = stereo_baselines[serial]

    for serial, params in intrinsics.items():
        with open(os.path.join(directories[serial]["video"], "intrinsics.pkl"), "wb") as f:
            pickle.dump(params, f)
        print(f"Saved intrinsics for camera {serial}")

        with open(os.path.join(directories[serial]["video"], "intrinsics.json"), "w") as f:
            json.dump(params, f, ensure_ascii=False, indent=4)
        print(f"Saved intrinsics (json) for camera {serial}")

    hole_fill_filter = rs.hole_filling_filter()
    frame_count = 0

    # 初始化录制状态对象
    state = RecordingState()

    def on_press(key):
        try:
            if key.char.lower() == 'a':
                for serial, pipeline in zip(serial_numbers, pipelines):
                    frames = pipeline.wait_for_frames()
                    aligned_frames = rs.align(align_to).process(frames)

                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    ir_left_frame = aligned_frames.get_infrared_frame(1)
                    ir_right_frame = aligned_frames.get_infrared_frame(2)

                    if not color_frame or not depth_frame:
                        continue

                    # depth_frame = hole_fill_filter.process(depth_frame)
                    depth_frame.keep()
                    color_frame.keep()
                    ir_left_frame.keep()
                    ir_right_frame.keep()


                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())

                    timestamp = int(time.time())

                    cv2.imwrite(f"{directories[serial]['images']}/rgb_{serial}_{timestamp}.png", color_image)
                    cv2.imwrite(f"{directories[serial]['images']}/depth_{serial}_{timestamp}.png", depth_image)

                    if ir_left_frame:
                        ir_left_image = np.asanyarray(ir_left_frame.get_data())
                        cv2.imwrite(f"{directories[serial]['images']}/ir_left_{serial}_{timestamp}.png", ir_left_image)

                    if ir_right_frame:
                        ir_right_image = np.asanyarray(ir_right_frame.get_data())
                        cv2.imwrite(f"{directories[serial]['images']}/ir_right_{serial}_{timestamp}.png", ir_right_image)

                    print(f"Saved image for camera {serial} at timestamp {timestamp}")

            elif key.char.lower() == 'r' and not state.recording:
                (state.rgb_writers, state.depth_writers, state.ir_left_writers, state.ir_right_writers,
                state.depth_pkl_files, state.depth_data_lists,
                state.ir_left_pkl_files, state.ir_left_data_lists,
                state.ir_right_pkl_files, state.ir_right_data_lists) = start_recording(directories, serial_numbers)
                state.recording = True
                print("Started recording...")

            elif key.char.lower() == 's' and state.recording:
                print("Stopping current recording...")
                stop_recording(state)
                state.recording = False
                recording_start_time = None

            elif key.char == 'q':
                return False

        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        while True:
            all_frames = [pipeline.wait_for_frames() for pipeline in pipelines]
            all_aligned_frames = [rs.align(align_to).process(frames) for frames in all_frames]

            for serial, aligned_frames in zip(serial_numbers, all_aligned_frames):
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                ir_left_frame = aligned_frames.get_infrared_frame(1)
                ir_right_frame = aligned_frames.get_infrared_frame(2)

                if not color_frame or not depth_frame:
                    continue

                # depth_frame = hole_fill_filter.process(depth_frame)
                # color_frame = hole_fill_filter.process(color_frame)
                depth_frame.keep()
                color_frame.keep()

                # 不录视频不要keep
                # ir_left_frame.keep()
                # ir_right_frame.keep()

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

                if ir_left_frame:
                    ir_left_image = np.asanyarray(ir_left_frame.get_data())
                else:
                    ir_left_image = np.zeros((CONFIG["height"], CONFIG["width"]), dtype=np.uint8)

                if ir_right_frame:
                    ir_right_image = np.asanyarray(ir_right_frame.get_data())
                else:
                    ir_right_image = np.zeros((CONFIG["height"], CONFIG["width"]), dtype=np.uint8)


                # combined = np.hstack((
                #     cv2.cvtColor(ir_left_image, cv2.COLOR_GRAY2BGR),
                #     ,
                #     color_image,
                #     depth_colormap
                # ))
                # cv2.imshow(f'Camera {serial}', combined)

                # 构建 2x2 网格布局
                top_row = np.hstack((color_image, depth_colormap))
                bottom_row = np.hstack((cv2.cvtColor(ir_left_image, cv2.COLOR_GRAY2BGR), cv2.cvtColor(ir_right_image, cv2.COLOR_GRAY2BGR)))
                grid_view = np.vstack((top_row, bottom_row))

                # 缩放可选（如果图像太大）
                grid_view = cv2.resize(grid_view, None, fx=0.5, fy=0.5)

                # 显示
                cv2.imshow(f"Camera {serial}", grid_view)
                

                if state.recording:
                    state.rgb_writers[serial].write(color_image)
                    state.depth_writers[serial].write(depth_image)
                    state.ir_left_writers[serial].write(ir_left_image)
                    state.ir_right_writers[serial].write(ir_right_image)

                    state.depth_data_lists[serial].append(depth_image)
                    state.ir_left_data_lists[serial].append(ir_left_image)
                    state.ir_right_data_lists[serial].append(ir_right_image)

            if cv2.waitKey(1) == ord('q') or not listener.running:
                break

    finally:
        for pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()
        if state.recording:
            stop_recording(state)
        print("Program terminated.")


if __name__ == "__main__":
    main()