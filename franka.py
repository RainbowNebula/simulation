import argparse

import numpy as np

import genesis as gs

from genesis.ext.pyrender.camera import Camera
import threading

np.set_printoptions(linewidth=200)

import asyncio
import threading
import time
import cv2



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )

    cam_0 = scene.add_camera(
        res=(1280, 960),
        pos=(0, 0, 0.3),
        lookat=(1.0, 0.0, 0.0),
        fov=60,
        GUI=False,

    ) 

    print(scene._visualizer._cameras)
    # cam_1 = scene.add_camera()

    # 创建一个长方体木块
    # wood_block = scene.add_entity(
    #     morph=gs.morphs.Box(
    #         pos=(0.5, 0.0, 0.05),  # 长方体的位置
    #         size=(0.15, 0.4, 0.1),  # 长方体的尺寸（长、宽、高）
    #     ),
    #     material=gs.materials.Rigid(
    #         rho=1000,  # 密度
    #         friction=0.5,  # 摩擦系数
    #     ),
    # )

    scene.add_entity(
        morph=gs.morphs.MJCF(
            file="/home/haichao/workspace/real2sim/simulation/assets/table2.xml",
            # scale=0.09,
            pos=(0.45, 0.0, 0.0),
            euler=(0, 0, 0),
            collision=True,
            convexify=False
        ),
    )

    franka = scene.add_entity(
        morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml",
                             pos=(0.0, 0.0, 0.85)),
    )

    tomato = scene.add_entity(
        morph=gs.morphs.Mesh(file="/home/haichao/workspace/real2sim/data/2025-06-27/3d_assets/tomato_0.stl",
                             pos=(0.2, 0.0, 0.85)),
    )
    
    tomato = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="/home/haichao/workspace/real2sim/data/2025-06-27/3d_assets/tomato_0.stl",
            pos=(0.4, 0.0, 1.85),
        )
    )   
    ########################## build ##########################
    scene.build()

    # 启动线程
    # thread = threading.Thread(target=generate_images, args=(cam_0,))
    # # 设置为守护线程，主线程退出后，结束其他线程任务
    # thread.daemon = True
    # thread.start()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    # Optional: set control gains
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    end_effector = franka.get_link("hand")

    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.25 + 0.85]),
        quat=np.array([0, 1, 0, 0]),
    )
    qpos[-2:] = 0.04
    path = franka.plan_path(qpos)
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        cam_0.render(rgb=True, depth=True)
        scene.step()

    while True:
        scene.step()

    # move to pre-grasp pose
    # qpos = franka.inverse_kinematics(
    #     link=end_effector,
    #     pos=np.array([0.65, 0.0, 0.25]),
    #     quat=np.array([0, 1, 0, 0]),
    # )
    # qpos[-2:] = 0.04
    # path = franka.plan_path(qpos)
    # for waypoint in path:
    #     franka.control_dofs_position(waypoint)
    #     cam_0.render(rgb=True, depth=True)
    #     scene.step()

    # # reach
    # qpos = franka.inverse_kinematics(
    #     link=end_effector,
    #     pos=np.array([0.65, 0.0, 0.2]),
    #     quat=np.array([0, 1, 0, 0]),
    # )
    # franka.control_dofs_position(qpos[:-2], motors_dof)
    # for i in range(100):
    #     cam_0.render(rgb=True, depth=True)
    #     scene.step()

    # # grasp
    # franka.control_dofs_position(qpos[:-2], motors_dof)
    # franka.control_dofs_position(np.array([0, 0]), fingers_dof)  # you can use position control
    # for i in range(10):
    #     cam_0.render()
    #     scene.step()

    # # lift
    # qpos = franka.inverse_kinematics(
    #     link=end_effector,
    #     pos=np.array([0.65, 0.0, 0.3]),
    #     quat=np.array([0, 1, 0, 0]),
    # )

    # franka.control_dofs_position(qpos[:-2], motors_dof)
    # franka.control_dofs_force(np.array([-20, -20]), fingers_dof)  # can also use force control
    # for i in range(1000):
    #     cam_0.render()
    #     scene.step()


if __name__ == "__main__":
    main()
