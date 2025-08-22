import argparse

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug")

    ########################## create a scene ##########################

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=10,
        ),
        # SPH 即 Smoothed Particle Hydrodynamics，光滑粒子流体动力学，常用于流体 / 颗粒体仿真
        sph_options=gs.options.SPHOptions(
            lower_bound=(0.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 2.4),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, -3.15, 2.42),
            camera_lookat=(0.5, 0.0, 0.5),
            camera_fov=40,
            max_FPS=60,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    frictionless_rigid = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    cup = scene.add_entity(morph=gs.morphs.Mesh(
                                file="./cup.glb",
                                pos=(0.24623267, -0.04144618, 0.03),
                                euler=(0.0, 0.0, 0.0),
                                scale=0.06,
                                ),
                                material=gs.materials.Rigid(
                                            rho=750,
                                            needs_coup=True, 
                                            coup_friction=0.3,
                                            coup_softness=0.01,
                                        ),
                                surface=gs.surfaces.Default(
                                        # color = (0.8, 0.8, 0.8),
                                            vis_mode = 'collision',
                                    ),
                                        # visualize_contact=True
                                )

    water = scene.add_entity(
                            material=gs.materials.SPH.Liquid(),
                            morph=gs.morphs.Box(
                                        pos=(0.1, 0.1, 0.1),
                                        size=(0.05, 0.05, 0.05),
                                    ),
                            surface=gs.surfaces.Default(
                                color=(0.2, 0.6, 1.0, 1.0),
                                # vis_mode="particle",
                            ),
                        )

    ########################## build ##########################
    scene.build()

    for i in range(5000):
        scene.step()


if __name__ == "__main__":
    main()