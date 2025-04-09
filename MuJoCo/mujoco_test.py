import mujoco as mj
import mujoco.viewer
import numpy as np
from robot_descriptions import go2_mj_description
from robot_descriptions.loaders.mujoco import load_robot_description
import mediapy as media
import matplotlib.pyplot as plt
from robot_descriptions import go2_mj_description

from mujoco_base import MuJoCoBase
from mujoco.glfw import glfw

class mujoco_quad(MuJoCoBase):
    def __init__(self, xml):
        super().__init__(xml)

        # How long to run the sim
        self.simend = 30.0

        self.fsm_hip = None
        self.fsm_knee1 = None
        self.fsm_knee2 = None

    def reset(self):
        # Set camera configuration
        self.cam.azimuth = 120.89  # 89.608063
        self.cam.elevation = -15.81  # -11.588379
        self.cam.distance = 8.0  # 5.0
        self.cam.lookat = np.array([0.0, 0.0, 2.0])

        self.data.qpos[4] = 0.5
        self.data.ctrl[0] = self.data.qpos[4]

        self.model.opt.gravity[0] = 9.81 * np.sin(0.1)
        self.model.opt.gravity[2] = -9.81 * np.cos(0.1)

        # self.fsm_hip = FSM_LEG2_SWING
        # self.fsm_knee1 = FSM_KNEE1_STANCE
        # self.fsm_knee2 = FSM_KNEE2_STANCE

    def simulate(self):
        while not glfw.window_should_close(self.window):
            simstart = self.data.time

            # Run the step and control for a short period and update the sim at 60Hz
            while (self.data.time - simstart < 1.0/60.0):
                # Step simulation environment
                mj.mj_step(self.model, self.data)

                # Apply control
                # self.controller(self.model, self.data)

            if self.data.time >= self.simend:
                break

            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # Show joint frames
            self.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = 1

            # Update scene and render
            self.cam.lookat[0] = self.data.qpos[0]
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()


def main():
    
    # Load go2 xml path
    # xml = go2_mj_description.MJCF_PATH
    xml = '/home/jblevins32/.cache/robot_descriptions/mujoco_menagerie/agility_cassie/room_scene.xml'
    # xml = '/home/jblevins32/.cache/robot_descriptions/mujoco_menagerie/unitree_go2/scene.xml'
    sim = mujoco_quad(xml)
    sim.reset()
    sim.simulate()

if __name__ == "__main__":
    main()



# # Directly loading an instance of MjModel.
# model = mujoco.MjModel.from_xml_path(go2_mj_description.MJCF_PATH)
# data = mujoco.MjData(model)
# print("Number of contacts:", data.ncon)

# # Run simulation
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     # Enable wireframe rendering of the entire scene.
#     viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1

#     while viewer.is_running():
#         mujoco.mj_step(model, data)
#         viewer.sync()

# mujoco.mj_kinematics(model, data) # Load the kinematics
# print(data.geom_xpos)

# # Make renderer, render and show the pixels
# with mujoco.Renderer(model) as renderer:
#     mujoco.mj_forward(model, data) # This propogates the simulation with the data
#     renderer.update_scene(data)

#     plt.imshow(renderer.render())
#     plt.axis('off')
#     plt.show()

# duration = 3.8  # (seconds)
# framerate = 60  # (Hz)

# # Simulate and display video.
# frames = []
# mujoco.mj_resetData(model, data)  # Reset state and time.
# with mujoco.Renderer(model) as renderer:
#   while data.time < duration:
#     mujoco.mj_step(model, data)
#     if len(frames) < data.time * framerate:
#       renderer.update_scene(data)
#       pixels = renderer.render()
#       frames.append(pixels)

# media.show_video(frames, fps=framerate)

# print(model)
# model = load_robot_description("go1_mj_description")
# data = mujoco.MjData(model)

# target_qpos = np.array([0.0] * model.nq)
# target_qpos[0] = 360  # Move first joint

# Simulation loop
# with mujoco.viewer.launch_passive(model, data) as viewer:
    # for _ in range(10000):
    #     # Apply control
    #     data.ctrl[:] = 0.1*target_qpos[:model.nu]

    #     # Step the simulation
    #     mujoco.mj_step(model, data)

    #     # Print joint positions
    #     print("Joint positions:", data.qpos[:model.nq])
        # viewer.sync()

    # viewer.close()
    # while viewer.is_running():
    #     mujoco.mj_step(model, data)
    #     viewer.sync()