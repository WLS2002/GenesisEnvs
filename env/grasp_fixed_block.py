import numpy as np
import genesis as gs
import torch


class GraspFixedBlockEnv:
    def __init__(self, vis, device, num_envs=1, max_episode_length=1000):
        self.device = device
        self.action_space = 8
        self.state_dim = 6
        self.max_episode_length = max_episode_length
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=0.01,
            ),
            rigid_options=gs.options.RigidOptions(
                box_box_detection=True,
            ),
            show_viewer=vis,
        )
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml"),
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04),  # block
                pos=(0.65, 0.0, 0.02),
            )
        )
        self.num_envs = num_envs
        self.scene.build(n_envs=self.num_envs)
        self.envs_idx = np.arange(self.num_envs)
        self.build_env()

    def build_env(self):
        self.motors_dof = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7, 9).to(self.device)
        franka_pos = torch.tensor(
            [-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04]
        ).to(self.device)
        franka_pos = franka_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self.franka.set_qpos(franka_pos, envs_idx=self.envs_idx)
        self.scene.step()

        self.end_effector = self.franka.get_link("hand")
        ## here self.pos and self.quat is target for the end effector; not the cube. cube position is set in reset()
        pos = torch.tensor([0.65, 0.0, 0.135], dtype=torch.float32, device=self.device)
        self.pos = pos.unsqueeze(0).repeat(self.num_envs, 1)
        quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=self.device)
        self.quat = quat.unsqueeze(0).repeat(self.num_envs, 1)
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=self.pos,
            quat=self.quat,
        )
        self.franka.control_dofs_position(
            self.qpos[:, :-2], self.motors_dof, self.envs_idx
        )

    def reset_idx(self, envs_idx):
        """
        重置指定索引（envs_idx）的环境，并返回重置后的观测状态。
        参数:
            envs_idx: 一个包含需要重置的环境索引的数组或张量
        返回:
            对应环境的状态观测
        """

        self.envs_episode_len[envs_idx] = 0

        # 1. 重置机器人（Franka）的初始关节配置
        franka_pos = torch.tensor(
            [-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04],
            dtype=torch.float32,
            device=self.device,
        )
        self.franka.set_qpos(
            franka_pos.unsqueeze(0).repeat(len(envs_idx), 1), envs_idx=envs_idx
        )
        # self.scene.step()

        # 2. 构造目标位置和姿态张量
        self.pos[envs_idx] = (
            torch.tensor([0.65, 0.0, 0.135], dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .repeat(len(envs_idx), 1)
        )

        self.quat[envs_idx] = (
            torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .repeat(len(envs_idx), 1)
        )

        # self.qpos = self.franka.inverse_kinematics(
        #     link=self.end_effector,
        #     pos=self.pos,
        #     quat=self.quat,
        # )
        # self.franka.control_dofs_position(
        #     self.qpos[:, :-2], self.motors_dof, self.envs_idx
        # )

        # 4. 重置 cube（抓取目标块）的位置（只更新部分环境）
        cube_pos = torch.tensor(
            [0.65, 0.0, 0.02], dtype=torch.float32, device=self.device
        )
        self.cube.set_pos(
            cube_pos.unsqueeze(0).repeat(len(envs_idx), 1), envs_idx=envs_idx
        )

        # # 5. 获取重置后的观测状态（cube 位置与夹爪中间位置的均值）
        # obs_cube = self.cube.get_pos()[envs_idx]
        # left_finger = self.franka.get_link("left_finger").get_pos()[envs_idx]
        # right_finger = self.franka.get_link("right_finger").get_pos()[envs_idx]
        # obs_gripper = (left_finger + right_finger) / 2
        # state = torch.cat([obs_cube, obs_gripper], dim=1)
        # return state

    def reset(self):
        self.build_env()
        self.envs_episode_len = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        # fixed cube position
        cube_pos = np.array([0.65, 0.0, 0.02])
        cube_pos = np.repeat(cube_pos[np.newaxis], self.num_envs, axis=0)
        self.cube.set_pos(cube_pos, envs_idx=self.envs_idx)

        obs1 = self.cube.get_pos()
        obs2 = (
            self.franka.get_link("left_finger").get_pos()
            + self.franka.get_link("right_finger").get_pos()
        ) / 2
        state = torch.concat([obs1, obs2], dim=1)
        
        return state

    def step(self, actions):
        action_mask_0 = actions == 0  # Open gripper
        action_mask_1 = actions == 1  # Close gripper
        action_mask_2 = actions == 2  # Lift gripper
        action_mask_3 = actions == 3  # Lower gripper
        action_mask_4 = actions == 4  # Move left
        action_mask_5 = actions == 5  # Move right
        action_mask_6 = actions == 6  # Move forward
        action_mask_7 = actions == 7  # Move backward

        finger_pos = torch.full(
            (self.num_envs, 2), 0.04, dtype=torch.float32, device=self.device
        )
        finger_pos[action_mask_1] = 0
        finger_pos[action_mask_2] = 0

        pos = self.pos.clone()
        pos[action_mask_2, 2] = 0.4
        pos[action_mask_3, 2] = 0
        pos[action_mask_4, 0] -= 0.05
        pos[action_mask_5, 0] += 0.05
        pos[action_mask_6, 1] -= 0.05
        pos[action_mask_7, 1] += 0.05

        self.pos = pos
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=pos,
            quat=self.quat,
        )

        self.franka.control_dofs_position(
            self.qpos[:, :-2], self.motors_dof, self.envs_idx
        )
        self.franka.control_dofs_position(finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()
        self.envs_episode_len += 1

        old_block_position = self.cube.get_pos()
        old_gripper_position = (
            self.franka.get_link("left_finger").get_pos()
            + self.franka.get_link("right_finger").get_pos()
        ) / 2
        dones = (old_block_position[:, 2] > 0.35) | (self.envs_episode_len > self.max_episode_length)
        rewards = (
            -torch.norm(old_block_position - old_gripper_position, dim=1)
            + torch.maximum(torch.tensor(0.02), old_block_position[:, 2]) * 10
        )
        
        # reset end environments
        self.reset_idx(dones.nonzero(as_tuple=False).flatten())
        
        block_position = self.cube.get_pos()
        gripper_position = (
            self.franka.get_link("left_finger").get_pos()
            + self.franka.get_link("right_finger").get_pos()
        ) / 2
        states = torch.concat([block_position, gripper_position], dim=1)

        return states, rewards, dones


if __name__ == "__main__":
    gs.init(backend=gs.gpu, precision="32")
    env = GraspFixedBlockEnv(vis=True)
