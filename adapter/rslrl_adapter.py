import torch

class RslrlAdapter:
    def __init__(self, genesis_env):
        self.env = genesis_env

        self.num_envs = self.env.num_envs
        self.num_obs = self.env.state_dim 
        self.num_actions = self.env.action_space
        self.num_privileged_obs = None
        self.obs_buf = None

        self.max_episode_length = self.env.max_episode_length

    def reset(self):
        obs = self.env.reset()
        self.obs_buf = obs
        return obs, None
    
    def step(self, actions):

        # 如果 actions 的形状为 [num_envs, num_actions]，则转换为离散动作索引
        if actions.dim() > 1 and actions.size(1) == self.num_actions:
            actions = torch.argmax(actions, dim=1)

        obs, rewards, dones = self.env.step(actions)
        self.obs_buf = obs

        infos = {}
        return self.obs_buf, None, rewards, dones, infos
    
    def get_observations(self):
        return self.obs_buf 

    def get_privileged_observations(self):
        return None

    def start_recording(self):
        # self.env.start_recording()
        pass

    def get_recorded_frames(self):
        return None

    @property
    def dt(self):
        return self.env.scene.sim_options.dt