import torch

class RslrlAdapter:
    def __init__(self, genesis_env, max_episode_length=1000):
        self.env = genesis_env

        self.num_envs = self.env.num_envs
        self.num_obs = self.env.state_dim 
        self.num_actions = self.env.action_space  # 8
        self.num_privileged_obs = None
        self.obs_buf = None

        self.max_episode_length = max_episode_length
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int32, device=self.env.device)

    def reset(self):
        obs = self.env.reset()
        self.obs_buf = obs
        self.episode_length_buf.zero_()
        return obs, None
    
    def step(self, actions):

        # 如果 actions 的形状为 [num_envs, num_actions]，则转换为离散动作索引
        if actions.dim() > 1 and actions.size(1) == self.num_actions:
            actions = torch.argmax(actions, dim=1)

        obs, rewards, dones = self.env.step(actions)
        self.episode_length_buf += 1
        self.obs_buf = obs
        dones = (self.episode_length_buf >= self.max_episode_length) | dones

        infos = {}
        return obs, None, rewards, dones, infos
    
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