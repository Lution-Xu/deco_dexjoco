import os
import json
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms



class libero_Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None, **args):
        self.img_list = []
        self.root = data_dir
        self.transform = transform
        self.norm_type = args['norm_type']
        self.chunksize = args['chunk_size']
        self.lable_dict = {} 

        self.obs_mean = torch.tensor(args['observation_mean'])
        self.obs_std = torch.tensor(args['observation_std']).clamp_min(1e-8)
        self.obs_min = torch.tensor(args['observation_min'])
        self.obs_max = torch.tensor(args['observation_max'])


        self.action_mean = torch.tensor(args['action_mean'])
        self.action_std = torch.tensor(args['action_std']).clamp_min(1e-8)
        self.action_min = torch.tensor(args['action_min'])
        self.action_max = torch.tensor(args['action_max'])

        episode_list = os.listdir(data_dir)
        total_episodes = len(episode_list)
        random.seed(42)
        random.shuffle(episode_list)
        if train:
            episode_list = episode_list[:int(total_episodes)]
        else:
            episode_list = episode_list[int(total_episodes * 0.9):]
        
        for episode in episode_list:
            episode_path = os.path.join(data_dir, episode, 'img1')
            self.lable_dict[episode] = os.path.join(episode, 'episode_info.pkl')
            for img in os.listdir(episode_path):
                self.img_list.append(os.path.join(episode, 'img1', img))
        

    def __getitem__(self, index):
        relat_path = self.img_list[index]
        episode_id, img_name = relat_path.split(os.sep)[0], relat_path.split(os.sep)[-1]
        img_idx = int(img_name.split('frame_')[1].split('.')[0])

        img1_path = os.path.join(self.root, relat_path)
        img2_path = img1_path.replace('img1', 'img2')

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        seed = random.randint(0, 1000000000)
        if self.transform is not None: # same transform for both images
            self.seed_all(seed)
            img1 = self.transform(img1)
            self.seed_all(seed)
            img2 = self.transform(img2)
        
        label_pkl = pd.read_pickle(os.path.join(self.root, self.lable_dict[episode_id]))
        task_condion = torch.tensor(label_pkl.iloc[img_idx]['task'], dtype=torch.long)
        obs_state = torch.tensor(label_pkl.iloc[img_idx]['state']).float()
        action = np.stack(label_pkl.iloc[img_idx:]['action'], axis=0)
        action = torch.from_numpy(action).float()

        if action.size(0) < self.chunksize:      # padding action2chunksize and add mask
            padd_len = self.chunksize - action.size(0)
            last_action = action[-1].unsqueeze(0).repeat(padd_len, 1)  # [padd_len, 26]
            action_padd = torch.cat([action, last_action], dim=0)  # [self.chunksize, 26]
            mask = torch.cat([torch.ones(action.size(0), dtype=torch.bool), torch.zeros(padd_len, dtype=torch.bool)], dim=0)
        else:
            action_padd = action[:self.chunksize]  # clamp to chunksize and add mask 
            mask = torch.ones(self.chunksize, dtype=torch.bool)
        
        if self.norm_type == 'min_max':
            obs_state = (obs_state - self.obs_min) / (self.obs_max - self.obs_min)
            action_padd = (action_padd - self.action_min[None, :]) / (self.action_max[None, :] - self.action_min[None, :])
        else:
            obs_state = (obs_state - self.obs_mean) / self.obs_std
            action_padd = (action_padd - self.action_mean[None, :]) / self.action_std[None, :]

        tac1, tac2 = torch.tensor([-1]), torch.tensor([-1])

        return img1, img2, tac1, tac2, obs_state, action_padd, mask, task_condion

    def __len__(self):
        return len(self.img_list)

    def seed_all(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
