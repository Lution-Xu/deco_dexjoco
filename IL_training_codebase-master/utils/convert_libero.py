import os
import io
import pandas as pd
from tqdm import tqdm

data_path = '/root/yusun/datasets/libero_ori/data/chunk-000'
save_path = '/root/yusun/datasets/data_libero'
parq_list = sorted(os.listdir(data_path))
episode_info = None
for parq in tqdm(parq_list):
    parq_path = os.path.join(data_path, parq)
    file = pd.read_parquet(parq_path)
    for i in range(len(file)):
        current_episode = str(file.loc[i, 'episode_index']).zfill(4)
        save_episode = os.path.join(save_path, current_episode)
        save_img1 = os.path.join(save_episode, 'img1')
        save_img2 = os.path.join(save_episode, 'img2')
        if not os.path.exists(save_episode):
            if episode_info is not None:
                df = pd.DataFrame(episode_info).T
                df.reset_index(inplace=True)
                df.to_pickle(os.path.join(save_path, last_episode, 'episode_info.pkl'))
            episode_info = {}  # initialize episode_info for each new episode
            os.makedirs(save_episode)
            os.makedirs(save_img1)
            os.makedirs(save_img2)
        
        img_dict = file.loc[i, 'observation.images.image']
        img_wrist_dict = file.loc[i, 'observation.images.image2']
        with open(os.path.join(save_img1, img_dict['path']), 'wb') as f:
            f.write(img_dict['bytes'])

        with open(os.path.join(save_img2, img_wrist_dict['path']), 'wb') as f:
            f.write(img_wrist_dict['bytes'])
        
        episode_info[i] = {'state': file.loc[i, 'observation.state'], 'action': file.loc[i, 'action'], 'task': file.loc[i, 'task_index']}
        last_episode = current_episode

# save the last episode
if episode_info is not None:
    df = pd.DataFrame(episode_info).T
    df.reset_index(inplace=True)
    df.to_pickle(os.path.join(save_path, last_episode, 'episode_info.pkl'))