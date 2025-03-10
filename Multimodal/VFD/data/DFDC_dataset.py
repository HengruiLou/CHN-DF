import os
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import numpy as np
from tqdm import tqdm
import warnings
import librosa
warnings.filterwarnings('ignore')


class DFDCDataset(BaseDataset):

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.stage = opt.mode
        video_dataset_path_real = os.path.join(self.opt.dataroot, 'real')
        video_dataset_path_fake = os.path.join(self.opt.dataroot, 'fake')
        audio_dataset_path = os.path.join(self.opt.dataroot, 'audio_feat')
        self.video_real, self.video_fake, self.audio_real, self.audio_fake, self.audio_name = \
            self.get_video_list(video_dataset_path_real, video_dataset_path_fake, audio_dataset_path)

        self.transform = get_transform(self.opt)


    def __getitem__(self, index):

        video_real = self.video_real
        video_fake = self.video_fake
        audio_real = self.audio_real
        audio_fake = self.audio_fake
        audio_name = self.audio_name

        scale = len(video_fake)/len(video_real)
        #print("scale",scale)

        img_real = []
        aud_real = []
        img_fake = []
        aud_fake = []

        image_input = Image.open(video_real[index]).convert('RGB')
        img_d = self.transform(image_input)
        audio = np.load(audio_real[index])
        audio_d = librosa.util.normalize(audio)
        img_real.append(img_d)
        aud_real.append(audio_d)

        audio_id = audio_name[index]

        start = round(scale * index)
        end = max(round(scale * (index + 1)),start+1)
        #print(f"index: {index}, start: {start}, end: {end}")
        for ind in range(start, min(end, start + 30)):
            image_input = Image.open(video_fake[ind]).convert('RGB')
            img_d = self.transform(image_input)
            img_fake.append(img_d)

            audio = np.load(audio_fake[ind])
            audio_d = librosa.util.normalize(audio)
            #print("audio_d",audio_d)
            aud_fake.append(audio_d)
        #print("aud_real",aud_real)
        aud_real = np.stack(aud_real, axis=0)
        #print("aud_fake",aud_fake)
        aud_fake = np.stack(aud_fake, axis=0)
        img_real = np.stack(img_real, axis=0)
        img_fake = np.stack(img_fake, axis=0)
        aud_real = np.expand_dims(aud_real, 1)
        aud_fake = np.expand_dims(aud_fake, 1)
        return {
          'id': audio_id,
          'img_real': img_real,
          'img_fake': img_fake,
          'aud_real': aud_real,
          'aud_fake': aud_fake,
        }

    def __len__(self):

        return len(self.video_real)

    def get_video_list(self, dataset_path_real, dataset_path_fake, audio_dataset_path):
        video_feat_path_real = dataset_path_real
        video_feat_path_fake = dataset_path_fake
        audio_feat_path = audio_dataset_path

        video_real_path = []
        video_fake_path = []
        audio_real_path = []
        audio_fake_path = []
        audio_name = []
        for i in tqdm(os.listdir(video_feat_path_real)):
            video_real_path.append(os.path.join(video_feat_path_real, i))
            video_name = i.split('.png')[0]
            audio_real_path.append(os.path.join(audio_feat_path, video_name+'.npy'))
            audio_name.append(video_name)

        for i in tqdm(os.listdir(video_feat_path_fake)):
            video_fake_path.append(os.path.join(video_feat_path_fake, i))
            video_name = i.split('.png')[0]
            audio_fake_path.append(os.path.join(audio_feat_path, video_name+'.npy'))
            audio_name.append(video_name)
        return video_real_path, video_fake_path, audio_real_path, audio_fake_path, audio_name
