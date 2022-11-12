# encoding: utf-8
"""
@author: Haoxu, Ming Cheng
"""

from abc import abstractmethod
import os, random, torch
import numpy as np
from torch.utils.data import Dataset
# from utils.tools import *
from data.audio_aug import load_audio, norm_spectrum, spec_aug, spectrums_clipping
from data.audio_aug import extract_fbank, slice_spectrum
from data.video_aug import load_video, video_clipping

class AVBaseDataset(Dataset):
    def __init__(self, data_dir, is_train, groups, ref_fields, fields, hparams=None, use_beamforming=False, video_fields=None):
        super(torch.utils.data.Dataset, self).__init__()
        self.is_train = is_train
        # self.pos_durs = self.init_pos_durs(hparams.audio.neg_subsegment_conf['pos_utt2dur_file'])
        
        # self.ref_utt2path = self.init_ref_utt2path(data_dir, ref_fields)

        # init data list
        self.datalist = self.get_data_records(data_dir, groups, ref_fields)
        self.hparams = hparams
        self.fields = fields

        self.labels = {'FREETEXT':0, 'XIAOTXIAOT':1, 'unknown':2}
        self.use_beamforming = float(use_beamforming)

        self.utt2video, self.video_collection = self.get_utt2video(data_dir, video_fields)

        self.ref2utt_fields = dict()

        # for field in ['near', 'middle','far','near_aug','train_mvdr']:
        for field in fields:
            self.ref2utt_fields[field] = self.add_ref_audio('./{}/{}'.format(data_dir,field))

    def add_ref_audio(self, data_dir):
        # near utt, ref id -> each field utt list
        ref2utt = dict()
        for data in self.datalist:
            ref2utt[data['ref_id']] = []

        utt2wav = dict()

        with open(os.path.join(data_dir, 'wav.scp'), 'r') as fin:
            for line in fin:
                line = line.strip()
                wavid = line.split(' ')[0]
                wavpath = line.split(' ')[1]
                utt2wav[wavid] = wavpath

        if os.path.exists(os.path.join(data_dir, 'utt2ref')):
            with open(os.path.join(data_dir, 'utt2ref'), 'r') as fin:
                for line in fin:
                    line = line.strip()
                    wavid = line.split(' ')[0]
                    ref = line.split(' ')[1]
                    ref2utt[ref].append(utt2wav[wavid])

        return ref2utt

    
    def __len__(self):
        return len(self.datalist)


    @abstractmethod
    def load_audio_data(self, audio_path, now_label):
        pass

    def load_audio_data_frontend(self, audio_path, now_label):

        # 读取音频 read audio
        signal, sr = load_audio(audio_path, sample_rate=16000)
        signal = signal / (1 << 15)

        # extract fbank feature
        # 提取fbank特征
        spectrum = extract_fbank(signal, sr, self.hparams['audio']['feature_extraction_conf'])

        # normalize feature
        # 对特征进行归一化处理
        spectrum = norm_spectrum(spectrum)

        # specaug
        if self.is_train:
            spectrum = spec_aug(spectrum)

        return spectrum, now_label

    def __getitem__(self, idx):
        ref_utt_id = self.datalist[idx]['ref_id']
        # ['near', 'middle','far','near_aug','train_mvdr']
        audio_pool = []
        for field in self.fields:
            # add audio uttid from each field
            audio_pool += self.ref2utt_fields[field][ref_utt_id]

        if len(audio_pool) == 0:
            raise ValueError('error')
        audio_path = np.random.choice(audio_pool)
        now_label = self.datalist[idx]['label']

        audio_x, y = self.load_audio_data(audio_path, now_label)

        if self.is_train:
            video_pool = [ref_utt_id.replace('Near', 'Middle'), ref_utt_id.replace('Near', 'Far')]
        else:
            video_pool = [ref_utt_id]
        video_new_pool = video_pool.copy()
        for video_utt in video_pool:
            if video_utt not in self.utt2video.keys():
                # pool.
                video_new_pool.remove(video_utt)
        if len(video_new_pool) == 0:
            video_new_pool.append(np.random.choice(self.video_collection['neg' if y == 0 else 'pos']))

        # random change the video from same label
        # 随机替换同标签的视频
        if self.is_train and self.hparams['va']['random_replace_video'] and np.random.random() < 0.5:
            video_path = self.random_replace_video(y)
            video_new_pool = [video_path]
        
        video_x = self.load_video_data(self.utt2video[np.random.choice(video_new_pool)])

        return [audio_x, video_x], y

    def get_utt2video(self, data_dir, train_loader):
        utt2video = dict()
        video_collection = dict()
        video_collection['pos'] = []
        video_collection['neg'] = []
        for video_field in train_loader:
            with open(os.path.join(os.path.join(data_dir, video_field), 'lip.scp'), 'r') as fin:
                for line in fin:
                    line = line.strip()
                    uttid = line.split(' ')[0]
                    videopath = line.split(' ')[1]
                    utt2video[uttid] = videopath
            with open(os.path.join(os.path.join(data_dir, video_field), 'text'), 'r') as fin:
                for line in fin:
                    line = line.strip()
                    uttid = line.split(' ')[0]
                    txt = line.split(' ')[1]
                    if txt == 'XIAOTXIAOT':
                        video_collection['pos'].append(uttid)
                    else:
                        video_collection['neg'].append(uttid)

        return utt2video, video_collection

    def get_data_records(self, data_dir, groups, fields):

        tasks = []
        for field in fields:
            now_data = os.path.join(data_dir, field);
            self.add_utt_from_dir(now_data, tasks)
        
        print("mono: {}".format(len(tasks)))

        if self.is_train:
            random.shuffle(tasks)
        
        return tasks

    def add_utt_from_dir(self, data_dir, tasks):
        wav2txt = dict()
        utt2ref = dict()
        with open(os.path.join(data_dir, 'text'), 'r') as fin:
            for line in fin:
                line = line.strip()
                wavid = line.split(' ')[0]
                txt = line.split(' ')[1]
                wav2txt[wavid] = txt
                utt2ref[wavid] = None

        if os.path.exists(os.path.join(data_dir, 'utt2ref')):
            with open(os.path.join(data_dir, 'utt2ref'), 'r') as fin:
                for line in fin:
                    line = line.strip()
                    wavid = line.split(' ')[0]
                    ref = line.split(' ')[1]
                    utt2ref[wavid] = ref

        with open(os.path.join(data_dir, 'wav.scp'), 'r') as fin:
            for line in fin:
                line = line.strip()
                wavid = line.split(' ')[0]
                wavpath = line.split(' ')[1]
                tasks.append({'wavid':wavid,'audio_path':wavpath, 'label':wav2txt[wavid], 'ref_id':utt2ref[wavid]})

        return tasks

    def load_video_data(self, video_path):
        video_data = load_video(video_path, target_size=tuple(self.hparams['video']['frame_size']))



        bound_data, video_data = video_clipping(video_data, 
            target_length=self.hparams['video']['frame_nums'], 
            random=self.is_train,
            pad_middle=self.hparams['video']['pad_middle']
        )

        if self.is_train:
            # video_augmentation
            pass

        # normalize
        video_data = video_data / 255.0
        # to pytorch-like tensor
        x = torch.from_numpy(video_data).permute(3,0,1,2).float()
    
        return x
    
    def init_pos_durs(self, filepath):
        pos_durs = set()
        with open(filepath, 'r') as fin:
            for line in fin:
                line = line.strip()
                dur = float(line.split(' ')[1])
                pos_durs.add(dur)
        return list(pos_durs)

    def random_replace_video(self, label):
        if label == 1:
            new_video_path = np.random.choice(self.video_collection['pos'])
        if label == 0:
            new_video_path = np.random.choice(self.video_collection['neg'])
   
        return new_video_path


class AVDataset3D(AVBaseDataset):
    
    def __init__(self, data_dir, is_train, groups, ref_fields, fields, hparams=None, use_beamforming=False, video_fields=None):
        super().__init__(data_dir, 
            is_train, 
            groups,
            ref_fields,
            fields,
            hparams=hparams, 
            use_beamforming=use_beamforming, 
            video_fields=video_fields)




    def load_audio_data(self, audio_path, now_label):
        spectrum, now_label = self.load_audio_data_frontend(audio_path, now_label)

        # time维度 进行slice 以0.4s为步长 slice出80*80的图
        # (T, 80) -> (T/4, 80, 80, 1)
        spectrums = slice_spectrum(spectrum, self.hparams.audio.slice_stride, self.hparams.audio.slice_length)
        # 进行pad (T/4, 80, 80, 1) -> (frame_nums 64 , 80, 80, 1)
        bound_data, spectrums = spectrums_clipping(spectrums, self.hparams.audio.frame_nums, 
            random=self.is_train, pad_middle=self.hparams.audio.pad_middle)

        # to pytorch-like tensor
        x = torch.from_numpy(spectrums).permute(3,0,1,2).float()
        y = self.labels[now_label]

        return x,y

    
    
