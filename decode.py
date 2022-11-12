# encoding: utf-8
"""
@author: Haoxu, Ming Cheng
"""

import importlib

from tqdm import tqdm
from data.audiovideo_ref_dataset import AVDataset3D
import argparse
import os
from easydict import EasyDict as edict
import torch
from core.metrics import get_accuracy, get_false_reject_rate, get_false_accept_rate, get_WWS_score
import warnings

warnings.filterwarnings('ignore')


def get_args():
    parser = argparse.ArgumentParser(description='decode with pretrain model')
    parser.add_argument('--model_lib', default='models.hma_fusion', help='model library')
    parser.add_argument('--model_name', default='SimAM_HMAFusion', help='model name')
    parser.add_argument('--decode_modal', choices=['audio', 'video', 'audiovideo'], default='audiovideo', help='the modal decoded')

    parser.add_argument('--checkpoint', required=True, help='checkpoint model file')
    parser.add_argument('--test_audio_data', default='misp_dataset/eval_far_v1', help='test audio data dir')
    parser.add_argument('--test_video_data', default='misp_dataset/eval_lips_video_v1', help='test audio data dir')

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()

    audio_field = os.path.basename( args.test_audio_data ) # eval_far_v1
    video_field = os.path.basename( args.test_video_data ) # eval_lips_video_v1
    super_dir = os.path.dirname( args.test_audio_data ) # misp_dataset
    
    
    hparams = edict()

    hparams.model_lib = args.model_lib
    hparams.model_name = args.model_name

    hparams.audio = edict()
    hparams.audio.valid_field = [audio_field]
    hparams.audio.valid_video_field = [video_field]
    hparams.audio.frame_nums = 64 # 2.56s
    hparams.audio.slice_stride = 4 # 0.1s * 4 = 0.04s = 1 / 25 s
    hparams.audio.slice_length = 80
    hparams.audio.feature_extraction_conf=dict(
        mel_bins=80,
        frame_shift=10,
        frame_length=25)
    hparams.audio.pad_middle = True

    hparams.video = edict()
    hparams.video.frame_nums = 64
    hparams.video.frame_size = [112,112]
    hparams.video.pad_middle = True

    hparams.va = edict()
    hparams.va.random_replace_video = True

    valid_set = AVDataset3D(data_dir=super_dir, 
                        is_train=False,
                        groups=['eval'],
                        ref_fields=[audio_field],
                        fields=hparams.audio.valid_field,
                        hparams=hparams,
                        use_beamforming=0.5,
                        video_fields=hparams.audio.valid_video_field)

    model = importlib.import_module(hparams.model_lib).__getattribute__(hparams.model_name)(
        num_class = 1)

    model.load_state_dict(torch.load(args.checkpoint))
    print('Load pre-trained model from: {}'.format(args.checkpoint))

    y_pred_list = []
    y_true_list = []

    model.eval()
    model.cuda()

    for i in tqdm(range(len(valid_set))):
        [audio, video], y = valid_set[i]
        audio = audio.unsqueeze(0).cuda()
        video = video.unsqueeze(0).cuda()
        o, _ = model([audio, video])
        y_true_list += [y]
        y_pred_list += (torch.sigmoid(o.squeeze(-1)) > 0.5).int().tolist()

    av_train_metrics = {
                        'train_accuracy':get_accuracy(y_true_list, y_pred_list),
                        'train_FRR':get_false_reject_rate(y_true_list, y_pred_list),
                        'train_FAR':get_false_accept_rate(y_true_list, y_pred_list),
                        'train_WWS':get_WWS_score(y_true_list, y_pred_list),
                        }
    
    print(av_train_metrics)

if __name__ == '__main__':
    main()