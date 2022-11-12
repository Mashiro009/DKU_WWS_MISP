# encoding: utf-8
"""
@author: Haoxu, Ming Cheng
"""

import os
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
# from utils.tools import *


def load_video(video_path, target_size=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileExistsError('Cannot Open: {}'.format(video_path))

    w,h = int(cap.get(3)), int(cap.get(4)) 
    num = int(cap.get(7))
    fps = int(cap.get(5))

    frames = []
    for _ in range(num):
        ret,img = cap.read()
        if not ret:
            img = np.zeros((h,w,3), dtype=np.uint8) 
        if target_size is not None:
            img = cv2.resize(img, target_size)

        frames.append(img)
    
    video_data = np.array(frames)
    cap.release()
    
    return video_data

def video_padding(video_data, target_length, pad_value, pad_middle=True):
    video_data = video_data.copy()
    # 默认整段clip都是有效的
    bound_data = np.ones(len(video_data))
    gap = target_length - len(video_data)
    if gap > 0:
        # 计算clip前后需要插入的帧数
        num1 = int(gap/2)
        num2 = int(np.ceil(gap/2))
        # 生成对应的张量, 完成拼接
        pad1 = np.ones([num1]+list(video_data.shape[1:]), dtype=np.uint8) * pad_value
        pad2 = np.ones([num2]+list(video_data.shape[1:]), dtype=np.uint8) * pad_value
        if pad_middle:
            video_data = np.concatenate([pad1,video_data,pad2])
        else:
            video_data = np.concatenate([video_data,pad1,pad2])
        # 同步处理对应的bound信息
        bd1 = np.zeros(num1)
        bd2 = np.zeros(num2)
        if pad_middle:
            bound_data = np.concatenate([bd1,bound_data,bd2])
        else:
            bound_data = np.concatenate([bound_data,bd1,bd2])
        
    return bound_data, video_data


def video_clipping(video_data, target_length, random, pad_value=128, pad_middle=True):
    video_data = video_data.copy()
    
    current_length = len(video_data)
    if random:
        # 随意裁剪出一个连续的区间
        s = np.random.randint(max(current_length-target_length+1, 1))
        e = s + target_length
    else:
        # 如果不是random clip，从中心裁剪
        c = current_length / 2
        s = int(c - target_length/2)
        e = int(c + target_length/2)
    # 防止边界溢出，截取视频
    s = max(s, 0)
    e = min(e, len(video_data))
    video_data = video_data[s:e]
    # 对于长度不够的视频需要pad，并指出有效区间
    bound_data, video_data = video_padding(video_data, target_length, pad_value=pad_value, pad_middle=pad_middle)
    
    return bound_data, video_data