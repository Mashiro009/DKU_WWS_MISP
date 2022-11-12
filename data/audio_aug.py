# encoding: utf-8
"""
@author: Haoxu, Ming Cheng
"""


import os
import scipy.io.wavfile as sciwav
import librosa
import random
import numpy as np
import torchaudio
import torch


def load_audio(audio_path, sample_rate, channel=0):
    if os.path.splitext(audio_path)[-1] == '.wav':
        sr, signal = sciwav.read(audio_path, mmap=True)
    elif os.path.splitext(audio_path)[-1] == '.m4a':
        signal, sr = librosa.load(audio_path, sr=sample_rate)
    if sr != sample_rate:
        signal, sr = librosa.load(audio_path, sr=sample_rate)
    if len(signal.shape) == 2 and channel:
        channel = random.choice(channel) if type(channel)==list else channel
        return signal[:, channel]
    
    return signal, sr

def extract_fbank(signal, sr, feature_extraction_conf):
    if isinstance(signal, np.ndarray):
        signal = torch.from_numpy(signal)
        
    if len(signal.shape) < 2:
        signal = signal.unsqueeze(0)
        
    signal = signal * (1<<15)
    mat = torchaudio.compliance.kaldi.fbank(
            signal,
            num_mel_bins=feature_extraction_conf['mel_bins'],
            frame_length=feature_extraction_conf['frame_length'],
            frame_shift=feature_extraction_conf['frame_shift'],
            energy_floor=0.0,
            sample_frequency=sr)

    # centralize
    # mat = mat - mat.mean()
    mat = mat.detach().cpu().numpy()
    
    return mat



def norm_spectrum(spectrum):
    spectrum = spectrum.copy()
    _mean = np.mean(spectrum, axis=0)
    _std = np.std(spectrum, axis=0)
    if 0 in _std:
        return spectrum - _mean
    # if _std == 0:
    #     return spectrum
    # spectrum = (spectrum - _mean) / _std
    spectrum = spectrum - _mean
    # signal = (signal - np.mean(signal)) / np.std(signal)

    return spectrum

def spec_aug(spectrum, num_t_mask=2, num_f_mask=2, max_t=20, max_f=10):
    """ Do spec augmentation
        Inplace operation
        Args:
            spectrum: 
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
        Returns
            Iterable[{key, feat, label}]
    """
    y = spectrum.copy()
    max_frames = y.shape[0]
    max_freq = y.shape[1]
    for i in range(num_t_mask):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        y[start:end, :] = 0
    # freq mask
    for i in range(num_f_mask):
        start = random.randint(0, max_freq - 1)
        length = random.randint(1, max_f)
        end = min(max_freq, start + length)
        y[:, start:end] = 0
    return y



def slice_spectrum(spectrum, slice_stride, slice_length):
    if len(spectrum) < slice_length:
        h,w = spectrum.shape
        pad = np.zeros((slice_length-h,w))
        spectrum = np.vstack((spectrum,pad))

    results = []
    for i in range(0,len(spectrum),slice_stride):
        s = i
        e = i + slice_length
        if e <= len(spectrum):
            data = np.expand_dims(spectrum[s:e],axis=-1)
            results.append(data)

    results = np.array(results)      
    
    return results


def spectrums_clipping(spectrums, target_length, random, pad_middle=True):
    spectrums = spectrums.copy()
    current_length = len(spectrums)
    if random:
        # random select start point 
        # 随意裁剪出一个连续的区间
        s = np.random.randint(max(current_length-target_length+1, 1))
        e = s + target_length
    else:
        # select middle as center
        # 如果不是random clip，从中心裁剪
        c = current_length / 2
        s = int(c - target_length/2)
        e = int(c + target_length/2)
    # prevent bound overflow
    s = max(s, 0)
    e = min(e, len(spectrums))
    spectrums = spectrums[s:e]

    # pad the spectrums
    bound_data, spectrums = spectrums_padding(spectrums, target_length, pad_value=0, pad_middle=pad_middle)
    
    return bound_data, spectrums


def spectrums_padding(spectrums, target_length, pad_value, pad_middle=True):
    spectrums = spectrums.copy()
    # 默认整段clip都是有效的
    bound_data = np.ones(len(spectrums))
    gap = target_length - len(spectrums)
    if gap > 0:
        # 计算clip前后需要插入的帧数
        num1 = int(gap/2)
        num2 = int(np.ceil(gap/2))
        # 生成对应的张量, 完成拼接
        pad1 = np.ones([num1]+list(spectrums.shape[1:]), dtype=np.uint8) * pad_value
        pad2 = np.ones([num2]+list(spectrums.shape[1:]), dtype=np.uint8) * pad_value
        if pad_middle:
            spectrums = np.concatenate([pad1,spectrums,pad2])
        else:
            spectrums = np.concatenate([spectrums,pad1,pad2])
        # 同步处理对应的bound信息
        bd1 = np.zeros(num1)
        bd2 = np.zeros(num2)
        if pad_middle:
            bound_data = np.concatenate([bd1,bound_data,bd2])
        else:
            bound_data = np.concatenate([bound_data,bd1,bd2])
        
    return bound_data, spectrums