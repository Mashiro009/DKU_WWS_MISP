# encoding: utf-8
"""
@author: Haoxu, Ming Cheng
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unimodal import UniModel, UniModel_SimAMBlock

class BaseFusion2D(nn.Module):
        
    def __init__(self, in_features, out_features):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.Sigmoid(),
        )
        
    def forward(self, x1, x2, x3):  
        x2 = F.adaptive_avg_pool2d(x2,(1,1))
        x3 = F.adaptive_avg_pool2d(x3,(1,1))
        x2 = x2.view(x2.size(0),-1)
        x3 = x3.view(x3.size(0),-1)
        
        fused = x1 * torch.cat((x2,x3), axis=1) 
        fused = self.fc(fused)
        
        return fused


class HMAFusion(nn.Module):
        
    def __init__(self, num_class):
        super().__init__()
    
        self.num_class = num_class
        self.audionet = UniModel(1, num_class)
        self.videonet = UniModel(3, num_class)

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,  out_features=self.num_class),
        )

        self.fusion5 = BaseFusion2D(32*2,64*2)
        self.fusion6 = BaseFusion2D(64*2,128*2)
        self.fusion7 = BaseFusion2D(128*2,256*2)
        self.fusion8 = BaseFusion2D(256*2,256*2)


    def _get_2d_hist_embd(self, audio_feats, video_feats):
        hist_embd1 = self.fusion5(1, audio_feats['layer1'], video_feats['layer1'])
        hist_embd2 = self.fusion6(hist_embd1, audio_feats['layer2'], video_feats['layer2'])
        hist_embd3 = self.fusion7(hist_embd2, audio_feats['layer3'], video_feats['layer3'])
        hist_embd4 = self.fusion8(hist_embd3, audio_feats['layer4'], video_feats['layer4'])

        return hist_embd4
       
    
    def forward(self, x):
        feats = {}
        audio,video = x
        with torch.no_grad():
            self.audionet.eval()
            self.videonet.eval()
            audio_feats = self.audionet(audio)[1]
            video_feats = self.videonet(video)[1]
           
        decoder_hist_embd4 = self._get_2d_hist_embd(audio_feats['decoder_feats'],video_feats['decoder_feats'])
        
        total_embd =  decoder_hist_embd4 
        feats['total_embd'] = total_embd
        out = self.fc(total_embd)

        return out, feats

class SimAM_HMAFusion(HMAFusion):
    def __init__(self, num_class):
        super().__init__(num_class)

        self.audionet = UniModel_SimAMBlock(1, num_class)
        self.videonet = UniModel_SimAMBlock(3, num_class)