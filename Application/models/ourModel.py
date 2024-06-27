import torch
import torch.nn as nn
from models.modulator import Modulator
from models.efficientface import LocalFeatureExtractor, InvertedResidual
from models.transformer_timm import AttentionBlock, Attention
from cgitb import text
import re
import os
import time
import sys
import json
from tkinter import NONE
# from sqlalchemy import true
import yaml
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import librosa
import pandas as pd
from functools import reduce
import random
import copy
import math

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.funcctional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, BertConfig, AutoConfig
from transformers import  Wav2Vec2Model, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaEncoder
from infonce_loss import InfoNCE, SupConLoss
from mmi_module import MMI_Model

def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True)) 

class EfficientFaceTemporal(nn.Module):

    def __init__(self, stages_repeats, stages_out_channels, num_classes=7, im_per_sample=25):
        super(EfficientFaceTemporal, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True),)
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        self.local = LocalFeatureExtractor(29, 116, 1)
        self.modulator = Modulator(116)

        output_channels = self._stage_out_channels[-1]

        self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True),)
        self.conv1d_0 = conv1d_block(output_channels, 64)
        self.conv1d_1 = conv1d_block(64, 64)
        self.conv1d_2 = conv1d_block(64, 128)
        self.conv1d_3 = conv1d_block(128, 128)

        self.classifier_1 = nn.Sequential(
                nn.Linear(128, num_classes),
            )
        self.im_per_sample = im_per_sample
        
    def forward_features(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.modulator(self.stage2(x)) + self.local(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3]) #global average pooling
        return x

    def forward_stage1(self, x):
        #Getting samples per batch
        assert x.shape[0] % self.im_per_sample == 0, "Batch size is not a multiple of sequence length."
        n_samples = x.shape[0] // self.im_per_sample
        x = x.view(n_samples, self.im_per_sample, x.shape[1])
        x = x.permute(0,2,1)
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x
        
        
    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x
    
    def forward_classifier(self, x):
        x = x.mean([-1]) #pooling accross temporal dimension
        x1 = self.classifier_1(x)
        return x1
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x  

def init_feature_extractor(model, path):
    if path == 'None' or path is None:
        return
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    pre_trained_dict = checkpoint['state_dict']
    pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
    print('Initializing efficientnet')
    model.load_state_dict(pre_trained_dict, strict=False)

def init_feature_extractor_audio(model, path):
    if path == 'None' or path is None:
        return
    checkpoint = torch.load(path)
    state_dict = {key.replace('module.', ''):value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    
def get_model(num_classes, task, seq_length):
    model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, task, seq_length)
    return model  


def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding='valid'),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True), nn.MaxPool1d(2,1))

class FuseModel(nn.Module):

    def __init__(self, text_config):

        super().__init__()

        tran_dim = 768

        self.config_mmi = BertConfig('config.json')
        self.model_mmi = MMI_Model(self.config_mmi,len(audio_processor.tokenizer),4)

        self.temperature = 0.07

        self.orgin_linear_change = nn.Sequential(
            nn.Linear(tran_dim*2, tran_dim),
            ActivateFun('gelu'),
            nn.Linear(tran_dim, tran_dim)
        )

        self.augment_linear_change = nn.Sequential(
            nn.Linear(tran_dim*2, tran_dim),
            ActivateFun('gelu'),
            nn.Linear(tran_dim, tran_dim)
        )

    def forward_encoder(self, text_output, attention_mask, audio_inputs, audio_length, ctc_labels, emotion_labels, augmentation = False):

        #bert_attention_mask, audio_input, audio_length, ctc_labels, emotion_labels, text_output, augmentation = False
        emotion_logits, logits, loss_cls, loss_ctc = self.model_mmi(text_output, attention_mask, audio_inputs, audio_length, ctc_labels, emotion_labels, augmentation = augmentation)

        return emotion_logits, logits, loss_cls, loss_ctc

    def forward(self, text_output, attention_mask, audio_inputs, audio_length, ctc_labels, emotion_labels):

        emotion_logits, logits, loss_cls, loss_ctc = self.forward_encoder(text_output, attention_mask, audio_inputs, audio_length, ctc_labels, emotion_labels)

        return emotion_logits

class MultiModalCNN(nn.Module):
    def __init__(self, num_classes=8, fusion='ia', seq_length=15, pretr_ef='None', num_heads=1):
        super(MultiModalCNN, self).__init__()
        
        #self.audio_model = AudioCNNPool(num_classes=num_classes)
        self.visual_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, seq_length)
        
        config_mmi = BertConfig('config.json')
        self.audio_model = FuseModel(config_mmi)
        del config_mmi

        init_feature_extractor(self.visual_model, pretr_ef)
        init_feature_extractor_audio(self.audio_model, pretr_ef)
                           
        # e_dim = 128
        # input_dim_video = 128
        # input_dim_audio = 128
        self.fusion=fusion

        # self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim, num_heads=num_heads)
        # self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim, num_heads=num_heads)
  
        self.classifier_1 = nn.Sequential(
                    nn.Linear(1792, num_classes),
                )
        
            

    def forward(self, x_audio, x_visual,bert_output, attention_mask, acoustic_input, acoustic_length, ctc_labels, emotion_labels):
       
        x_visual = self.visual_model.forward_features(x_visual) 

        x_audio = self.audio_model(bert_output, attention_mask, acoustic_input, acoustic_length, ctc_labels, emotion_labels)
           
        # proj_x_a = proj_x_a.permute(0, 2, 1)
        # proj_x_v = proj_x_v.permute(0, 2, 1)
        # h_av = self.av(proj_x_v, proj_x_a)
        # h_va = self.va(proj_x_a, proj_x_v)
       
        # audio_pooled = h_av.mean([1]) #mean accross temporal dimension
        # video_pooled = h_va.mean([1])

        x = torch.cat((x_audio, x_visual), dim=-1)  
        x1 = self.classifier_1(x)
        return x1
 