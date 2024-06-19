import torch
import torch.utils.data as data

import os
import pandas as pd
from ravdess import get_default_video_loader, get_mfccs, load_audio
import iemocap_preprocessing.config as config


class IEMOCAP(data.Dataset):
    def __init__(self, subset,  spatial_transform=None,
                 get_loader=get_default_video_loader, data_type = 'audiovisual', audio_transform=None):
        
        data_df = pd.read_csv(config.CSV_PATH)
        self.data = data_df[data_df["category"]== subset]   
        self.audio_root_path = config.AUDIO_PATH
        self.video_root_path =  config.FACE_EXTRACTED_NPM 
        
        self.spatial_transform = spatial_transform
        self.audio_transform=audio_transform
        self.loader = get_loader()
        self.data_type = data_type 
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        name = row["RecordingName"]
        label = row["Emotion"]
        text = row["Transcription"]
        
        # prepare the video
        path_video = os.path.join(self.video_root_path, name+".npy")
        clip = self.loader(path_video)
        if self.spatial_transform is not None:               
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]            
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        
        # prepare the audio
        path_audio = os.path.join(self.audio_root_path, name+".wav")
        y, sr = load_audio(path_audio) 
        if self.audio_transform is not None:
                self.audio_transform.randomize_parameters()
                y = self.audio_transform(y)     
                
        mfcc = get_mfccs(y, sr)            
        audio_features = mfcc 
        
        if self.data_type == "audio":
            return audio_features, label
        elif self.data_type == "video":
            return clip, label
        elif self.data_type == "audiovisual":
            return audio_features, clip, label
        elif self.data_type == "text":
            return text, label
        elif self.data_type == "audiotext":
            return audio_features, text, label
        elif self.data_type == "videotext":
            return clip, text, label
        elif self.data_type == "audiovisualtext":
            return audio_features, clip, text, label
    
    def __len__(self):
        return len(self.data)