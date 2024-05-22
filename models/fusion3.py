import torch
import torch.nn as nn
from models.modulator import Modulator
from models.efficientface import LocalFeatureExtractor, InvertedResidual
import numpy as np
from functools import reduce
import torch
import torch.nn as nn
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
from pydub import AudioSegment
from models.modulator import Modulator
from models.efficientface import LocalFeatureExtractor, InvertedResidual
from models.transformer_timm import AttentionBlock, Attention
import time

def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True)) 

def weights_initialize(model):
        model.projector = nn.Linear(1024, 1024, bias=True)
        model.classifier = nn.Linear(1024, 8, bias=True)
        torch_state_dict = torch.load('pytorch_model.bin', map_location=torch.device('cuda'))
        

        model.projector.weight.data = torch_state_dict['classifier.dense.weight']
        model.projector.bias.data = torch_state_dict['classifier.dense.bias']

        model.classifier.weight.data = torch_state_dict['classifier.output.weight']
        model.classifier.bias.data = torch_state_dict['classifier.output.bias']
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        model.classifier = nn.Identity()

        return model, feature_extractor


def predict_emotion(audio_file, model, feature_extractor):
        model.eval()
        sound = AudioSegment.from_file(audio_file)
        sound = sound.set_frame_rate(16000)
        sound_array = np.array(sound.get_array_of_samples())
        # this model is VERY SLOW, so best to pass in small sections that contain
        # emotional words from the transcript. like 10s or less.
        # how to make sub-chunk  -- this was necessary even with very short audio files
        # test = torch.tensor(input.input_values.float()[:, :100000])

        input = feature_extractor(
            raw_speech=sound_array,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt")
        
        
        with torch.no_grad():
            input = input.to('cuda:0')
            result = model.forward(input.input_values.float())
            result = result.logits

        return result

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
        
        self.convR_1 = nn.Conv2d(in_channels=15, out_channels=1, kernel_size=1)

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
        x = torch.mean(x, dim=0, keepdim=True)
        
        # x = self.forward_stage1(x)
        # x = self.forward_stage2(x)
        # n_samples = x.shape[0] // self.im_per_sample
        # x = x.view(n_samples, self.im_per_sample, x.shape[1])
        # x = x.permute(0,2,1)
        #x = self.convR_1(x)

        # x = self.forward_classifier(x)
        # x = self.conv1(x)
        # x = self.maxpool(x)
        # x = self.modulator(self.stage2(x)) + self.local(x)
        # x = self.stage3(x)
        # x = self.stage4(x)
        # x = self.conv5(x)
        return x  

def init_feature_extractor(model, path):
    if path == 'None' or path is None:
        return
    checkpoint = torch.load(path, map_location=torch.device('cuda'))
    pre_trained_dict = checkpoint['state_dict']
    pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
    model.load_state_dict(pre_trained_dict, strict=False)


def get_model(num_classes, task, seq_length):
    model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, task, seq_length)
    return model  


def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding='valid'),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True), nn.MaxPool1d(2,1))


class AudioCNNPool(nn.Module):

    def __init__(self, num_classes=8):
        super(AudioCNNPool, self).__init__()

        input_channels = 10
        self.conv1d_0 = conv1d_block_audio(input_channels, 64)
        self.conv1d_1 = conv1d_block_audio(64, 128)
        self.conv1d_2 = conv1d_block_audio(128, 256)
        self.conv1d_3 = conv1d_block_audio(256, 128)
        
        self.classifier_1 = nn.Sequential(
                nn.Linear(128, num_classes),
            )
            
    def forward(self, x):
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x


    def forward_stage1(self,x):            
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x
    
    def forward_stage2(self,x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)   
        return x
    
    def forward_classifier(self, x):   
        x = x.mean([-1]) #pooling accross temporal dimension
        x1 = self.classifier_1(x)
        return x1

    

class Visual_Model(nn.Module):

    def __init__(self, num_classes=8, fusion='ia', seq_length=15, pretr_ef='./checkpoints/EfficientFace_Trained_on_AffectNet7.pth', num_heads=1):
        super(Visual_Model, self).__init__()
        assert fusion in ['ia', 'it', 'lt'], print('Unsupported fusion method: {}'.format(fusion))

        self.audio_model = AudioCNNPool(num_classes=num_classes)
        self.visual_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, seq_length)

        init_feature_extractor(self.visual_model, pretr_ef)
                           
        e_dim = 128
        input_dim_video = 128
        input_dim_audio = 128
        self.fusion=fusion

        if fusion in ['lt', 'it']:
            if fusion  == 'lt':
                self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim, num_heads=num_heads)
                self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim, num_heads=num_heads)
            elif fusion == 'it':
                input_dim_video = input_dim_video // 2
                self.av1 = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
                self.va1 = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)   
        
        elif fusion in ['ia']:
            input_dim_video = input_dim_video // 2
            
            self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
            self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)

            
        self.classifier_1 = nn.Sequential(
                    nn.Linear(e_dim*2, num_classes),
                )
        
            

    def forward_visual(self, x_visual):
        x_visual = self.visual_model.forward(x_visual)
        return x_visual 
    
    def forward(self, x_visual):
        x_visual = self.visual_model.forward(x_visual)
        return x_visual
        # if self.fusion == 'lt':
        #     return self.forward_transformer(x_audio, x_visual)

        # elif self.fusion == 'ia':
        #     return self.forward_feature_2(x_audio, x_visual)
       
        # elif self.fusion == 'it':
        #     return self.forward_feature_3(x_audio, x_visual)
        

 

class MultiModalCNN(nn.Module):
    def __init__(self):
        super(MultiModalCNN, self).__init__()
 
        #self.visual_model = Visual_Model(8, 'it', seq_length = 15, pretr_ef='./checkpoints/EfficientFace_Trained_on_AffectNet7.pth', num_heads=1)
        self.visual_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], 8, 15)
        self.visual_model = self.visual_model.to('cuda')
        init_feature_extractor(self.visual_model, './checkpoints/EfficientFace_Trained_on_AffectNet7.pth')


        


        self.audio_model = AutoModelForAudioClassification.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')
        self.audio_model = self.audio_model.to('cuda')
        self.audio_model, self.feature_extractor = weights_initialize(self.audio_model)

        
                                   
        # e_dim = 128
        # input_dim_video = 128
        # input_dim_audio = 128
       

        # self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim, num_heads=num_heads)
        # self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim, num_heads=num_heads)
  

        #self.summing_frames = nn.Linear(1024*15,1024)
        self.classifier_1 = nn.Sequential(
                    nn.Linear(1024*2, 8),
                )
        
            

    def forward(self, x_audio, x_visual):
        visual_start_time = time.time()
        # Set the model to evaluation mode
        self.visual_model.eval()
        
        x_visual = x_visual.to('cuda:0')
        # Perform inference
        with torch.no_grad():
            x_visual = self.visual_model(x_visual)
        
        visual_end_time = time.time()  # End the timer
        visual_elapsed_time =  visual_end_time -  visual_start_time  # Calculate elapsed time
        
        print("Visual Model took {:.4f} seconds".format( visual_elapsed_time))
        
            

        x_visual = torch.flatten(x_visual,  start_dim=1 , end_dim=-1 )
        
        #x_visual = self.summing_frames(x_visual)
        
        
        audio_start_time = time.time()
        x_audio = predict_emotion(x_audio, self.audio_model,self.feature_extractor) 
        audio_end_time = time.time()  # End the timer
        audio_elapsed_time =  audio_end_time -  audio_start_time  # Calculate elapsed time
        
        print("Audio Model took {:.4f} seconds".format( audio_elapsed_time))
        # print('///////////////////////////////////')
        # print(x_audio.size())
        # print('///////////////////////////////////')

        #x_audio = self.audio_model(bert_output, attention_mask, acoustic_input, acoustic_length, ctc_labels, emotion_labels)
           
        # proj_x_a = proj_x_a.permute(0, 2, 1)
        # proj_x_v = proj_x_v.permute(0, 2, 1)
        # h_av = self.av(proj_x_v, proj_x_a)
        # h_va = self.va(proj_x_a, proj_x_v)
       
        # audio_pooled = h_av.mean([1]) #mean accross temporal dimension
        # video_pooled = h_va.mean([1])
        x_audio = x_audio.to('cuda:0')
        #x_audio =  torch.flatten(x_audio)

       
        x = torch.cat((x_audio, x_visual), dim=-1) 

        x = x.to('cuda:0')

        x1 = self.classifier_1(x)
        return x1
 