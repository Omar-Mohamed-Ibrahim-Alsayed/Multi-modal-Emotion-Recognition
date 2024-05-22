# Import necessary libraries and modules
import os
import json
import torch
from torch import nn
from PIL import Image
import functools
import numpy as np
import librosa
from torch.autograd import Variable
from opts import parse_opts
from model import generate_model
from models.fusion5 import MultiModalCNN
from torch.nn.parallel import DataParallel
import transforms
from preprocess_input import extract_faces,extract_audios
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

video_name="original"

def video_loader(video_dir_path):
    video = np.load(video_dir_path)  
    video_data = []
    for i in range(np.shape(video)[0]):
        video_data.append(Image.fromarray(video[i,:,:,:]))   
    return video_data

def get_default_video_loader():
    return functools.partial(video_loader)

def load_audio(audiofile):
    y, sr = librosa.load(audiofile, sr=22050)
    return y, sr


def get_mfccs(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    return mfcc

def input(data_type,video_path,audio_path,spatial_transform=None, audio_transform=None):
    loader = get_default_video_loader()       
    path = video_path
    clip = loader(path)

    if data_type == 'video' or data_type == 'audiovisual':        
        if spatial_transform is not None:               
            spatial_transform.randomize_parameters()
            clip = [spatial_transform(img) for img in clip]  

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
                
        if data_type == 'video':
            return clip
                
    if data_type == 'audio' or data_type == 'audiovisual':
            path = audio_path

            y, sr = load_audio(path) 
            
            if audio_transform is not None:
                  audio_transform.randomize_parameters()
                  y = audio_transform(y)     
                 
            mfcc = get_mfccs(y, sr)            
            audio_features = mfcc 

            if data_type == 'audio':
                return audio_features
            
    if data_type == 'audiovisual':
            return audio_features, clip

def getinput(video_path, audio_path, modality='both'):
    
    input_video_path = video_path # Replace with your video path
    input_audio_path = audio_path

    video_transform = transforms.Compose([
                transforms.ToTensor(255)])
    
    inputs_audio,inputs_visual = input('audiovisual', input_video_path, input_audio_path, video_transform)
    modality = 'both'
    if modality == 'audio':
            print('Skipping video modality')
            if dist == 'noise':
                print('Evaluating with full noise')
                inputs_visual = torch.randn(inputs_visual.size())
            elif dist == 'addnoise': #opt.mask == -4:
                print('Evaluating with noise')
                inputs_visual = inputs_visual + (torch.mean(inputs_visual) + torch.std(inputs_visual)*torch.randn(inputs_visual.size()))
            elif dist == 'zeros':
                inputs_visual = torch.zeros(inputs_visual.size())
            else:
                print('UNKNOWN DIST!')
    elif modality == 'video':
            print('Skipping audio modality')
            if dist == 'noise':
                print('Evaluating with noise')
                inputs_audio = torch.randn(inputs_audio.size())
            elif dist == 'addnoise': #opt.mask == -4:
                print('Evaluating with added noise')
                inputs_audio = inputs_audio + (torch.mean(inputs_audio) + torch.std(inputs_audio)*torch.randn(inputs_audio.size()))

            elif dist == 'zeros':
                inputs_audio = torch.zeros(inputs_audio.size())
    a = []
    a.append(inputs_visual)
    a = torch.stack(a, dim=0)
    inputs_visual = a
    #inputs_visual = inputs_visual.unsqueeze(0)
    #inputs_visual = inputs_visual.repeat(8, 1, 1, 1, 1)
    inputs_visual = inputs_visual.permute(0,2,1,3,4)
    inputs_visual = inputs_visual.reshape(inputs_visual.shape[0]*inputs_visual.shape[1], inputs_visual.shape[2], inputs_visual.shape[3], inputs_visual.shape[4])
    inputs_audio = np.array([inputs_audio])
    inputs_audio = torch.tensor(inputs_audio)
    #inputs_audio = np.array([inputs_audio,inputs_audio,inputs_audio,inputs_audio,inputs_audio,inputs_audio,inputs_audio,inputs_audio])
    with torch.no_grad():
        inputs_visual = Variable(inputs_visual)
        inputs_audio = Variable(inputs_audio)

    return  inputs_audio,inputs_visual   

def print_emotion(output_tensor):
    emotions = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]

    # Use softmax to convert tensor values to probabilities
    probabilities = torch.softmax(output_tensor, dim=1)

    # Get the index of the maximum probability
    max_index = torch.argmax(probabilities, dim=1).item()

    # Print the corresponding emotion
    print(f"Predicted Emotion: {emotions[max_index]}")



def predict(video_name):
        # Parse command-line options
    opt = parse_opts()

    pretrained = opt.pretrain_path != 'None' 

    # Set the device to 'cuda' if available, otherwise use 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}')

    # Create a directory to store results if it doesn't exist
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)


    
    video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])
    
    # Generate the model and its parameters
    torch.manual_seed(opt.manual_seed)
    model1, parameters = generate_model(opt)

    
    fold = 1

    model = MultiModalCNN()
    model.to('cuda')


    # Load a single input video (you can replace this with your specific input path)
    video_audio_paths = [
            (f'../input/{video_name}.npy',f'../input/{video_name}.wav')
        
    ]
    extract_faces.extract_faces(video_name)
    extract_audios.extract_audios(video_name)

    # Loop over each pair of video and audio paths
    for video_path, audio_path in video_audio_paths:
        # Load the model and perform other initializations here if necessary
        

        # Load a single input video and audio
        audio, clip = getinput(video_path, audio_path, modality='both')

        # Set the model to evaluation mode
        model.eval()

        # Perform inference
        with torch.no_grad():
            output = model(f'../input/{video_name}.wav', clip)

        print(output.data)

        print_emotion(output.data)

class AnnotationReader:
    def __init__(self):
        self.annotation_file = './ravdess_preprocessing/annotations.txt'
        with open(self.annotation_file, 'r') as file:
            self.annotations = file.readlines()
        self.current_indices = {"training": 0, "testing": 0, "validation": 0}

    def get_batch(self, split_type, batch_size=500):
        batch_data = []
        current_index = self.current_indices.get(split_type, 0)

        while len(batch_data) < batch_size and current_index < len(self.annotations):
            line = self.annotations[current_index].strip().split(';')

            if line[-1] == split_type and (current_index < 1029 or current_index > 1033):
                first_path = line[0]
                second_path = line[1]
                first_path =  first_path[:]
                second_path =  second_path[:]
                label = int(line[2]) - 1
                batch_data.append((first_path, second_path, label))

            current_index += 1

        self.current_indices[split_type] = current_index
        return batch_data

    def reset_training_index(self):
        self.current_indices["training"] = 0

    def reset_validation_index(self):
        self.current_indices["validation"] = 0       

annotation_reader = AnnotationReader()

def get_accuracy(model, data_type ,annotation_reader):
    model.eval()
    correct = 0

    with torch.no_grad():
        annotation_reader.reset_validation_index()
        c  = annotation_reader.get_batch(data_type)
        i=0
        total = len(all_data)
        for data in all_data:
            i+=1
            if(i%100==0):
              print(i)
            video0, audio, label = data[0], data[1], data[2]
            aud, video = getinput(video0, audio, modality='both')
            outputs = model(audio, video)
            _, predicted = torch.max(outputs.data, 1)
            label_tensor = torch.tensor([label], dtype=torch.long).to(outputs.device)
          
            
            correct += (predicted == label_tensor).sum().item()
    accuracy = correct / total
    return accuracy

def plot_accuracies(validation_accuracies):
    epochs = list(range(1, len(validation_accuracies) + 1))

    plt.plot(epochs, validation_accuracies, label='Validation Accuracy')

    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()


def train():
  
  model = MultiModalCNN()
  model.cuda()

  #model.load_state_dict(torch.load('model2.pth'))

  optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
  scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler
  criterion = torch.nn.CrossEntropyLoss()

  num_epochs = 100
  batch_size = 457

  max_gradient_norm = 1.0  # Maximum gradient norm for gradient clipping
  max_validation_acc=-1
  validation_accuracies=[]
  for epoch in range(num_epochs):
      model.train()
      
      for i in range (4):
        #print('batch number ' + str(i+1) + ' ended')
        batch_data = annotation_reader.get_batch("training",batch_size)
        for data in batch_data:
            video0, audio, label = data[0], data[1], data[2]
            aud, video = getinput(video0, audio, modality='both')

            optimizer.zero_grad()
            outputs = model(audio, video)
            label_tensor = torch.tensor([label], dtype=torch.long).to(outputs.device)
            loss = criterion(outputs, label_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_gradient_norm)

            optimizer.step()
      print("--------------------------------------------\n")      
      annotation_reader.reset_training_index()
      acc = get_accuracy(model,"validation" ,annotation_reader)
      validation_accuracies.append(int(acc * 100))
      if acc>max_validation_acc:
        max_validation_acc=acc
        torch.save(model.state_dict(), 'model5_copy.2pth')
        print('Saving model')
      print(f'Validation Accuracy: {acc * 100:.2f}%')
      scheduler.step()  # Update learning rate after each epoch 
      print("\nEpoch number: " + str(epoch + 1) +" ENDED\n")  

  plot_accuracies(validation_accuracies)
  plt.savefig('plot.png')


def print_test_acc():  
  model2 = MultiModalCNN()
  model2.load_state_dict(torch.load('model5 - Copy.pth'))
  model2.to('cuda')
  acc = get_accuracy(model2,"testing" ,annotation_reader)    
  print(f'Testing Accuracy: {acc * 100:.2f}%')

#train()
print_test_acc()  
