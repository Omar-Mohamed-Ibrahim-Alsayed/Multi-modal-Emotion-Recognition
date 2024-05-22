import moviepy.editor as mp
import soundfile as sf
import numpy as np
import librosa

def extract_audios(video_name):
 
  input_video_path = f'./Examples/{video_name}.mp4'
  output_audio_path = f'./Examples/{video_name}.wav'
  preprocessed_audio_path = f'./Examples/{video_name}.wav'
  target_time = 1  # sec

  # Load the input video and extract audio
  video_clip = mp.VideoFileClip(input_video_path)
  audio_clip = video_clip.audio
  audio_clip.write_audiofile(output_audio_path, codec='pcm_s16le')  # Save as .wav file

  audios = librosa.core.load(output_audio_path, sr=22050)
    
  y = audios[0]
  sr = audios[1]
  target_length = int(sr * target_time)
  if len(y) < target_length:
      y = np.array(list(y) + [0 for i in range(target_length - len(y))])
  else:
      remain = len(y) - target_length
      y = y[remain//2:-(remain - remain//2)]


  # Save the original audio directly without modifications
  sf.write(preprocessed_audio_path, y, sr, format='WAV')
