import cv2
import moviepy.editor as mp
import pandas as pd
import os
import re
from tqdm import tqdm
import config

def crop_video_by_time(video_path, bounding_box, output_path, start_time, end_time):
  """
  Crops a specific region from a video based on a bounding box and saves 
  frames between start_time (inclusive) and end_time (exclusive) as a new video.

  Args:
      video_path (str): Path to the AVI video file.
      bounding_box (tuple): A tuple of (x_min, y_min, x_max, y_max) defining the bounding box.
      output_path (str): Path to save the cropped video.
      start_time (float): Start time in seconds (inclusive).
      end_time (float): End time in seconds (exclusive).
  """

  # Open video capture
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print("Error opening video!")
    return

  # Get video properties
  fps = cap.get(cv2.CAP_PROP_FPS)

  # Calculate start and end frames based on time
  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  start_frame = int(start_time * fps)
  end_frame = int(end_time * fps)

  # Define bounding box coordinates
  x_min, y_min, x_max, y_max = bounding_box

  # Define codec for output video (adjust based on your needs)
  fourcc = cv2.VideoWriter_fourcc(*'XVID')

  # Create video writer object for cropped video
  out = cv2.VideoWriter(output_path, fourcc, fps, (x_max - x_min, y_max - y_min))
  if not out.isOpened():
    print("Error opening video writer!")
    cap.release()
    return

  # Process video frames
  frame_count = 0
  while True:
    ret, frame = cap.read()

    # Check if at the end of the video or exceeded end_frame
    if not ret or frame_count >= end_frame:
      break

    # Check if frame read successfully
    if not ret:
      print("Error reading frame!")
      continue

    # Skip frames before start_frame
    if frame_count < start_frame:
      frame_count += 1
      continue

    # Crop frame based on bounding box
    cropped_frame = frame[y_min:y_max, x_min:x_max]

    # Write cropped frame to output video
    out.write(cropped_frame)

    frame_count += 1

  # Release resources
  cap.release()
  out.release()
  cv2.destroyAllWindows()
  
  
def cut_audio_from_avi(video_path, output_path, start_time, end_time):
  """
  Cuts the audio from an AVI file between start_time (inclusive) and 
  end_time (exclusive) and saves it as a .wav file.

  Args:
      video_path (str): Path to the AVI video file.
      output_path (str): Path to save the cut audio as a .wav file.
      start_time (float): Start time in seconds (inclusive).
      end_time (float): End time in seconds (exclusive).
  """

  # Use MoviePy to load the video clip
  clip = mp.VideoFileClip(video_path)

  # Extract the audio from the video clip
  audio = clip.audio

  # Select the desired audio portion based on time
  cut_audio = audio.subclip(start_time, end_time)

  # Write the cut audio to a .wav file
  cut_audio.write_audiofile(output_path, verbose=False, logger=None)

  # print("Cut audio saved to", output_path)

# Example usage
video_path = r"D:\GP\Data\IEMOCAP_full_release\IEMOCAP_full_release\Session3\dialog\avi\DivX\Ses03F_impro01.avi"
output_path = "cut_audio.wav"
start_time = 5.0  # Start at 5 seconds
end_time = 15.0  # End at 10 seconds (exclusive)



# absolute path to dataset (change to the path on ypur machine)
PATH_TO_DATASET = config.IEMOCAP_RAW_DATASET_PATH
# relative pathes inside the data set 
WHOLE_VIDEO_PATH = r"\dialog\avi\DivX"
# Define path to your CSV file
CSV_PATH = config.CSV_PATH
# Read the CSV file into a pandas DataFrame
df = pd.read_csv(CSV_PATH)

AUDIO_PATH = config.AUDIO_PATH
VIDEO_PATH = config.VIDEO_PATH
# create a folder for the videos and audios
if not os.path.exists(AUDIO_PATH):
  os.makedirs(AUDIO_PATH)
if not os.path.exists(VIDEO_PATH):
  os.makedirs(VIDEO_PATH)



# Loop through each row in the DataFrame
for index, row in tqdm(df.iterrows()):
  # print(f"Name: {row['RecordingName']}")  # Access data using column name
  
  # 1. Determine whether its left or right cut
  splits = row['RecordingName'].split("_")
  n1, n2 = splits[0][-1], splits[-1][0]
  if (n1 == n2 ):
    # left bounding box
    bounding_box = (0,0,360,480)
  else:
    # right bounding box
    bounding_box = (361,0,720,480)
  # 2. Get the path of the avi file
  session = int(re.findall(r'\d+',splits[0])[0])
  session = f"Session{session}"
  video_name = "_".join(splits[:-1])+".avi"
  single_video_path = os.path.join(PATH_TO_DATASET,session,"dialog","avi","DivX", video_name)
  # print(single_video_path)
  
  start_time = row['Start']
  end_time = row['End']
  # 3. cut the video
  if not os.path.exists(os.path.join(VIDEO_PATH, row['RecordingName']+".avi")):
    crop_video_by_time(single_video_path, bounding_box, os.path.join(VIDEO_PATH, row['RecordingName']+".avi"), start_time, end_time)
  # 4. cut the audio
  if not os.path.exists(os.path.join(AUDIO_PATH, row['RecordingName']+".wav")):
    cut_audio_from_avi(single_video_path, os.path.join(AUDIO_PATH, row['RecordingName']+".wav"), start_time, end_time)