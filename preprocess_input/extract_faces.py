import os
import numpy as np          
import cv2
import torch
from facenet_pytorch import MTCNN
import time


def extract_faces(video_name):
  print(cv2.__version__)

  # Check for GPU availability
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Initialize MTCNN for face detection
  #80*80
  mtcnn = MTCNN(image_size=(720, 1280), device=device)

  # Path to the single video you want to process
  video_path = video_name

  # Define processing parameters
  save_frames = 15
  input_fps = 30
  save_length = 5  # seconds
  save_avi = True
  failed_videos = []

  # Select frames distribution lambda function
  select_distributed = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]

  # Open the video file for reading
  cap = cv2.VideoCapture(video_path)

  # Get total number of frames in the video
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  # Calculate the desired number of frames
  desired_frames = int(save_length * input_fps)

  # Skip frames if necessary
  if desired_frames < total_frames:
      skip_frames = int((total_frames - desired_frames) // 2)
      cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
  else:
      failed_videos.append("Video length is insufficient.")

  frames_to_select = select_distributed(save_frames, desired_frames)
  save_fps = save_frames // (desired_frames // input_fps)

  if save_avi:
      out = cv2.VideoWriter(f'{video_name}.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), save_fps, (224, 224))

  numpy_video = []
  frame_ctr = 0

  while True:
      ret, frame = cap.read()
      if not ret:
          break

      if frame_ctr not in frames_to_select:
          frame_ctr += 1
          continue
      else:
          frames_to_select.remove(frame_ctr)
          frame_ctr += 1

      try:
          # Convert to RGB for MTCNN
          rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          # Detect faces
          boxes, _ = mtcnn.detect(rgb_frame)
          if boxes is not None:
              x1, y1, x2, y2 = map(int, boxes[0])
              face = frame[y1:y2, x1:x2]
              face = cv2.resize(face, (224, 224))
              if save_avi:
                  out.write(face)
              numpy_video.append(face)
          else:
              # Handle when face detection fails
              numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))
      except Exception as e:
          # Handle any other exceptions during frame processing
          failed_videos.append(f"Error processing frame {frame_ctr}: {str(e)}")
          break

  # Add blank frames if necessary
  if len(frames_to_select) > 0:
      for _ in range(len(frames_to_select)):
          if save_avi:
              out.write(np.zeros((224, 224, 3), dtype=np.uint8))
          numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))

  # Release video capture and writer
  cap.release()
  if save_avi:
      out.release()

  # Save processed frames as numpy array
  video_name = os.path.splitext(video_name)[0]
  np.save(f'{video_name}.npy', np.array(numpy_video))

  # Check for any errors during video processing
  if len(numpy_video) != save_frames:
      failed_videos.append('Error: Processed frames do not match desired frame count.')

  # Handle failed videos or any other necessary cleanup
  if failed_videos:
      print('Failed videos:', failed_videos)

def extract_faces2(video_name):
  #print(cv2.__version__)
  # Check for GPU availability
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Initialize MTCNN for face detection
  mtcnn = MTCNN(image_size=(720, 1280), device=device)

  # Path to the single video you want to process
  video_path = f'./Application/{video_name}.mp4'

  # Define processing parameters
  save_frames = 15*8
  input_fps = 30
  save_length = 8 # seconds
  save_avi = True
  failed_videos = []

  # Select frames distribution lambda function
  select_distributed = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]

  # Open the video file for reading
  cap = cv2.VideoCapture(video_path)

  # Get total number of frames in the video
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  # Calculate the desired number of frames
  desired_frames = int(save_length * input_fps)

  # Skip frames if necessary
  if desired_frames < total_frames:
      skip_frames = int((total_frames - desired_frames) // 2)
      cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
  else:
      failed_videos.append("Video length is insufficient.")

  frames_to_select = select_distributed(save_frames, desired_frames)
  save_fps = save_frames // (desired_frames // input_fps)
  if save_avi:
    out = cv2.VideoWriter(f'./Application/{video_name}.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), save_fps, (224, 224))

  numpy_video = []
  frame_ctr = 0
  start_time_all = time.time()

  while True:
     ret, frame = cap.read()
     if not ret:
         break

     if frame_ctr not in frames_to_select:
         frame_ctr += 1
         continue
     else:
         frames_to_select.remove(frame_ctr)
         frame_ctr += 1

     try:
         # Convert to RGB for MTCNN
         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         # Detect faces
         start_time = time.time()
         boxes, _ = mtcnn.detect(rgb_frame)
         end_time = time.time()  # End the timer
         elapsed_time = end_time - start_time  # Calculate elapsed time
            
         print("Extraction of faces took {:.4f} seconds".format(elapsed_time))
         if boxes is not None:
             x1, y1, x2, y2 = map(int, boxes[0])
             face = frame[y1:y2, x1:x2]
             face = cv2.resize(face, (224, 224))
             if save_avi:
                  out.write(face)
             numpy_video.append(face)
         else:
             # Handle when face detection fails
             numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))
     except Exception as e:
         # Handle any other exceptions during frame processing
         failed_videos.append(f"Error processing frame {frame_ctr}: {str(e)}")
         break
  end_time_all = time.time() 
  elapsed_time_all = end_time_all - start_time_all  # Calculate elapsed time
            
  print("Extraction of faces took AS A WHOLE {:.4f} seconds".format(elapsed_time_all))
  # Add blank frames if necessary
  if len(frames_to_select) > 0:
      for _ in range(len(frames_to_select)):
          numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))

  
  # Release video capture and writer
  cap.release()
  if save_avi:
      out.release()
 
  # Check for any errors during video processing
  if len(numpy_video) != save_frames:
      if save_avi:
              out.write(np.zeros((224, 224, 3), dtype=np.uint8))
      failed_videos.append('Error: Processed frames do not match desired frame count.')

  # Handle failed videos or any other necessary cleanup
  if failed_videos:
      print('Failed videos:', failed_videos)
  
  return np.array(numpy_video)




