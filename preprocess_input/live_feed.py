import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN
import time
import sounddevice as sd
import soundfile as sf

def extract_faces_from_camera():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    mtcnn = MTCNN(image_size=(224, 224), device=device)

    # Define processing parameters
    save_frames = 15
    save_length = 1  # seconds
    frame_rate = 30  # frames per second
    audio_rate = 22050  # audio sampling rate

    # Start capturing frames from the camera
    cap = cv2.VideoCapture(0)  # 0 for default camera, change if needed

    # Define the video codec and create a VideoWriter object for saving the output
    #out_video = cv2.VideoWriter(f'{video_name}.avi', cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (720, 1280))

    numpy_video = []
    frames_processed = 0
    audio_frames = []

    # Start recording audio
    recording = sd.rec(int(save_length * audio_rate), samplerate=audio_rate, channels=2, dtype='float32')

    start_time = time.time()


    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    start_time_all = time.time()
    boxes, _ = mtcnn.detect(rgb_frame)
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time 
     # Calculate elapsed time

    print("Face detection took {:.4f} seconds".format(elapsed_time))

    if boxes is not None:
        x1, y1, x2, y2 = map(int, boxes[0])
        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, (224, 224))
        #out_video.write(face)
        numpy_video.append(face)
        frames_processed += 1

        # Check if 15 frames have been processed
        if frames_processed == save_frames:
            #cap.release()
            #out_video.release()
            
            # Capture audio frames
            audio_frames.append(recording)
            audio_frames = np.concatenate(audio_frames)
            end_time_all = time.time()
            elapsed_time_all = end_time_all - start_time_all  # Calculate elapsed time

            print("faces found in:{:.4f} seconds".format(elapsed_time_all))

            return np.array(numpy_video), np.array(audio_frames)

            

    end_time_all = time.time()
    elapsed_time_all = end_time_all - start_time_all  # Calculate elapsed time

    print("No faces found in:{:.4f} seconds".format(elapsed_time_all))

    # Save audio to file
    # audio_frames.append(recording)
    # audio_frames = np.concatenate(audio_frames)
    # sf.write(audio_filename, audio_frames, audio_rate)

    # Release the camera and writer
    # cap.release()
    # out_video.release()

    return None, None
