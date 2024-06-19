# -*- coding: utf-8 -*-
import os
import numpy as np          
import cv2
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN
import config
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


mtcnn = MTCNN(image_size=(720, 1280), device=device)

#mtcnn.to(device)
save_frames = 15
input_fps = 30

save_length = 3.6 #seconds
save_avi = True

failed_videos = []

# new path to the data clips
root = config.VIDEO_PATH
cropped_avi_root = config.FACE_EXTRACTED_AVI
cropped_np_root = config.FACE_EXTRACTED_NPM

# create cropped_roots if not found
if not os.path.exists(cropped_np_root):
    os.makedirs(cropped_np_root)
if not os.path.exists(cropped_avi_root):
    os.makedirs(cropped_avi_root)

select_distributed = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
n_processed = 0
for filename in tqdm(sorted(os.listdir(root))):         
    if filename.endswith('.avi'):
        
        # skip the extracted faces already
        if os.path.exists(os.path.join(cropped_np_root, filename.replace('.avi', '.npy'))) and \
        os.path.exists(os.path.join(cropped_avi_root, filename)):
            # print("true")
            continue
                    
        cap = cv2.VideoCapture(os.path.join(root, filename))  
        #calculate length in frames
        framen = 0
        while True:
            i,q = cap.read()
            if not i:
                break
            framen += 1
        cap = cv2.VideoCapture(os.path.join(root, filename))

        if save_length*input_fps > framen:                    
            skip_begin = int((framen - (save_length*input_fps)) // 2)
            for i in range(skip_begin):
                _, im = cap.read() 
                
        framen = int(save_length*input_fps)    
        frames_to_select = select_distributed(save_frames,framen)
        save_fps = save_frames // (framen // input_fps) 
        if save_avi:
            out = cv2.VideoWriter(os.path.join(cropped_avi_root, filename[:-4]+'.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), save_fps, (224,224))

        numpy_video = []
        success = 0
        frame_ctr = 0
        
        while True: 
            ret, im = cap.read()
            if not ret:
                break
            if frame_ctr not in frames_to_select:
                frame_ctr += 1
                continue
            else:
                frames_to_select.remove(frame_ctr)
                frame_ctr += 1

            try:
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            except:
                failed_videos.append((filename, i))
                break
    
            temp = im[:,:,-1]
            im_rgb = im.copy()
            im_rgb[:,:,-1] = im_rgb[:,:,0]
            im_rgb[:,:,0] = temp
            im_rgb = torch.tensor(im_rgb)
            im_rgb = im_rgb.to(device)

            bbox = mtcnn.detect(im_rgb)
            if bbox[0] is not None:
                bbox = bbox[0][0]
                bbox = [round(x) for x in bbox]
                x1, y1, x2, y2 = bbox
            im = im[y1:y2, x1:x2, :]
            print(filename)
            print(bbox)
            print(im.size)
            im = cv2.resize(im, (224,224))
            if save_avi:
                out.write(im)
            numpy_video.append(im)
        if len(frames_to_select) > 0:
            for i in range(len(frames_to_select)):
                if save_avi:
                    out.write(np.zeros((224,224,3), dtype = np.uint8))
                numpy_video.append(np.zeros((224,224,3), dtype=np.uint8))
        if save_avi:
            out.release() 
        np.save(os.path.join(cropped_np_root, filename[:-4]+'.npy'), np.array(numpy_video))
        if len(numpy_video) != 15:
            print('Error', filename)    
                        
n_processed += 1      
print(failed_videos)
