

# Import necessary libraries and modules
import os
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Add the parent directory to the system path
sys.path.insert(0, parent_dir)

from application import run
import numpy as np
from PIL import Image
import numpy as np
import os
import numpy as np          


from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
import functools
import time


def predict(video_name):
    start_time = time.time()

    video_audio_paths = []

    directory = f'./session/{video_name}/'
    for filename in os.listdir(directory):
        if filename.startswith('f') and filename.endswith('.mp4'):
            video_audio_paths.append(os.path.join(directory, filename))

    emotions = {}
    #print(video_audio_paths)

    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers based on your system
        future_to_video = {executor.submit(run, video): video for video in video_audio_paths}

        for future in as_completed(future_to_video):
            video = future_to_video[future]
            try:
                result = future.result()
                # Assuming run function returns an emotion prediction result
                video_id = os.path.splitext(os.path.basename(video))[0]
                index = video_id.find('f')
                if index != -1:
                    video_id = video_id[index + 1:]
                else:
                    video_id = ''
                print(video_id)
                emotions[video_id] = result
            except Exception as exc:
                print(f'{video} generated an exception: {exc}')
    
    emotions = dict(sorted(emotions.items()))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Full time taken {:.4f} seconds".format(elapsed_time))

    return emotions

# emotions = predict('2024-06-23_02-04-32')
# print(emotions)