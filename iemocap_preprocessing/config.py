import os
# Set to iemocap raw dataset path and desired path for processed data
IEMOCAP_RAW_DATASET_PATH = r"D:\GP\Data\IEMOCAP_full_release\IEMOCAP_full_release"
IEMOCAP_PROCESSED_DATASET_PATH = r"D:\GP\Data\IEMOCAP_full_release\Data_Prep"

CSV_PATH = os.path.join(IEMOCAP_PROCESSED_DATASET_PATH, "Annotations.csv")
VIDEO_PATH = os.path.join(IEMOCAP_PROCESSED_DATASET_PATH, "video")
AUDIO_PATH = os.path.join(IEMOCAP_PROCESSED_DATASET_PATH, "audio")

FACE_EXTRACTED_AVI = os.path.join(IEMOCAP_PROCESSED_DATASET_PATH, "face_avi")
FACE_EXTRACTED_NPM = os.path.join(IEMOCAP_PROCESSED_DATASET_PATH, "face_npy")
