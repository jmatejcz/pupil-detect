import cv2
import pandas as pd

def get_frames_from_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        res=(int(width), int(height))
        while True:
            try:
                is_success, frame = cap.read()
            except cv2.error:
                print("err")
                continue
            if not is_success:
                print("can't receive frame")
                break 
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(image)
    cap.release()
    return frames

def get_labels_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.drop(['p', 'w', 'h'], axis=1, inplace=True)
    opened = [1 if x>0 else 0 for x in df['x']]
    df.rename(columns={"x":"pupil_center_x_coord", "y":"pupil_center_y_coord", "a":"corneal_reflection_x_coord", "b":"corneal_reflection_y_coord"}, inplace=True)
    df['opened'] = opened
    return df