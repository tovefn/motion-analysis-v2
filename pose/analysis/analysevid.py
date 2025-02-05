import os
from argparse import ArgumentParser, ArgumentTypeError
import cv2
from mmdet.apis import (inference_detector, init_detector, show_result)
from mmpose.apis import (inference_topdown, init_model, vis_pose_result)
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

base = r'C:\Users\tovef\Documents\Exjobb\POEs'


def rotate_video(video_path):
    # Initialize the model with the correct config and checkpoint file
    config_file = r'C:\Users\tovef\Documents\Exjobb\POEs\mmdetection-master\configs\faster_rcnn\faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = r'C:\Users\tovef\Documents\Exjobb\POEs\faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-5324cff8.pth'
    
    model = init_detector(config_file, checkpoint_file, device='cpu')  # Or use 'cuda' if GPU available

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open video.")
        return
    
    # Get video details (frame width, height, and number of frames)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a VideoWriter to save the rotated video output
    output_video_path = 'rotated_' + video_path.split('/')[-1]
    new_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (frame_width, frame_height))
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Perform inference on the current frame
        result = inference_detector(model, frame)
        bbox_p = result[0]  # Assuming it's the first class

        # Check each bounding box (you can add further conditions for rotation if necessary)
        for bbox in bbox_p:
            x_min, y_min, x_max, y_max, score = bbox
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            if bbox_width > bbox_height:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate frame if bbox is wider than tall

        # Write the processed (and possibly rotated) frame to the output video
        new_video.write(frame)

        # Optional: Display the frame (this is useful for debugging)
        cv2.imshow('Processed Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop early
            break

    # Release resources
    cap.release()
    new_video.release()
    cv2.destroyAllWindows()

    print(f"Processed video saved as {output_video_path}")

if __name__ == "__main__":
    video_path = r'C:\Users\tovef\Documents\Exjobb\POEs\Friska ungdomar\2013_01_2.mp4'  # Correct path to your video
    rotate_video(video_path)
