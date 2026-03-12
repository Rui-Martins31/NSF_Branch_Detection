import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Constants
IMAGE_COLOR_LENGTH: int = 1280
IMAGE_COLOR_WIDTH: int  = 720

IMAGE_DEPTH_LENGTH: int = 640
IMAGE_DEPTH_WIDTH: int  = 480

CAMERA_FPS: float       = 30.0

# Configure and start streaming
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.depth, IMAGE_DEPTH_LENGTH, IMAGE_DEPTH_WIDTH, rs.format.z16, 30)
config.enable_stream(rs.stream.color, IMAGE_COLOR_LENGTH, IMAGE_COLOR_WIDTH, rs.format.bgr8, 30)

# Initialize VideoWriter for MP4 output
os.makedirs('recordings', exist_ok=True)
fourcc    = cv2.VideoWriter_fourcc(*'mp4v')
out_rgb   = cv2.VideoWriter(
    filename  = 'recordings/realsense_capture_rgb.mp4',
    fourcc    = fourcc,
    fps       = CAMERA_FPS,
    frameSize = (IMAGE_COLOR_LENGTH, IMAGE_COLOR_WIDTH),
    isColor   = True
)
out_depth  = cv2.VideoWriter(
    filename  = 'recordings/realsense_capture_depth.mp4',
    fourcc    = fourcc,
    fps       = CAMERA_FPS,
    frameSize = (IMAGE_DEPTH_LENGTH, IMAGE_DEPTH_WIDTH),
    isColor   = False
)
out_diff  = cv2.VideoWriter(
    filename  = 'recordings/realsense_capture_diff.mp4',
    fourcc    = fourcc,
    fps       = CAMERA_FPS,
    frameSize = (IMAGE_DEPTH_LENGTH, IMAGE_DEPTH_WIDTH),
    isColor   = True
)

pipeline_started = False
cv2.namedWindow('RealSense Video', cv2.WINDOW_NORMAL)
try:
    pipeline.start(config)
    pipeline_started = True
    print("Recording... press Ctrl+C to stop.")
    while True:
        frames      = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if (not color_frame) or (not depth_frame): continue

        color_image = np.asanyarray(color_frame.get_data()).copy()
        depth_image = np.asanyarray(depth_frame.get_data()).copy()

        out_rgb.write(color_image)
        out_depth.write(depth_image)

        images = np.hstack((color_image, depth_image))
        cv2.imshow('RealSense Video', images)

except KeyboardInterrupt:
    pass

finally:
    if pipeline_started:
        pipeline.stop()
    out_rgb.release()
    out_depth.release()
    out_diff.release()