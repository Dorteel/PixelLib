
from pixellib.torchbackend.instance import instanceSegmentation
import cv2

capture = cv2.VideoCapture(0)
model = r"C:\Users\dorte\Documents\Repositories\perceived-entity-typing\descriptors\pointrend_resnet50.pkl"

segment_video = instanceSegmentation()
segment_video.load_model(model, detection_speed = "fast")
segment_video.process_camera(capture,  show_bboxes = True, frames_per_second= 5, check_fps=True, show_frames= True,
frame_name= "frame")