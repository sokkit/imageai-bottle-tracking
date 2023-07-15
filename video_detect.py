import os
import cv2
from imageai.Detection import VideoObjectDetection
camera = cv2.VideoCapture(0)

execution_path = os.getcwd()
 
detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
# detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel(detection_speed="fastest")
 
video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "camera_recording.mp4"),
output_file_path=os.path.join(execution_path, "objects_detection")
,frames_per_second=20, log_progress=True)
 
# video_path = detector.detectObjectsFromVideo(camera_input=camera,
#     output_file_path=os.path.join(execution_path, "camera_detected_video")
#     , frames_per_second=20, log_progress=True, minimum_percentage_probability=30) 
print(video_path)