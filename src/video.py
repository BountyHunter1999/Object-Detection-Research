import cv2
import random
from ultralytics import YOLO

# load yolo model
"""
https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8s.pt

yolov8n.pt: Nano (fastest, less accurate)
yolov8s.pt: Small (moderate accuracy, moderate speed)
yolov8x.pt: Extra Large (most accurate, slower)
"""
yolo = YOLO("https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8s.pt")

def getColors(cls_num):
    """Generate unique colors for each class ID"""
    random.seed(cls_num)
    return tuple[int, ...](random.randint(0, 255) for _ in range(3))

def load_video(video_path):
    videoCapture = cv2.VideoCapture(video_path)
    if not videoCapture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    return videoCapture

def process_video(video_path, frame_count=0, threshold=0.4):
    videoCap = load_video(video_path)
    while True:
        # reads one frame from video
        ret, frame = videoCap.read()
        if not ret:
            break
        
        # stream: true,treats the input source as a continuous video stream.
        # Run yolo object detection on the frame
        results = yolo.track(frame, stream=True)
        
        for result in results:
            class_names = result.names
            
            # result.boxes: contains bounding boxes for detected objects
            for box in result.boxes:
                # box.conf[0]: confidence score for the detected object
                if box.conf[0] > threshold:
                    # box.xyxy[0]: bounding box coordinates for the detected object
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # box.cls[0]: class ID for the detected object
                    cls_id = int(box.cls[0])
                    
                    class_name = class_names[cls_id]
                    
                    conf = float(box.conf[0])

                    color = getColors(cls_id)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if frame_count < 20:
            cv2.imshow("Frame", frame)
        else:
            break
        
        frame_count += 1
        
    videoCap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    process_video("data/videos/sample.mp4")