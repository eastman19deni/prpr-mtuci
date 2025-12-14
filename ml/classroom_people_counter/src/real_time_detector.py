import cv2
import time
from src.models.yolo_config import YoloModel
from src.utils.visualization import visualize_frame
from src.utils.file_manager import ensure_dirs
from src.utils.config import load_config

class RealTimeDetector:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config('config/app_config.yaml')
        ensure_dirs(self.cfg)
        self.yolo = YoloModel(self.cfg['model'])

    def run(self, source=0):
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open source {source}')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.yolo.detect(frame)
            out_frame, info = visualize_frame(frame, detections, self.cfg)

            cv2.imshow('Classroom People Counter', out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
