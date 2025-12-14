import cv2
import numpy as np


class RealTimeDetector:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.desks = cfg['detection']['desks']

    def detect(self, frame):
        results = self.model(frame)

        detections = []

        for r in results:
            if not hasattr(r, 'boxes'):
                continue

            for box in r.boxes:
                cls = int(box.cls[0])

                if cls != 0:
                    continue

                conf = float(box.conf[0])
                if conf < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1

                cx = x1 + w // 2
                cy = y1 + int(h * 0.75)

                if h > 260:   
                    continue

                if not self._inside_any_desk(cx, cy):
                    continue

                detections.append({
                    'bbox': (x1, y1, w, h),
                    'confidence': conf,
                    'center': (cx, cy)
                })

        return detections

    def _inside_any_desk(self, cx, cy):
        for desk in self.desks:
            x1, y1, x2, y2 = desk['rect']
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        return False
