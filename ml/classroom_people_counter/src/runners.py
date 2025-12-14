import cv2
from src.models.yolo_config import YoloModel
from src.tracker.factory import create_tracker
from src.utils.visualization import visualize_frame_with_ids
from src.utils.file_manager import ensure_dirs
from src.utils.stats_logger import StatsLogger

class BaseRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        ensure_dirs(cfg)
        self.yolo = YoloModel(cfg['model'])
        self.tracker = create_tracker(cfg.get('tracker', {}))
        self.logger = StatsLogger(cfg['paths'].get('db'))

class RealtimeRunner(BaseRunner):
    def run(self, source=0):
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open source {source}')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = self.yolo.detect(frame)
            tracks = self.tracker.update_tracks(detections, frame=frame)
            out_frame, info = visualize_frame_with_ids(frame, tracks, self.cfg)
            self.logger.log_frame(info)
            cv2.imshow('Classroom People Counter', out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

class VideoRunner(BaseRunner):
    def run(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video {video_path}')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_dir = self.cfg['paths']['output'] + '/processed_videos'
        from pathlib import Path
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_file = f"{out_dir}/{Path(video_path).stem}_processed.avi"
        writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'XVID'), fps, (w,h))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = self.yolo.detect(frame)
            tracks = self.tracker.update_tracks(detections, frame=frame)
            out_frame, info = visualize_frame_with_ids(frame, tracks, self.cfg)
            self.logger.log_frame(info)
            writer.write(out_frame)
            cv2.imshow('Classroom People Counter', out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        writer.release()
        cap.release()
        cv2.destroyAllWindows()
