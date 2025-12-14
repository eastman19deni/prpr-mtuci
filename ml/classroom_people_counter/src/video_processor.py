import cv2
from pathlib import Path
from src.models.yolo_config import YoloModel
from src.utils.visualization import visualize_frame
from src.utils.file_manager import ensure_dirs
from src.utils.config import load_config

class VideoProcessor:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config('config/app_config.yaml')
        ensure_dirs(self.cfg)
        self.yolo = YoloModel(self.cfg['model'])

    def process_video(self, video_path):
        vid = cv2.VideoCapture(str(video_path))
        if not vid.isOpened():
            raise RuntimeError(f'Cannot open video {video_path}')

        fps = vid.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_dir = Path(self.cfg['paths']['output']) / 'processed_videos'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / (Path(video_path).stem + '_processed.avi')
        writer = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*'XVID'), fps, (w,h))

        while True:
            ret, frame = vid.read()
            if not ret:
                break
            detections = self.yolo.detect(frame)
            out_frame, info = visualize_frame(frame, detections, self.cfg)
            writer.write(out_frame)

        writer.release()
        vid.release()
        print(f'Processed video saved to: {out_file}')
