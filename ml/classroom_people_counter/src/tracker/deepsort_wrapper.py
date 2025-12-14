try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception as e:
    raise

class DeepSortTracker:
    def __init__(self, max_age=30):
        self.ds = DeepSort(max_age=max_age)

    def update_tracks(self, detections, frame=None):
        boxes = []
        for d in detections:
            x,y,w,h = d['bbox']
            boxes.append(([x,y,x+w,y+h], d['score'], 'person'))
        tracks = self.ds.update_tracks(boxes, frame=frame)
        result = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            ltrb = t.to_ltrb()
            x1,y1,x2,y2 = map(int, ltrb)
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w//2
            cy = y1 + h//2
            result.append({'track_id': tid, 'bbox':[x1,y1,w,h], 'center':(cx,cy)})
        return result
