import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 1
        self.objects = {}  
        self.bboxes = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def _centroid(self, bbox):
        x,y,w,h = bbox
        return (int(x + w/2), int(y + h/2))

    def update_tracks(self, detections, frame=None):
        input_centroids = []
        bboxes = []
        for d in detections:
            bboxes.append(d['bbox'])
            input_centroids.append(self._centroid(d['bbox']))
        if len(self.objects)==0:
            for c,b in zip(input_centroids,bboxes):
                oid = self.next_object_id
                self.objects[oid] = c
                self.bboxes[oid] = b
                self.disappeared[oid] = 0
                self.next_object_id += 1
            return [{'track_id': k, 'bbox': self.bboxes[k], 'center': self.objects[k]} for k in self.objects]
        obj_ids = list(self.objects.keys())
        obj_centroids = list(self.objects.values())
        if len(obj_centroids)==0:
            for c,b in zip(input_centroids,bboxes):
                oid = self.next_object_id
                self.objects[oid] = c
                self.bboxes[oid] = b
                self.disappeared[oid] = 0
                self.next_object_id += 1
            return [{'track_id': k, 'bbox': self.bboxes[k], 'center': self.objects[k]} for k in self.objects]
        D = np.linalg.norm(np.array(obj_centroids)[:,None] - np.array(input_centroids)[None,:], axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_cols = set()
        matched_obj_ids = set()
        for r,c in zip(rows,cols):
            if c in used_cols:
                continue
            oid = obj_ids[r]
            self.objects[oid] = input_centroids[c]
            self.bboxes[oid] = bboxes[c]
            self.disappeared[oid] = 0
            used_cols.add(c)
            matched_obj_ids.add(oid)
        for i in range(len(input_centroids)):
            if i not in used_cols:
                oid = self.next_object_id
                self.objects[oid] = input_centroids[i]
                self.bboxes[oid] = bboxes[i]
                self.disappeared[oid] = 0
                self.next_object_id += 1
        for oid in list(self.objects.keys()):
            if oid not in matched_obj_ids:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.bboxes[oid]
                    del self.disappeared[oid]
        return [{'track_id': k, 'bbox': self.bboxes[k], 'center': self.objects[k]} for k in self.objects]
