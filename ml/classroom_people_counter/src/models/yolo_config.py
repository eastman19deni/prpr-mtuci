import cv2
import numpy as np
from pathlib import Path

class YoloModel:
    def __init__(self, cfg):
        self.cfg_path = cfg.get('cfg_path')
        self.weights_path = cfg.get('weights_path')
        self.names_path = cfg.get('names_path')
        self.conf_threshold = cfg.get('conf_threshold', 0.5)
        self.nms_threshold = cfg.get('nms_threshold', 0.4)
        self.net = cv2.dnn.readNetFromDarknet(self.cfg_path, self.weights_path)
        self.classes = self._load_names(self.names_path)
        self.output_layers = self._get_output_layers()

    def _load_names(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return [l.strip() for l in f.readlines()]

    def _get_output_layers(self):
        layer_names = self.net.getLayerNames()
        try:
            return [layer_names[i[0]-1] for i in self.net.getUnconnectedOutLayers()]
        except Exception:
            return [layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, frame):
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        boxes, confidences, class_ids = [], [], []
        for out in outs:
            for det in out:
                scores = det[5:]
                cls_id = int(np.argmax(scores))
                conf = float(scores[cls_id])
                if conf > self.conf_threshold and cls_id < len(self.classes) and self.classes[cls_id]=='person':
                    cx = int(det[0]*W)
                    cy = int(det[1]*H)
                    w = int(det[2]*W)
                    h = int(det[3]*H)
                    x = int(cx - w/2)
                    y = int(cy - h/2)
                    boxes.append([x,y,w,h])
                    confidences.append(conf)
                    class_ids.append(cls_id)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        detections = []
        if len(idxs)>0:
            for i in idxs.flatten():
                x,y,w,h = boxes[i]
                detections.append({'bbox':[x,y,w,h],'score':float(confidences[i])})
        return detections
