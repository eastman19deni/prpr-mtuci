from .deepsort_wrapper import DeepSortTracker
from .centroid import CentroidTracker

def create_tracker(cfg=None):
    try:
        return DeepSortTracker()
    except Exception:
        return CentroidTracker()
