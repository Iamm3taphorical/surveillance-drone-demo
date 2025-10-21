import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger("tracker")

try:
    # Optional: prefer ByteTrack if installed (instructions in README)
    from bytetrack import ByteTrack  # placeholder import name — adjust if using a specific package
    BYTETRACK_AVAILABLE = True
except Exception:
    BYTETRACK_AVAILABLE = False

# Fallback: simple centroid tracker
class CentroidTracker:
    def __init__(self, max_lost=30):
        self.next_object_id = 1
        self.objects = {}          # object_id -> bbox
        self.lost = {}             # object_id -> lost_frames
        self.max_lost = max_lost

    def _centroid(self, bbox):
        x1,y1,x2,y2 = bbox
        return ((x1+x2)//2, (y1+y2)//2)

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        detections: list of dicts with xmin,ymin,xmax,ymax,conf,cls
        returns annotated list with 'id' field
        """
        if len(detections) == 0:
            # increase lost counter
            for oid in list(self.objects.keys()):
                self.lost[oid] = self.lost.get(oid, 0) + 1
                if self.lost[oid] > self.max_lost:
                    del self.objects[oid]; del self.lost[oid]
            return []

        input_centroids = [self._centroid((d['xmin'], d['ymin'], d['xmax'], d['ymax'])) for d in detections]
        if len(self.objects) == 0:
            # register all
            results = []
            for det, c in zip(detections, input_centroids):
                oid = self.next_object_id; self.next_object_id += 1
                self.objects[oid] = det
                self.lost[oid] = 0
                out = dict(det); out['id'] = oid
                results.append(out)
            return results

        # match by Euclidean distance
        object_ids = list(self.objects.keys())
        object_centroids = [self._centroid((self.objects[oid]['xmin'], self.objects[oid]['ymin'], self.objects[oid]['xmax'], self.objects[oid]['ymax'])) for oid in object_ids]

        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=float)
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i,j] = np.linalg.norm(np.array(oc)-np.array(ic))
        # greedy matching
        matched_rows, matched_cols = [], []
        results = []
        while D.size > 0:
            i,j = np.unravel_index(D.argmin(), D.shape)
            oid = object_ids[i]
            if D.min() > 100:  # distance threshold; if too far, break
                break
            # match object i with detection j
            det = detections[j]
            self.objects[oid] = det
            self.lost[oid] = 0
            out = dict(det); out['id'] = oid
            results.append(out)
            matched_rows.append(i); matched_cols.append(j)
            D = np.delete(D, i, axis=0)
            D = np.delete(D, j, axis=1)
            object_ids.pop(i); input_centroids.pop(j)
            if D.size == 0:
                break
        # register unmatched detections
        for j, det in enumerate(detections):
            if j in matched_cols:
                continue
            oid = self.next_object_id; self.next_object_id += 1
            self.objects[oid] = det
            self.lost[oid] = 0
            out = dict(det); out['id'] = oid
            results.append(out)
        # increase lost for unmatched objects left (object_ids left)
        for oid in list(self.objects.keys()):
            if oid not in [r['id'] for r in results]:
                self.lost[oid] = self.lost.get(oid, 0) + 1
                if self.lost[oid] > self.max_lost:
                    del self.objects[oid]; del self.lost[oid]
        return results


class TrackerManager:
    def __init__(self, use_bytetrack: bool = False):
        self.use_bytetrack = use_bytetrack and BYTETRACK_AVAILABLE
        if self.use_bytetrack:
            logger.info("Using ByteTrack (external dependency).")
            # instantiate ByteTrack here (pseudocode — depends on your chosen implementation)
            # self.tracker = ByteTrack(...)
            raise NotImplementedError("ByteTrack integration: configure your ByteTrack package here per README.")
        else:
            logger.info("Using fallback CentroidTracker.")
            self.tracker = CentroidTracker(max_lost=30)

    def update(self, detections: List[dict]) -> List[dict]:
        return self.tracker.update(detections)
