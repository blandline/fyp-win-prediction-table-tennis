"""
Root-level conftest: ensures the sort module is available for tests.
If the sort/ directory doesn't exist (gitignored), installs a minimal stub.
"""

import sys
import os
import types

_project_root = os.path.dirname(os.path.abspath(__file__))
_sort_dir = os.path.join(_project_root, 'sort')

if os.path.isdir(_sort_dir):
    # Real sort/ directory exists — add it to path
    if _sort_dir not in sys.path:
        sys.path.insert(0, _sort_dir)
elif 'sort' not in sys.modules:
    # sort/ directory is gitignored and not present — create a minimal stub
    # so that `from sort import Sort` works in tests
    import numpy as np

    class _StubSort:
        """Minimal SORT stub for testing. Passes detections through with auto-IDs."""
        def __init__(self, max_age=5, min_hits=2, iou_threshold=0.3):
            self.max_age = max_age
            self.min_hits = min_hits
            self._next_id = 1
            self._tracks = {}

        def update(self, dets):
            """
            dets: numpy array of shape (N, 5) — [x1,y1,x2,y2,conf].
            Returns numpy array of shape (M, 5) — [x1,y1,x2,y2,track_id].
            """
            if len(dets) == 0:
                # Age out existing tracks
                to_delete = []
                for tid, info in self._tracks.items():
                    info['age'] += 1
                    if info['age'] > self.max_age:
                        to_delete.append(tid)
                for tid in to_delete:
                    del self._tracks[tid]
                # Return surviving tracks
                results = []
                for tid, info in self._tracks.items():
                    if info['hits'] >= self.min_hits:
                        results.append([*info['bbox'], tid])
                return np.array(results) if results else np.empty((0, 5))

            results = []
            for det in dets:
                x1, y1, x2, y2, conf = det[:5]
                # Simple: assign new ID or match to existing by closest centre
                matched = False
                best_tid = None
                best_dist = float('inf')
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                for tid, info in self._tracks.items():
                    bx1, by1, bx2, by2 = info['bbox']
                    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
                    d = np.sqrt((cx - bcx)**2 + (cy - bcy)**2)
                    if d < best_dist:
                        best_dist = d
                        best_tid = tid

                if best_tid is not None and best_dist < 100:
                    self._tracks[best_tid]['bbox'] = [x1, y1, x2, y2]
                    self._tracks[best_tid]['age'] = 0
                    self._tracks[best_tid]['hits'] += 1
                    tid = best_tid
                else:
                    tid = self._next_id
                    self._next_id += 1
                    self._tracks[tid] = {'bbox': [x1, y1, x2, y2], 'age': 0, 'hits': 1}

                if self._tracks[tid]['hits'] >= self.min_hits:
                    results.append([x1, y1, x2, y2, tid])

            # Age out unmatched tracks
            for tid in list(self._tracks.keys()):
                if self._tracks[tid]['age'] > self.max_age:
                    del self._tracks[tid]

            return np.array(results) if results else np.empty((0, 5))

    # Register the stub module
    sort_module = types.ModuleType('sort')
    sort_module.Sort = _StubSort
    sys.modules['sort'] = sort_module


# ---------------------------------------------------------------------------
# Stub for ultralytics if not installed (unit tests use mock models)
# ---------------------------------------------------------------------------
try:
    import ultralytics  # noqa: F401
except ImportError:
    _ul_module = types.ModuleType('ultralytics')

    class _StubYOLO:
        """Stub YOLO class — raises RuntimeError if actually called."""
        def __init__(self, model_path=None, **kwargs):
            self._path = model_path

        def __call__(self, *args, **kwargs):
            raise RuntimeError(
                f"StubYOLO called with real inference. "
                f"Install ultralytics or use MockYOLOModel in tests."
            )

    _ul_module.YOLO = _StubYOLO
    sys.modules['ultralytics'] = _ul_module
