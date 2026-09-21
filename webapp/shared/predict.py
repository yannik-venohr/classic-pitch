"""Lazily-loaded, process-wide ClassicPitch singleton.

Loading the model takes long enough to be worth doing once, but not at
import time — an app that only serves already-cached items never needs it.
"""
import threading

import torch

from predictor import ClassicPitch

_predictor = None
_predictor_lock = threading.Lock()


def get_predictor():
    global _predictor
    with _predictor_lock:
        if _predictor is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"[webapp] loading ClassicPitch model on device={device} ...")
            _predictor = ClassicPitch(device=device)
    return _predictor
