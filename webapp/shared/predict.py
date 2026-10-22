"""Lazily-loaded, process-wide ClassicPitch instances, one per checkpoint.

Loading a model takes long enough to be worth doing once, but not at
import time — an app that only serves already-cached items never needs
one. Apps that let the user switch checkpoints (webapp/rubato) ask for
several over a session, and each stays loaded: the weights are a few
megabytes, and switching back and forth is the point of the control.
"""
import threading

from classic_pitch import ClassicPitch, DEFAULT_VARIANT, default_device

_classic_pitches = {}
_classic_pitch_lock = threading.Lock()


def get_classic_pitch(variant=DEFAULT_VARIANT):
    with _classic_pitch_lock:
        if variant not in _classic_pitches:
            device = default_device()
            print(f"[webapp] loading ClassicPitch '{variant}' on device={device} ...")
            _classic_pitches[variant] = ClassicPitch(device=device, variant=variant)
    return _classic_pitches[variant]
