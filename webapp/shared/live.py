"""Running the model over one short window of audio, for the live app.

The offline path (``predictor.ClassicPitch.predict``) is built around a
whole file: load it, transform it, run the model over it in 1000-frame
chunks, decode the lot into notes. Live transcription wants the same
model applied the other way round — over the last few seconds of a
microphone, as often as the machine can manage — so this module is the
windowed counterpart. It deliberately uses the same singleton the other
apps do (``webapp.shared.predict.get_predictor``), so the live view and
the file views are the same weights on the same features.

Two things are done differently from the offline path, both because the
window is short:

* The model is called directly rather than through
  ``predict_full_track``, which pads whatever it is handed out to 1000
  frames. For a 6-second window (259 frames at hop 512) that padding is
  roughly four times the work, all of it on silence — measured at 274ms
  vs. 61ms per window on an M-series GPU.

* The CQT's tuning is pinned instead of estimated per call.
  ``librosa.estimate_tuning`` over a 6-second window wanders by 0.04 to
  0.10 semitones from window to window on real recordings, with
  occasional half-semitone excursions on near-silence. A CQT bin here is
  a third of a semitone, so that is the whole frequency grid sliding
  under the model between consecutive windows — which blurs exactly the
  averaging the live view depends on. A fixed grid is worth more than a
  per-window estimate at this length.

Nothing is decoded into note events here: the app sends the two
posteriograms to the browser, which decodes them with a port of the file
apps' decoder. It has to happen there, because only the browser holds the
average over overlapping windows that is worth decoding — and it keeps
the thresholds live controls and the server a pure function of the audio
it is given.
"""
import base64
import threading

import librosa
import numpy as np
import torch

from src.data.utils import audio_to_hcqt
from src.postprocess.postprocess import MIDI_OFFSET
from webapp.shared.predict import get_predictor

# Pinned tuning for the live CQT: 0.0 is A=440. See the module docstring.
LIVE_TUNING = 0.0

# The model's two heads, in the order the frontend expects them.
HEADS = ("note", "onset")

# One window through the GPU at a time. Two browser tabs (or a reload
# that leaves a request in flight) would otherwise have Flask's threads
# enter the same model concurrently, which on MPS is both unsupported
# and slower than taking turns.
_infer_lock = threading.Lock()


def predict_window(audio, sr):
    """The two posteriograms for one window of mono audio.

    Returns ``(outputs, axes)``: ``outputs`` maps head name to a
    ``(n_pitches, n_frames)`` array of probabilities, and ``axes``
    describes where those numbers sit in pitch and time, which is all
    the browser needs to place them on its own timeline.
    """
    predictor = get_predictor()
    audio = np.asarray(audio, dtype=np.float32)

    # Browsers are asked for a 22050 Hz capture and essentially always
    # honour it; this is the fallback for one that doesn't, so a wrong
    # sample rate shows up as a little extra latency rather than as a
    # transcription that is confidently a few semitones off.
    if sr != predictor.sr_audio:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=predictor.sr_audio)

    hcqt, _, _ = audio_to_hcqt(audio, predictor.sr_audio, predictor.param_hcqt,
                               tuning=LIVE_TUNING)
    x = torch.as_tensor(hcqt, dtype=torch.float32,
                        device=predictor.lightning.device).unsqueeze(0)
    with _infer_lock, torch.no_grad():
        out = predictor.lightning.model(x)
    outputs = {head: torch.sigmoid(out[head])[0].float().cpu().numpy() for head in HEADS}

    n_pitches, n_frames = outputs["note"].shape
    axes = {
        "n_frames": int(n_frames),
        "n_pitches": int(n_pitches),
        "min_pitch": MIDI_OFFSET,
        # Frame f of the returned arrays covers the audio centred on
        # sample f * hop_length of the window that was sent — that, plus
        # the window's own position, is how the browser lines successive
        # windows up with each other.
        "hop_length": int(predictor.param_hcqt.hop_length),
        "sample_rate": int(predictor.sr_audio),
    }
    return outputs, axes


def encode_posteriogram(array):
    """A posteriogram as base64, one byte per (pitch, frame), row-major.

    Probabilities are only ever compared against a two-decimal threshold
    and painted, so 8 bits is well past enough — and it keeps a window's
    response at ~25KB per head instead of ~100KB as raw float32, or
    several times that as JSON numbers.
    """
    quantized = np.round(np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0) * 255.0)
    return base64.b64encode(quantized.astype(np.uint8).tobytes()).decode("ascii")
