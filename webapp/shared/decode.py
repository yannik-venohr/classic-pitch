"""Re-decoding a cached transcription without re-running the model.

The model's real output is two posteriograms — an onset one and a frame
("note") one, each a (n_pitches, n_frames) array of probabilities.
Turning those into note events is a separate, cheap thresholding step
(src/postprocess/postprocess.py). Keeping the posteriograms next to the
cached transcription is therefore what lets the UI expose the decoding
thresholds as live controls: changing one costs a second of decoding
plus a re-render, not a full inference pass.

This module owns that stored form: writing it, reading it back, turning
it into images for the UI, and running the decode.
"""
import numpy as np
from PIL import Image

from src.data.utils import merge_overlapping_notes, note_events_to_midi
from src.postprocess.postprocess import MIDI_OFFSET, output_to_note_events
from webapp.shared.synth import notes_from_midi, render_midi_track

# Files written into an item's cache directory, alongside meta.json.
POSTERIOGRAM_ARRAYS = "posteriograms.npz"
POSTERIOGRAM_IMAGES = {"onset": "onset.png", "frame": "frame.png"}
MIDI_AUDIO = "midi.wav"

# Every file the /media route must be willing to serve for an app that
# has the model panel turned on.
MEDIA_FILES = tuple(POSTERIOGRAM_IMAGES.values())

# What a freshly transcribed item is decoded at, and what the UI's
# sliders start from. Matches predictor.ClassicPitch's own defaults.
DEFAULT_DECODING = {"onset_threshold": 0.4, "frame_threshold": 0.4, "merge_notes": False}

# The sliders' range. Thresholds of exactly 0 or 1 are degenerate (every
# local maximum is an onset / nothing is), so the ends are pulled in.
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95

# Posteriogram images are tinted rather than grayscale so they read as
# the same two quantities the piano roll already draws in these colours:
# note bodies in blue, onsets in yellow. The floor is .roll-container's
# own background, so an all-but-silent posteriogram blends into the page.
POSTERIOGRAM_BACKGROUND = (24, 26, 32)
POSTERIOGRAM_COLORS = {"onset": (255, 211, 90), "frame": (90, 169, 255)}

# How activation probability maps to display brightness, per head.
#
# `gamma` (<1) lifts the midrange: activations cluster low, and a linear
# ramp would render almost the whole image as background and hide exactly
# the weak-but-real detail the threshold sliders exist to explore.
#
# That lift alone, though, also promotes the model's noise floor into a
# visible haze — 0.1 would come out a quarter of full brightness. `knee`
# and `gate` fade that lift back in over [0, knee] instead of applying it
# from zero, so the dirt stays dark while everything above the knee is
# left exactly as it was.
#
# `gate` is how hard the fade bites: 1 is a gentle S-curve, higher
# values push more of the damping down towards zero. Damped, never
# clipped — THRESHOLD_MIN is 0.05, so the picture has to keep showing
# something the sliders can still select down there. `knee` of 0 turns
# the damping off and leaves a head on the plain gamma lift.
POSTERIOGRAM_CURVES = {
    "onset": {"gamma": 0.6, "knee": 0.2, "gate": 2},
    "frame": {"gamma": 0.6, "knee": 0.2, "gate": 2},
}

# Bumped whenever the curves above change, so already-cached items get
# their images redrawn instead of being stranded on the old rendering
# (see refresh_posteriogram_images). It also cache-busts the URLs.
POSTERIOGRAM_RENDER_VERSION = 3


def clean_decoding(data):
    """A decoding spec from untrusted request JSON, clamped to sane values."""
    def threshold(key):
        try:
            value = float(data[key])
        except (KeyError, TypeError, ValueError):
            return DEFAULT_DECODING[key]
        if not np.isfinite(value):
            return DEFAULT_DECODING[key]
        return round(min(THRESHOLD_MAX, max(THRESHOLD_MIN, value)), 3)

    return {
        "onset_threshold": threshold("onset_threshold"),
        "frame_threshold": threshold("frame_threshold"),
        "merge_notes": bool(data.get("merge_notes", DEFAULT_DECODING["merge_notes"])),
    }


def _tone_curve(x, curve):
    """Activation probability -> display brightness. See POSTERIOGRAM_CURVES."""
    knee, gate, gamma = curve["knee"], curve["gate"], curve["gamma"]
    x = np.clip(x, 0.0, 1.0)
    lifted = x ** gamma
    if knee <= 0:
        return lifted
    # Smoothstep, which is flat at both ends — so the damping fades in
    # and out without a kink, and no contour line appears at the knee
    # itself. Above the knee the gate is exactly 1 and `lifted` stands.
    u = np.clip(x / knee, 0.0, 1.0)
    return lifted * (u * u * (3.0 - 2.0 * u)) ** gate


def _write_posteriogram_image(array, kind, path):
    """One posteriogram as a tinted PNG, one pixel per (pitch, frame).

    Written at the model's own resolution and scaled in the browser, so
    the same image serves every zoom level. Row 0 is the *highest* pitch,
    matching how the piano roll is drawn — the frontend can then blit a
    pitch range straight out of it without flipping anything.
    """
    intensity = _tone_curve(np.asarray(array, dtype=np.float32), POSTERIOGRAM_CURVES[kind])
    intensity = intensity[::-1, :, None]

    base = np.array(POSTERIOGRAM_BACKGROUND, dtype=np.float32)
    color = np.array(POSTERIOGRAM_COLORS[kind], dtype=np.float32)
    rgb = base + intensity * (color - base)
    Image.fromarray(np.round(rgb).astype(np.uint8), mode="RGB").save(path)


def save_posteriograms(item_dir, item_id, outputs, times):
    """Store the raw model output, and return the meta fields describing it.

    Arrays go to disk as float16: they are probabilities read back only
    to be compared against a two-decimal threshold, so the halved file
    size costs nothing that matters (a 10-minute track is ~45MB at
    float32, ~22MB here).
    """
    onset = np.asarray(outputs["onset"], dtype=np.float32)
    frame = np.asarray(outputs["note"], dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)

    np.savez(
        item_dir / POSTERIOGRAM_ARRAYS,
        onset=onset.astype(np.float16),
        note=frame.astype(np.float16),
        times=times,
    )
    _render_images(item_dir, {"onset": onset, "frame": frame})

    n_pitches, n_frames = frame.shape
    # The images span this many seconds edge to edge — one frame's worth
    # past the last frame's timestamp, since each column is a frame's
    # duration wide rather than an instant.
    frame_seconds = float(times[1] - times[0]) if len(times) > 1 else 0.0
    return {
        "posteriograms": {
            **_image_urls(item_id),
            "n_frames": int(n_frames),
            "min_pitch": MIDI_OFFSET,
            "max_pitch": MIDI_OFFSET + int(n_pitches) - 1,
            "duration": round(n_frames * frame_seconds, 4),
        }
    }


def _render_images(item_dir, arrays):
    for kind, array in arrays.items():
        _write_posteriogram_image(array, kind, item_dir / POSTERIOGRAM_IMAGES[kind])


def _image_urls(item_id):
    """The image URLs plus the render they belong to.

    The version is in the query string as well as the meta, because the
    PNGs are overwritten in place: without it a browser would keep
    showing the old tone curve out of its own cache.
    """
    return {
        "urls": {
            kind: f"/media/{item_id}/{name}?v={POSTERIOGRAM_RENDER_VERSION}"
            for kind, name in POSTERIOGRAM_IMAGES.items()
        },
        "render_version": POSTERIOGRAM_RENDER_VERSION,
    }


def refresh_posteriogram_images(item_dir, item_id, meta):
    """Redraw a cached item's PNGs if POSTERIOGRAM_CURVES has moved on.

    Returns the meta fields to merge, or None when nothing was stale.
    Reads only the stored arrays, so a colour-curve change costs a
    fraction of a second per item rather than another inference pass.
    """
    info = meta.get("posteriograms")
    if not info or info.get("render_version") == POSTERIOGRAM_RENDER_VERSION:
        return None
    loaded = load_posteriograms(item_dir)
    if loaded is None:
        return None
    outputs, _ = loaded
    _render_images(item_dir, {"onset": outputs["onset"], "frame": outputs["note"]})
    return {"posteriograms": {**info, **_image_urls(item_id)}}


def has_posteriograms(item_dir):
    """Whether ``item_dir`` holds everything a re-decode needs.

    Items cached before this existed have a meta.json but no arrays, so
    every caller has to be prepared for the answer to be no.
    """
    return (item_dir / POSTERIOGRAM_ARRAYS).exists() and all(
        (item_dir / name).exists() for name in POSTERIOGRAM_IMAGES.values()
    )


def load_posteriograms(item_dir):
    """``(outputs, times)`` as predict() returned them, or None if not stored."""
    path = item_dir / POSTERIOGRAM_ARRAYS
    if not path.exists():
        return None
    with np.load(path) as data:
        outputs = {"onset": data["onset"].astype(np.float32), "note": data["note"].astype(np.float32)}
        times = data["times"].astype(np.float64)
    return outputs, times


def decode_to_midi(outputs, times, decoding):
    """Posteriograms -> pretty_midi, at the given thresholds.

    Deliberately the same three calls predictor.ClassicPitch.predict
    makes after inference, so a re-decode and the original transcription
    can't come out differently for the same settings.
    """
    note_events = output_to_note_events(
        outputs,
        times,
        onset_threshold=decoding["onset_threshold"],
        frame_threshold=decoding["frame_threshold"],
    )
    if decoding["merge_notes"]:
        note_events = merge_overlapping_notes(note_events)
    return note_events_to_midi(note_events)


def apply_decoding(item_dir, item_id, outputs, times, decoding, audio_path):
    """Decode at ``decoding``, re-render the MIDI track, return meta fields.

    The one path both a fresh transcription and a later threshold change
    go through, so the cached wav always matches the cached note list.
    """
    midi = decode_to_midi(outputs, times, decoding)
    midi_path = item_dir / MIDI_AUDIO
    render_midi_track(midi, audio_path, midi_path)
    return {
        "notes": notes_from_midi(midi),
        "decoding": decoding,
        # The wav is overwritten in place, so the URL alone would let a
        # browser keep playing the previous decoding out of its cache.
        "midi_audio_url": f"/media/{item_id}/{MIDI_AUDIO}?v={midi_path.stat().st_mtime_ns}",
    }
