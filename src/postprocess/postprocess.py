import numpy as np
import scipy.signal
from typing import List, Tuple

MIDI_OFFSET = 24


def output_to_note_events(
    model_output: dict,
    times: list,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
    min_frames: int = 4,
    frames_below_threshold: int = 5,
) -> List[Tuple[float, float, int, float]]:
    """Decode raw model output to note events.

    Args:
        model_output: Dict with 'note' and 'onset' arrays of shape (n_freqs, n_times).
        times: Time in seconds for each frame index.
        onset_threshold: Minimum onset activation amplitude to be considered an onset.
        frame_threshold: Minimum frame activation for a note to remain "on".
        min_frames: Minimum allowed note length in frames.
        frames_below_threshold: Number of consecutive sub-threshold frames tolerated before ending a note.
        onset_drop_ratio: Fraction of the smaller peak that the valley between two onset peaks must fall
            below for the second peak to count as a new onset. Higher = stricter (fewer re-triggers).
            Set to 1.0 to disable filtering.

    Returns:
        list of tuples [(start_time_s, end_time_s, pitch_midi, amplitude)]
    """
    frames = model_output['note']
    onsets = model_output['onset']
    n_frames = frames.shape[1]

    peak_thresh_mat = np.zeros(onsets.shape)
    peaks = scipy.signal.argrelmax(onsets, axis=1)
    peak_thresh_mat[peaks] = onsets[peaks]

    onset_idx = np.where(peak_thresh_mat >= onset_threshold)
    onset_freq_idx = onset_idx[0]
    onset_time_idx = onset_idx[1]

    # sort descending in time for note-end scan
    order = np.argsort(onset_time_idx)[::-1]
    onset_freq_idx = onset_freq_idx[order]
    onset_time_idx = onset_time_idx[order]

    note_events = []
    for note_start_idx, freq_idx in zip(onset_time_idx, onset_freq_idx):
        if note_start_idx >= n_frames - 1:
            continue

        # find where frames drop below threshold, tolerating short gaps
        i = note_start_idx + 1
        k = 0
        while i < n_frames - 1 and k < frames_below_threshold:
            if frames[freq_idx, i] < frame_threshold:
                k += 1
            else:
                k = 0
            i += 1
        i -= k  # go back to last frame above threshold

        if i - note_start_idx <= min_frames:
            continue

        amplitude = np.mean(frames[freq_idx, note_start_idx:i])
        note_events.append((times[note_start_idx], times[i], freq_idx + MIDI_OFFSET, amplitude))

    return note_events
