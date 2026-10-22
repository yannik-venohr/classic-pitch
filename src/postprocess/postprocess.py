import numpy as np
import scipy.signal
from typing import List, Tuple

MIDI_OFFSET = 24  # MIDI pitch of the lowest frequency bin


def output_to_note_events(
    model_output: dict,
    times: list,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
    min_frames: int = 2,
    frames_below_threshold: int = 5,
    midi_offset: int = MIDI_OFFSET,
) -> List[Tuple[float, float, int, float]]:
    """Decode raw model output to note events.

    A note starts at every local maximum of the onset activation above onset_threshold.
    It lasts while the frame activation stays above frame_threshold, but at most until the
    next onset of the same pitch.

    Args:
        model_output: Dict with 'note' and 'onset' arrays of shape (n_freqs, n_times).
        times: Time in seconds for each frame index.
        onset_threshold: Minimum onset activation amplitude to be considered an onset.
        frame_threshold: Minimum frame activation for a note to remain "on".
        min_frames: Notes of min_frames frames or shorter are discarded.
        frames_below_threshold: Number of consecutive sub-threshold frames tolerated before ending a note.
        midi_offset: MIDI pitch of the lowest frequency bin.

    Returns:
        list of tuples [(start_time_s, end_time_s, pitch_midi, amplitude)]
    """
    frames = model_output['note']
    onsets = model_output['onset']
    n_frames = frames.shape[1]

    # keep only local maxima
    peak_thresh_mat = np.zeros(onsets.shape)
    peaks = scipy.signal.argrelmax(onsets, axis=1)
    peak_thresh_mat[peaks] = onsets[peaks]

    note_events = []
    for freq_idx in range(frames.shape[0]):
        pitch_onsets = np.where(peak_thresh_mat[freq_idx] >= onset_threshold)[0]
        for j, note_start_idx in enumerate(pitch_onsets):
            if note_start_idx >= n_frames - 1:
                continue

            # note ends at the next onset for this pitch, or at the frame boundary
            next_onset = int(pitch_onsets[j + 1]) if j + 1 < len(pitch_onsets) else n_frames

            # scan forward, tolerating short dips below threshold
            i = note_start_idx + 1
            k = 0
            while i < min(next_onset, n_frames - 1) and k < frames_below_threshold:
                if frames[freq_idx, i] < frame_threshold:
                    k += 1
                else:
                    k = 0
                i += 1
            i -= k  # trim trailing sub-threshold frames

            if i - note_start_idx <= min_frames:
                continue

            amplitude = np.mean(frames[freq_idx, note_start_idx:i])
            note_events.append((times[note_start_idx], times[i], freq_idx + midi_offset, amplitude))

    return note_events
