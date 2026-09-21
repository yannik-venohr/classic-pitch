import numpy as np
import librosa
import pretty_midi
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

############# HCQT ############
def audio_to_hcqt(audio, sr_audio, param_hcqt, tuning=None):
    """
        Compute Harmonic CQT

        Parameters
        ----------
        audio_v : np.array
            mono audio signal as np.array
        sr_audio : int
            sampling rate of the audio signal
        param_hcqt : dict
            configuration parameters for the Harmonic CQT
        tuning : float or None
            deviation from A=440 in fractions of a semitone, used to
            shift the filterbank. None (the default) estimates it from
            `audio`, which is what whole-file transcription wants. Pass
            a number to pin the grid instead — see webapp/shared/live.py
            for why a short window is better off with a fixed one.

        Returns

        -------
        data_3m : np.array
            3D array of Harmonic CQT
        time_sec_v : np.array
            corresponding time [in sec] of analysis windows
        frequency_hz_v : np.array
            corresponding frequency [in Hz] of 1st harmonic bin
    """
    fmin = float(librosa.midi_to_hz(param_hcqt.min_note))
    tuning_est = librosa.estimate_tuning(y=audio) if tuning is None else tuning
    fmin_tuned = fmin * 2**(tuning_est / param_hcqt.bins_per_octave)

    hcqt_list = []
    min_time_frames = float('inf')
    for h in param_hcqt.harmonics:
        A_m = np.abs(librosa.cqt(y=audio, sr=sr_audio,
                                fmin=h*fmin_tuned,
                                hop_length=param_hcqt.hop_length,
                                bins_per_octave=param_hcqt.bins_per_octave,
                                n_bins=param_hcqt.n_bins,
                                tuning=0.0,
                                ))
        hcqt_list.append(A_m)
        min_time_frames = min(min_time_frames, A_m.shape[1])
    
    # Trim all HCQT arrays to the smallest frame length
    hcqt_list = [A[:, :min_time_frames] for A in hcqt_list]
    data_3m = np.stack(hcqt_list, axis=0)

    n_times = data_3m.shape[2]
    time_sec_v = librosa.frames_to_time(np.arange(n_times),
                                            sr=sr_audio,
                                            hop_length=param_hcqt.hop_length)
    
    frequency_hz_v = librosa.cqt_frequencies(n_bins=param_hcqt.n_bins,
                                                    fmin=fmin_tuned,
                                                    bins_per_octave=param_hcqt.bins_per_octave)

    return data_3m, time_sec_v, frequency_hz_v

def get_hcqt_times(param_hcqt, n_frames):
    """Get time stamps for HCQT frames.

    Parameters
    ----------
    param_hcqt : dict
        configuration parameters for the Harmonic CQT
    n_frames : int
        number of HCQT frames
    Returns
    -------
    time_sec_v : np.array
        corresponding time [in sec] of analysis windows
    """
    time_sec_v = librosa.frames_to_time(np.arange(n_frames),
                                        sr=param_hcqt.sr,
                                        hop_length=param_hcqt.hop_length)
    return time_sec_v

########### MIDI AND NOTE EVENTS ############
def note_events_to_midi(note_events, output_file=None, instrument_name='Acoustic Grand Piano'):
    """
    Convert a list of note events to a MIDI file using PrettyMIDI.

    Parameters:
        note_events (list of tuples): Each tuple contains
            (start_time, end_time, midi_number, amplitude),
            where amplitude is a float from 0.0 to 1.0.
        output_file (str): Path to save the generated MIDI file.
        instrument_name (str): Name of the instrument for the MIDI track.
    """
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()

    # Create an Instrument instance
    program = pretty_midi.instrument_name_to_program(instrument_name)
    instrument = pretty_midi.Instrument(program=program)

    # Add note events
    for start, end, pitch, amplitude in note_events:
        velocity = int(np.clip(amplitude * 127, 0, 127))  # Convert amplitude to MIDI velocity
        note = pretty_midi.Note(velocity=velocity, pitch=int(pitch),
                                start=start, end=end)
        instrument.notes.append(note)

    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(instrument)

    # Write the MIDI file to disk
    if output_file is not None:
        midi.write(output_file)

    return midi

def midi_file_to_note_events(midi):
    """Convert a midi file to a list of note events. Drum tracks are ignored.

    Args:
        midi: Path to a midi file or pretty_midi.PrettyMIDI object

    Returns:
        score: A list of note events where each note is specified as
               [start, end, pitch]
    """
    midi_data = pretty_midi.PrettyMIDI(str(midi))

    note_events = []
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch
            amplitude = note.velocity / 127.0
            note_events.append([start, end, pitch, amplitude])

    return note_events

def midi_file_to_note_events_with_pedal(midi_file):
    """Convert a midi file to a list of note events, taking into account the sustain pedal
    
    Args:
        midi_file: Path to a midi file or pretty_midi.PrettyMIDI object
    
    Returns:
        score: A list of note events where each note is specified as
               [start, end, pitch]
    """
    midi_data = pretty_midi.PrettyMIDI(str(midi_file))

    note_events = []
    
    for instrument in midi_data.instruments:

        sustain_times = [] # List of sustain pedal press & release times
        is_pedal_on = False
        for control in instrument.control_changes:
            if control.number == 64:  # Sustain pedal (CC 64)
                is_current_pedal_on = (control.value >= 64) # threshold for pedal pressed
                if not is_pedal_on and is_current_pedal_on:
                    # Pedal just got pressed
                    time_pedal_on = control.time
                    is_pedal_on = True
                elif is_pedal_on and not is_current_pedal_on:
                    # Pedal just got released
                    # Update the sustain times (store press and release)
                    sustain_times.append((time_pedal_on, control.time))
                    is_pedal_on = False

        for note in instrument.notes:
            start, end, pitch = note.start, note.end, note.pitch
            # Adjust end time based on sustain pedal activity
            for pedal_start, pedal_end in sustain_times:
                    if start < pedal_end and end <= pedal_start:
                        end = pedal_end  
                        break # No need to check further pedal activity
                    

            note_events.append([start, end, pitch])

    return note_events

def midi_files_to_note_events(midi_dir: Path):
    all_note_events = []
    for midi in midi_dir.glob('*.mid'):
        if 'kick' in midi.name or 'snare' in midi.name or 'ride' in midi.name:
            continue
        note_events = midi_file_to_note_events(midi)
        all_note_events.extend(note_events)
    return all_note_events

######### NOTE EVENT PROCESSING #########
def filter_duplicate_onsets(note_events):
    """Filter duplicate (note events with same onset and pitch) note onsets from a list of note events. 
    The longer duration note is kept.

    Args:
        note_events: A list of note events where each note is specified as
                     [start, end, pitch]
    Returns:
        filtered_events: A list of note events with duplicates removed.
    """
    unique_events = {}
    for start, end, pitch in note_events:
        key = (start, pitch)
        duration = end - start
        if key not in unique_events or duration > (unique_events[key][1] - unique_events[key][0]):
            unique_events[key] = (start, end, pitch)
    filtered_events = list(unique_events.values())
    return filtered_events

def filter_duplicate_onsets_by_frame(note_events, frame_size):
    """Filter duplicate note events considering a given frame size.
    Two notes with the same pitch are considered duplicates if their onsets
    fall within the same frame. The longer duration note is kept.

    Args:
        note_events: A list of note events where each note is specified as
                     [start, end, pitch]
        frame_size: Frame size in seconds.
    Returns:
        filtered_events: A list of note events with frame-level duplicates removed.
    """
    unique_events = {}
    for start, end, pitch in note_events:
        frame_idx = int(start // frame_size)
        key = (frame_idx, pitch)
        duration = end - start
        if key not in unique_events or duration > (unique_events[key][1] - unique_events[key][0]):
            unique_events[key] = (start, end, pitch)
    return list(unique_events.values())

def filter_invalid_intervals(note_events, verbose=True):
    """Filter note events with invalid intervals (end time <= start time).

    Args:
        note_events: A list of note events where each note is specified as
                     [start, end, pitch]
    Returns:
        filtered_events: A list of note events with invalid intervals removed.
    """
    filtered_events = [(s,e,p) for (s,e,p) in note_events if e > s]
    if verbose and len(filtered_events) < len(note_events):
        print(f"Filtered {len(note_events) - len(filtered_events)} invalid intervals")
    return filtered_events

def merge_overlapping_notes(note_events):
    """Merge overlapping note events of the same pitch into a single spanning note.

    Retriggers — where a new onset for a pitch occurs before the previous note ends —
    are collapsed into one continuous note from the earliest onset to the latest offset.

    Args:
        note_events: A list of note events where each note is specified as
                     [start, end, pitch]
    Returns:
        merged_events: A list of note events with overlapping same-pitch notes merged.
    """
    from collections import defaultdict

    by_pitch = defaultdict(list)
    for event in note_events:
        start, end, pitch = event[0], event[1], event[2]
        amplitude = event[3] if len(event) > 3 else None
        by_pitch[pitch].append((start, end, amplitude))

    has_amplitude = any(a is not None for intervals in by_pitch.values() for _, _, a in intervals)

    merged = []
    for pitch, intervals in by_pitch.items():
        intervals.sort()
        current_start, current_end, current_amp = intervals[0]
        for start, end, amp in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
                if has_amplitude:
                    current_amp = max(current_amp, amp)
            else:
                entry = (current_start, current_end, pitch, current_amp) if has_amplitude else (current_start, current_end, pitch)
                merged.append(entry)
                current_start, current_end, current_amp = start, end, amp
        entry = (current_start, current_end, pitch, current_amp) if has_amplitude else (current_start, current_end, pitch)
        merged.append(entry)

    return merged

######### NOTE EVENTS TO ARRAYS #########
def note_events_to_binary_mpe_array(note_events, times, min_note, n_pitch_bins, add_onsets=False):
    """
    Convert a list of note events to a binary multi-pitch array.

    Parameters
    ----------
    note_events : list
        List of note events in the format (start_time, end_time, pitch (midi)).
    times : np.array
        Array of time stamps in seconds.
    n_pitch_bins : int
        Number of pitch bins.
    min_note : int
        Minimum note in MIDI format.
    add_onsets : bool, optional
        If True, onset frames are marked with 2 and active frames with 1.
        Default is False.
    Returns
    -------
    mpe_array : np.ndarray
        Multi-pitch array of shape (n_pitch_bins, len(times)).

    """
    # Remove velocity information if present
    note_events = [(s,e,p) for (s,e,p, *_) in note_events]


    # Define pitch bins
    pitch_bins = np.arange(min_note, min_note + n_pitch_bins)
    # Initialize the mpe array with zeros
    mpe_array = np.zeros((n_pitch_bins,  len(times)))

    for start, end, pitch in note_events:
        # Convert start and end times to time steps
        start_idx = np.argmin(np.abs(start-times))
        end_idx = np.argmin(np.abs(end-times))
        pitch_idx = np.argmin(np.abs(pitch-pitch_bins))

        # Set the binary array for the active pitch range
        mpe_array[pitch_idx, start_idx:end_idx] = 1
        if add_onsets:
            mpe_array[pitch_idx, start_idx] = 2

    return mpe_array

def note_events_to_onset_array(note_events, times, min_note, n_pitch_bins, smear_radius=None):
    """
    Convert a list of note events to a Gaussian-smeared onset array.

    Parameters
    ----------
    note_events : list
        List of note events in the format (start_time, end_time, pitch (midi)).
    times : np.array
        Array of time stamps in seconds.
    n_pitch_bins : int
        Number of pitch bins.
    min_note : int
        Minimum note in MIDI format.
    smear_radius : int  
        Radius of Gaussian smear in frames (roughly 3*sigma for practical purposes).

    Returns
    -------
    onset_array : np.ndarray
        Smeared onset array of shape (n_pitch_bins, len(times)).
    """
    pitch_bins = np.arange(min_note, min_note + n_pitch_bins)
    onset_array = np.zeros((n_pitch_bins, len(times)))

    # First, create binary onset array
    for start, end, pitch in note_events:
        start_idx = np.argmin(np.abs(start - times))
        pitch_idx = np.argmin(np.abs(pitch - pitch_bins))
        if 0 <= pitch_idx < n_pitch_bins and 0 <= start_idx < len(times):
            onset_array[pitch_idx, start_idx] = 1.0

    if smear_radius is not None:
        # Apply Gaussian filter along the time axis (axis=1)
        sigma = smear_radius / 3.0  # Approximate smear_radius ≈ 3*sigma
        onset_array = gaussian_filter1d(onset_array, sigma=sigma, axis=1, mode='nearest')

    return onset_array

####### MPE ARRAY PROCESSING ########

def binary_mpe_array_to_pitch_class(pitch_array):
    """
    Convert a binary multi-pitch array to a pitch class array.

    Parameters
    ----------
    array : np.ndarray
        Binary multi-pitch array of shape (n_pitch_bins, n_time_steps).

    Returns
    -------
    pitch_class_array : np.ndarray
        Pitch class array of shape (12, n_time_steps).
    """
    n_pitch_bins = pitch_array.shape[0]
    pitch_classes = np.arange(n_pitch_bins) % 12
    pitch_class_array = np.zeros((12, pitch_array.shape[1]), dtype=bool)

    for pc in range(12):
        rows = pitch_array[pitch_classes == pc]
        if len(rows):
            pitch_class_array[pc] = rows.any(axis=0)

    return pitch_class_array.astype(int)

######## NOTE EVENT POST-PROCESSING ############
def time_to_frame(time, frame_size):
    return int(np.floor(time / frame_size))


# def remove_duplicate_onsets_intervals_pitches(ref_intervals, ref_pitches, frame_size):
#     """
#     Instrument-agnostic, frame-based duplicate onset removal, more efficient than pretty_midi function above
#     """
#     pitch_onset_set = set()
#     reduced_intervals = []
#     reduced_pitches = []
#     for (start, end), pitch in zip(ref_intervals, ref_pitches, strict=True):
#         if frame_size is not None:
#             onset = time_to_frame(start, frame_size)
#         else:
#             onset = start
#         if (pitch, onset) in pitch_onset_set: continue
#         pitch_onset_set.add((pitch, onset))
#         reduced_intervals.append((start, end))
#         reduced_pitches.append(pitch)
#     return np.array(reduced_intervals), np.array(reduced_pitches)
