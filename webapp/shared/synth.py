"""Rendering transcribed/annotated MIDI to an audio track for the crossfader."""
import numpy as np
import soundfile as sf

# The synthesized MIDI still has to cut through a full recording when the
# crossfade slider is toward it, so it is matched a little *above* the
# original's loudness rather than exactly to it.
MIDI_LOUDNESS_BOOST = 1.3


def synthesize_midi(midi, fs, attack=0.006, release=0.08, peak_percentile=99.5, target_peak=0.9):
    """Additive sine synth with a short per-note attack/release envelope.

    pretty_midi's built-in ``synthesize()`` fades notes out but not in, so
    every note onset is an instantaneous jump to full amplitude; when
    several notes start together (a chord) those jumps stack into an
    audible click. It also hard-normalizes the whole track by its single
    loudest sample, so one such click ends up dictating the loudness of
    the entire piece. Fixed here with a short raised attack ramp and a
    percentile-based (rather than single-max) normalization.
    """
    notes = [
        (n.start, n.end, n.pitch, n.velocity)
        for inst in midi.instruments if not inst.is_drum
        for n in inst.notes
    ]
    if not notes:
        return np.zeros(1, dtype=np.float32)

    end_time = max(n[1] for n in notes)
    total_samples = int(fs * (end_time + release)) + 1
    out = np.zeros(total_samples, dtype=np.float64)
    attack_len = max(1, int(fs * attack))
    release_len = max(1, int(fs * release))

    for start, end, pitch, velocity in notes:
        s = int(fs * start)
        e = max(s + 1, int(fs * end))
        n = e - s
        freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        wave = np.sin(2 * np.pi * freq * np.arange(n) / fs)

        env = np.ones(n)
        a = min(attack_len, n)
        if a > 0:
            env[:a] = np.minimum(env[:a], np.linspace(0.0, 1.0, a, endpoint=False))
        r = min(release_len, n)
        if r > 0:
            env[n - r:] = np.minimum(env[n - r:], np.linspace(1.0, 0.0, r, endpoint=True))

        out[s:e] += env * wave * (velocity / 127.0)

    abs_out = np.abs(out)
    loud = abs_out[abs_out > 1e-6]
    ref = float(np.percentile(loud, peak_percentile)) if loud.size else 0.0
    if ref > 1e-6:
        out = out / ref * target_peak

    # A hard np.clip here would flatten (distort) whatever still pokes
    # above 1.0 after the percentile-based scaling above — and since that's
    # a whole loud chord's worth of samples, not one freak spike, it reads
    # as an audible click/buzz right at the loudest moments. A pure linear
    # scale-down instead only ever changes overall volume, never shape.
    true_peak = float(np.max(np.abs(out))) if out.size else 0.0
    if true_peak > 0.999:
        out = out / true_peak * 0.999

    return out.astype(np.float32)


def audio_duration(audio_path):
    """Length of a recording in seconds, without decoding it."""
    info = sf.info(str(audio_path))
    return info.frames / info.samplerate


def render_midi_track(midi, audio_path, out_path):
    """Synthesize ``midi`` into a wav at ``out_path``, aligned to ``audio_path``.

    Padded/trimmed to the original recording's length so the two <audio>
    elements the frontend crossfades between share one timeline, and
    loudness-matched to it: peak-normalizing the synthesized MIDI alone
    isn't enough, since a sparse sine-synth track can have the same peak
    as a dense recording while sounding much quieter on average, which
    would leave the crossfade inaudible until the slider is almost all the
    way over.
    """
    info = sf.info(str(audio_path))
    sr = info.samplerate
    n_samples = info.frames

    midi_audio = synthesize_midi(midi, sr)
    if len(midi_audio) < n_samples:
        midi_audio = np.pad(midi_audio, (0, n_samples - len(midi_audio)))
    else:
        midi_audio = midi_audio[:n_samples]

    orig_samples, _ = sf.read(str(audio_path), dtype="float32", always_2d=False)
    orig_mono = orig_samples.mean(axis=1) if orig_samples.ndim > 1 else orig_samples
    orig_rms = float(np.sqrt(np.mean(np.square(orig_mono)))) if orig_mono.size else 0.0
    midi_rms = float(np.sqrt(np.mean(np.square(midi_audio)))) if midi_audio.size else 0.0
    if orig_rms > 1e-6 and midi_rms > 1e-6:
        midi_audio = midi_audio * (orig_rms * MIDI_LOUDNESS_BOOST / midi_rms)

    peak = float(np.max(np.abs(midi_audio))) if midi_audio.size else 0.0
    if peak > 0.98:
        midi_audio = midi_audio / peak * 0.98
    sf.write(str(out_path), midi_audio, sr)

    return sr, n_samples


def notes_from_midi(midi):
    """pretty_midi object -> the frontend's note list, sorted by onset."""
    notes = [
        {
            "start": round(float(note.start), 4),
            "end": round(float(note.end), 4),
            "pitch": int(note.pitch),
            "velocity": int(note.velocity),
        }
        for inst in midi.instruments
        for note in inst.notes
    ]
    notes.sort(key=lambda n: n["start"])
    return notes
