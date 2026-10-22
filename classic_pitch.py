from types import SimpleNamespace
from pathlib import Path
import time
import numpy as np
import librosa
import torch
from src.classic_pitch_lightning import ClassicPitchLightning
from src.data.utils import audio_to_hcqt, merge_overlapping_notes, note_events_to_midi
from src.postprocess.postprocess import output_to_note_events


CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_VARIANT = "classicPitchV2_large"


def available_variants():
	"""Names of the checkpoints in checkpoints/ (file name without '.ckpt'), sorted."""
	return sorted(p.stem for p in CHECKPOINT_DIR.glob("*.ckpt"))


def variant_label(variant):
	"""A checkpoint name for display: 'classicPitch_extra_small' -> 'classicPitch (extra small)'."""
	head, _, rest = variant.partition("_")
	return f"{head} ({rest.replace('_', ' ')})" if rest else head


def default_device():
	"""The fastest device available here: CUDA, then Apple's MPS, then the CPU."""
	if torch.cuda.is_available():
		return 'cuda'
	if torch.backends.mps.is_available():
		return 'mps'
	return 'cpu'


class ClassicPitch():
	def __init__(self, device=None, onset_thresh=0.4, frame_thresh=0.4, variant=DEFAULT_VARIANT, merge_notes=False):
		"""ClassicPitch class for the multi-pitch estimation model.
		Parameters
		----------
		device : str or None
			Device to run the model on. Options are 'cpu', 'cuda' or 'mps';
			None picks the fastest one available (see default_device()).
		variant : str
			Checkpoint to load, named after the file in checkpoints/ without
			the extension, e.g. 'classicPitch_large' or 'classicPitch_small'.
			See available_variants().
		onset_thresh : float
			Threshold for onset detection during decoding.
		frame_thresh : float
			Threshold for frame activation during decoding.
		merge_notes : bool
			Whether to merge overlapping notes.
		"""

		self.device = device or default_device()
		self.variant = variant
		self.onset_thresh = onset_thresh
		self.frame_thresh = frame_thresh
		self.merge_notes = merge_notes
		self.sr_audio = 22050
		self.param_hcqt = SimpleNamespace(
			min_note=24,  # C1 in MIDI
			harmonics=[0.5, 1, 2, 3, 4, 5],
			hop_length=512,
			n_bins=216,  # 6 octaves with 36 bins per octave
			bins_per_octave=36,  # 3 bins per semitone
		)

		checkpoint = CHECKPOINT_DIR / f"{variant}.ckpt"

		self.lightning = ClassicPitchLightning.load_from_checkpoint(checkpoint, map_location=self.device, weights_only=False)
		self.lightning.eval()
		print(f"Loaded checkpoint '{variant}' on device '{self.device}'.")

	def predict(self, audio_file, verbose=False, first_5_min=False, max_seconds=None):
		"""Predict the multi-pitch of an audio file.
		Parameters
		----------
		audio_file : str
			Path to the audio file.
		first_5_min : bool
			Whether to only transcribe the first 5 minutes of the audio.
		max_seconds : float or None
			Only transcribe this much of the start of the audio (the shorter
			of this and first_5_min applies when both are given).

		Returns
		----------
		midi : pretty_midi.PrettyMIDI
			Decoded MIDI object.
		outputs : dict
			Raw model outputs with keys 'note' and 'onset', each an np.ndarray
			of shape (n_pitch_bins, timeFrames) with values in [0, 1].
		times : np.ndarray
			Corresponding time stamps in seconds.
		"""
		def log(msg, t0=None):
			if verbose:
				elapsed = f"  ({time.perf_counter() - t0:.2f}s)" if t0 is not None else ""
				print(f"[classic_pitch] {msg}{elapsed}")

		t_total = time.perf_counter()

		log(f"Loading audio: {audio_file}")
		t = time.perf_counter()
		duration = 5 * 60 if first_5_min else None
		if max_seconds is not None:
			duration = max_seconds if duration is None else min(duration, max_seconds)
		audio, _ = librosa.load(audio_file, sr=self.sr_audio, mono=True, duration=duration)
		log(f"Audio loaded ({len(audio)/self.sr_audio:.1f}s of audio)", t)

		log("Computing HCQT...")
		t = time.perf_counter()
		hcqt, times, _ = audio_to_hcqt(audio, self.sr_audio, self.param_hcqt)
		log(f"HCQT computed: shape={hcqt.shape}, {len(times)} frames", t)

		log("Running model inference...")
		t = time.perf_counter()
		outputs = self.lightning.predict_full_track(hcqt)
		log(f"Model inference done", t)

		log("Decoding note events...")
		t = time.perf_counter()
		note_events = output_to_note_events(
			outputs,
			times,
			onset_threshold=self.onset_thresh,
			frame_threshold=self.frame_thresh,
		)

		log(f"Decoded {len(note_events)} note events", t)

		if self.merge_notes:
			note_events = merge_overlapping_notes(note_events)
			log(f"{len(note_events)} note events after merging overlaps", t)
		midi = note_events_to_midi(note_events)
		log(f"Total predict time", t_total)
		return midi, outputs, times



if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--audio', type=str, default='data/example.wav')
	parser.add_argument('--device', type=str, default=None,
						help="'cpu', 'cuda' or 'mps' (default: the fastest available)")
	parser.add_argument('--output', type=str, default='data/test_transcription')
	parser.add_argument('--variant', type=str, default=DEFAULT_VARIANT,
						help=f'Checkpoint to use (available: {", ".join(available_variants())})')
	parser.add_argument('--first-5-min', action='store_true', help='Only transcribe the first 5 minutes of the audio')
	args = parser.parse_args()

	classic_pitch = ClassicPitch(device=args.device, variant=args.variant)
	midi, outputs, times = classic_pitch.predict(args.audio, first_5_min=args.first_5_min)
	midi.write(f'{args.output}.mid')
	np.savez(f'{args.output}.npz', times=times, **outputs)
