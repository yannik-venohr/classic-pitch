from types import SimpleNamespace
import time
import numpy as np
import librosa
from src.classic_pitch import ClassicPitchLightning
from src.data.utils import audio_to_hcqt, merge_overlapping_notes, note_events_to_midi
from src.postprocess.postprocess import output_to_note_events
import yt_dlp


class ClassicPitch():
	def __init__(self, device='mps', onset_thresh=0.4, frame_thresh=0.4, variant='large', merge_notes=False):
		"""Predictor class for the multi-pitch estimation model.
		Parameters
		----------
		device : str
			Device to run the model on. Options are 'cpu', 'cuda' or 'mps'.
		variant : str
			Model variant to use. Options are 'large', 'small', 'extra_small' or 'pitch_class'.
		onset_thresh : float
			Threshold for onset detection during decoding.
		frame_thresh : float
			Threshold for frame activation during decoding.
		merge_notes : bool
			Whether to merge overlapping notes.
		"""

		self.device = device
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

		if variant == "small":
			checkpoint = "checkpoints/classic_pitch_small.ckpt"
		elif variant == "extra_small":
			checkpoint = "checkpoints/classic_pitch_extra_small.ckpt"
		elif variant == "pitch_class":
			checkpoint = "checkpoints/classic_pitch_with_pitch_class.ckpt"
		else:
			checkpoint ="checkpoints/classic_pitch_large.ckpt"

		self.lightning = ClassicPitchLightning.load_from_checkpoint(checkpoint, map_location=self.device, weights_only=False)
		self.lightning.eval()

	def get_audio_from_youtube(self, url):
		output_path = "data/temp_audio"
		ydl_opts = {
			'format': 'bestaudio/best',
			'postprocessors': [{
				'key': 'FFmpegExtractAudio',
				'preferredcodec': 'wav',
				'preferredquality': '192',
			}],
			'noplaylist': True,  
			'outtmpl': output_path,
		}
		with yt_dlp.YoutubeDL(ydl_opts) as ydl:
			ydl.download([url])
	
		return output_path + ".wav"
	
	def predict(self, audio_file, verbose=False):
		"""Predict the multi-pitch of an audio file.
		Parameters
		----------
		audio_file : str
			Path to the audio file.

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
				print(f"[predictor] {msg}{elapsed}")

		t_total = time.perf_counter()

		log(f"Loading audio: {audio_file}")
		t = time.perf_counter()
		audio, _ = librosa.load(audio_file, sr=self.sr_audio, mono=True)
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
	parser.add_argument('--audio', type=str, default='data/test.wav')
	parser.add_argument('--device', type=str, default='mps')
	parser.add_argument('--output', type=str, default='data/test_transcription')
	args = parser.parse_args()

	predictor = ClassicPitch(device=args.device)
	midi, outputs, times = predictor.predict(args.audio)
	midi.write(f'{args.output}.mid')
	np.savez(f'{args.output}.npz', times=times, **outputs)
