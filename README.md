# ClassicPitch

Robust audio-to-MIDI transcription for musicology.

## Citation

Code for the paper:

Yannik Venohr and Christof Weiß. Robust Instrument-Agnostic Music Transcription for Western Classical Music. In *Proceedings of the International Society for Music Information Retrieval Conference*, 2026.

```bibtex
@inproceedings{VenohrWeiss2026ClassicPitch,
  author    = {Yannik Venohr and Christof Wei{\ss}},
  title     = {Robust Instrument-Agnostic Music Transcription for Western Classical Music},
  booktitle = {Proceedings of the International Society for Music Information Retrieval Conference},
  year      = {2026}
}
```

## Setup

Requires Python 3.10.

```
pip install -r requirements.txt
```

The YouTube demo needs [`ffmpeg`](https://ffmpeg.org/) on your `PATH`.

`checkpoints/classic_pitch_large.ckpt` is included. The other variants (`small`, `extra_small`, `pitch_class`) will be added by 1 October 2026.

## Usage

### Get started

[`get_started.ipynb`](get_started.ipynb) is the quickest way in: load the model, transcribe a file or YouTube link, listen to the result, and plot the piano roll.

### Python

```python
from predictor import ClassicPitch

predictor = ClassicPitch(device="mps")  # or "cuda" / "cpu"
midi, outputs, times = predictor.predict("data/goodbye.wav")
midi.write("out.mid")
```

### Command line

```
python predictor.py --audio data/goodbye.wav --output out/goodbye --device mps
```

Writes `out/goodbye.mid` and `out/goodbye.npz` (raw model outputs).

`python predict.py` transcribes every `.wav` in `data/` into `data/predictions/`. Change `audio_dir` and `output_dir` in the script to use other folders.

### Web apps

| App | Command | URL |
| --- | --- | --- |
| YouTube demo | `python webapp/demo/app.py` | http://127.0.0.1:5001 |
| Live microphone | `python webapp/live/app.py` | http://127.0.0.1:5004 |

## Structure

```
get_started.ipynb  # walkthrough notebook
predictor.py       # ClassicPitch: load checkpoint, transcribe audio
predict.py         # batch transcription of a folder
src/               # model, feature extraction, postprocessing
webapp/            # local Flask apps
checkpoints/       # model weights
data/              # example audio
```

## License

[CC BY-NC 4.0](LICENSE): free for non-commercial use with attribution.
